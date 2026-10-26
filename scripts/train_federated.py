"""Run federated training simulation for skin lesion classification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.client.client import client_fn_factory
from src.data.dataset import (
    SkinLesionDataset,
    create_synthetic_dataset,
    partition_data_dirichlet,
    partition_data_iid,
    set_seed,
    split_dataset,
)
from src.data.transforms import build_transforms
from src.models.model import build_model
from src.server.strategy import FedAvgWithLogging, weighted_average
from src.utils.config import ensure_dirs, load_config
from src.utils.metrics import compute_metrics, plot_confusion_matrix, plot_metric_history, print_classification_report
from src.utils.training import evaluate


def load_data_for_simulation(config: dict, use_synthetic: bool = False):
    """Load and partition dataset for federated simulation."""
    set_seed(config["random_seed"])
    img_size = config["dataset"]["image_size"]
    num_classes = config["dataset"]["num_classes"]
    num_clients = config["federated"]["num_clients"]

    train_transform, eval_transform = build_transforms(img_size, augment=config["preprocessing"]["train_augmentation"])

    if use_synthetic:
        csv_path = create_synthetic_dataset(
            output_dir=config["dataset"]["processed_dir"],
            num_samples=config["dataset"].get("synthetic_samples", 2000),
            image_size=img_size,
            num_classes=num_classes,
            seed=config["random_seed"],
        )
        image_dir = Path(config["dataset"]["processed_dir"]) / "images"
    else:
        csv_path = Path(config["dataset"]["processed_dir"]) / "HAM10000_metadata.csv"
        image_dir = Path(config["dataset"]["raw_dir"]) / "images"

    full_dataset = SkinLesionDataset(
        csv_path=csv_path,
        image_dir=str(image_dir),
        transform=train_transform,
        label_col="label",
    )
    if not use_synthetic and "lesion_id" not in full_dataset.df:
        raise ValueError("Real-image training requires lesion_id metadata; rerun data preparation")

    train_set, val_set, test_set = split_dataset(
        full_dataset,
        val_ratio=config["preprocessing"]["val_test_split"][0],
        test_ratio=config["preprocessing"]["val_test_split"][1],
        seed=config["random_seed"],
        eval_transform=eval_transform,
    )

    # For FL, we partition only the training set across clients.
    # Each client also gets a slice of the validation set for local evaluation.
    if config["federated"]["iid"]:
        client_train_sets = partition_data_iid(train_set, num_clients, seed=config["random_seed"])
    else:
        client_train_sets = partition_data_dirichlet(
            train_set,
            num_clients,
            alpha=config["federated"]["dirichlet_alpha"],
            seed=config["random_seed"],
        )

    # Give each client a small local validation split from the central validation set
    val_indices = np.arange(len(val_set))
    client_val_indices = np.array_split(val_indices, num_clients)
    client_val_sets = [Subset(val_set, idx.tolist()) for idx in client_val_indices]

    print("Client train set sizes:", [len(ds) for ds in client_train_sets])
    print("Client val set sizes:", [len(ds) for ds in client_val_sets])

    return client_train_sets, client_val_sets, test_set, full_dataset.classes


def get_initial_parameters(model: torch.nn.Module) -> fl.common.Parameters:
    """Serialize initial model parameters for Flower."""
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)


def train_federated(config: dict, use_synthetic: bool = False) -> dict:
    ensure_dirs(config)
    set_seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_datasets, val_datasets, test_set, class_names = load_data_for_simulation(config, use_synthetic)

    num_classes = config["dataset"]["num_classes"]
    num_rounds = config["federated"]["num_rounds"]
    num_clients = config["federated"]["num_clients"]
    fraction_fit = config["federated"]["fraction_fit"]
    fraction_evaluate = config["federated"]["fraction_evaluate"]

    initial_model = build_model(
        config["model"]["backbone"],
        num_classes,
        dropout=config["model"]["dropout"],
        freeze_backbone=config["model"].get("freeze_backbone", True),
    )
    initial_parameters = get_initial_parameters(initial_model)

    round_metrics: list = []

    def log_callback(server_round: int, metrics: dict):
        round_metrics.append({"round": server_round, **metrics})
        print(f"Round {server_round}: {metrics}")

    strategy = FedAvgWithLogging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_clients,
        min_evaluate_clients=max(1, int(num_clients * fraction_evaluate)),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        accept_failures=False,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        log_callback=log_callback,
    )

    client_fn = client_fn_factory(train_datasets, val_datasets, config, device)

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0},
    )

    # Build the final global model from the strategy's latest parameters
    final_model = build_model(
        config["model"]["backbone"],
        num_classes,
        dropout=config["model"]["dropout"],
        freeze_backbone=config["model"].get("freeze_backbone", True),
    )
    final_parameters = strategy.latest_parameters
    if final_parameters is None:
        raise RuntimeError("No aggregated parameters were produced during federated training.")

    final_state_dict = {
        k: torch.tensor(v)
        for k, v in zip(
            final_model.state_dict().keys(),
            fl.common.parameters_to_ndarrays(final_parameters),
        )
    }
    final_model.load_state_dict(final_state_dict)
    final_model = final_model.to(device)

    # Save final global model
    out_dir = Path(config["outputs"]["models_dir"])
    torch.save(final_model.state_dict(), out_dir / "federated_global_best.pt")

    # Evaluate on global test set
    test_loader = DataLoader(test_set, batch_size=config["federated"]["batch_size"], shuffle=False, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(final_model, test_loader, criterion, device)
    print(f"Global test loss: {test_loss:.4f}, Global test accuracy: {test_acc:.4f}")

    # Collect predictions
    final_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = final_model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    metrics["test_loss"] = test_loss
    metrics["test_accuracy"] = test_acc
    if config["differential_privacy"]["enabled"]:
        metrics["privacy_epsilon"] = max(row["max_epsilon"] for row in round_metrics)
        metrics["privacy_delta"] = config["differential_privacy"]["target_delta"]
        metrics["privacy_accountant"] = "rdp"
    print("Federated test metrics:", metrics)
    print_classification_report(all_labels, all_preds, class_names)

    # Save metrics and plots
    plot_dir = Path(config["outputs"]["plots_dir"])
    with open(out_dir / "federated_metrics.json", "w") as f:
        json.dump({"final": metrics, "round_history": round_metrics}, f, indent=2)

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        output_path=str(plot_dir / "federated_confusion_matrix.png"),
        title="Federated Global Model - Confusion Matrix",
    )

    if round_metrics:
        if not config["differential_privacy"]["enabled"]:
            plot_metric_history(
                {"Train Accuracy": [row["train_accuracy"] for row in round_metrics]},
                output_path=str(plot_dir / "federated_accuracy_history.png"),
                title="Federated Training Accuracy per Round",
            )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated skin lesion training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for quick testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_federated(cfg, use_synthetic=args.synthetic)
