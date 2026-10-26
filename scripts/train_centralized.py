"""Train a centralized baseline model on HAM10000 (or synthetic data)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SkinLesionDataset, create_synthetic_dataset, set_seed, split_dataset
from src.data.transforms import build_transforms
from src.models.model import build_model
from src.utils.config import ensure_dirs, load_config
from src.utils.metrics import compute_metrics, plot_confusion_matrix, plot_metric_history, print_classification_report
from src.utils.training import compute_class_weights, evaluate, get_model_params, set_model_params, train_one_epoch


def train_centralized(config: dict, use_synthetic: bool = False) -> dict:
    set_seed(config["random_seed"])
    ensure_dirs(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = config["dataset"]["image_size"]
    batch_size = config["centralized"]["batch_size"]
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["class_names"]
    epochs = config["centralized"]["epochs"]
    lr = config["centralized"]["learning_rate"]
    weight_decay = config["centralized"]["weight_decay"]
    patience = config["centralized"]["patience"]

    train_transform, eval_transform = build_transforms(img_size, augment=config["preprocessing"]["train_augmentation"])

    if use_synthetic:
        print("Generating synthetic dataset for quick test...")
        csv_path = create_synthetic_dataset(
            output_dir=config["dataset"]["processed_dir"],
            num_samples=config["dataset"].get("synthetic_samples", 1000),
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

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(
        config["model"]["backbone"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
        freeze_backbone=config["model"].get("freeze_backbone", True),
    )
    model = model.to(device)

    class_weights = compute_class_weights(train_set, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2)

    out_dir = Path(config["outputs"]["models_dir"])
    log_dir = Path(config["outputs"]["logs_dir"]) / "centralized"
    writer = SummaryWriter(log_dir=str(log_dir))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:03d}/{epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "centralized_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    writer.close()

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(out_dir / "centralized_best.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Collect predictions for metrics and confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = torch.tensor(all_preds).numpy()
    all_labels = torch.tensor(all_labels).numpy()

    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    metrics["test_loss"] = test_loss
    metrics["test_accuracy"] = test_acc
    print("Test metrics:", metrics)
    print_classification_report(all_labels, all_preds, class_names)

    plot_dir = Path(config["outputs"]["plots_dir"])
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        output_path=str(plot_dir / "centralized_confusion_matrix.png"),
        title="Centralized Baseline - Confusion Matrix",
    )
    plot_metric_history(
        {"Train Accuracy": history["train_acc"], "Validation Accuracy": history["val_acc"]},
        output_path=str(plot_dir / "centralized_accuracy_history.png"),
        title="Centralized Accuracy History",
    )
    plot_metric_history(
        {"Train Loss": history["train_loss"], "Validation Loss": history["val_loss"]},
        output_path=str(plot_dir / "centralized_loss_history.png"),
        title="Centralized Loss History",
    )

    with open(out_dir / "centralized_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train centralized skin lesion classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for quick testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_centralized(cfg, use_synthetic=args.synthetic)
