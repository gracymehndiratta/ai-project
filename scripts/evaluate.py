"""Evaluate a saved model on a test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SkinLesionDataset
from src.data.transforms import build_transforms
from src.models.model import build_model
from src.utils.config import load_config
from src.utils.metrics import compute_metrics, plot_confusion_matrix, print_classification_report
from src.utils.training import evaluate as eval_model


def evaluate_saved_model(
    model_path: str,
    config_path: str = "config.yaml",
    csv_path: str | None = None,
    image_dir: str | None = None,
    use_synthetic: bool = False,
) -> dict:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["class_names"]
    img_size = config["dataset"]["image_size"]

    _, eval_transform = build_transforms(img_size, augment=False)

    if use_synthetic:
        from src.data.dataset import create_synthetic_dataset
        csv_path = create_synthetic_dataset(
            output_dir=config["dataset"]["processed_dir"],
            num_samples=1000,
            image_size=img_size,
            num_classes=num_classes,
            seed=config["random_seed"],
        )
        image_dir = Path(config["dataset"]["processed_dir"]) / "images"
    else:
        if csv_path is None:
            csv_path = Path(config["dataset"]["processed_dir"]) / "HAM10000_metadata.csv"
        if image_dir is None:
            image_dir = Path(config["dataset"]["raw_dir"]) / "images"

    dataset = SkinLesionDataset(
        csv_path=csv_path,
        image_dir=str(image_dir),
        transform=eval_transform,
        label_col="label",
    )
    loader = DataLoader(dataset, batch_size=config["centralized"]["batch_size"], shuffle=False, num_workers=0)

    model = build_model(config["model"]["backbone"], num_classes, dropout=config["model"]["dropout"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = eval_model(model, loader, criterion, device)
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    metrics["loss"] = test_loss
    metrics["accuracy"] = test_acc
    print_classification_report(all_labels, all_preds, class_names)

    plot_dir = Path(config["outputs"]["plots_dir"])
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        output_path=str(plot_dir / "evaluation_confusion_matrix.png"),
        title="Evaluation - Confusion Matrix",
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved skin lesion model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--csv", type=str, default=None, help="Path to metadata CSV")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for quick testing")
    args = parser.parse_args()

    metrics = evaluate_saved_model(args.model, args.config, args.csv, args.image_dir, args.synthetic)
    print(json.dumps(metrics, indent=2))
