"""Compare centralized and federated model metrics side-by-side."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_metrics(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compare(config_path: str = "config.yaml") -> None:
    from src.utils.config import load_config

    config = load_config(config_path)
    models_dir = Path(config["outputs"]["models_dir"])
    plots_dir = Path(config["outputs"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    centralized_path = models_dir / "centralized_metrics.json"
    federated_path = models_dir / "federated_metrics.json"

    rows = []
    if centralized_path.exists():
        cm = load_metrics(centralized_path)
        rows.append(
            {
                "Approach": "Centralized",
                "Accuracy": cm.get("test_accuracy", cm.get("accuracy")),
                "Precision": cm.get("precision"),
                "Recall": cm.get("recall"),
                "F1": cm.get("f1"),
                "Loss": cm.get("test_loss", cm.get("loss")),
            }
        )
    if federated_path.exists():
        fm = load_metrics(federated_path)
        rows.append(
            {
                "Approach": "Federated",
                "Accuracy": fm["final"].get("test_accuracy", fm["final"].get("accuracy")),
                "Precision": fm["final"].get("precision"),
                "Recall": fm["final"].get("recall"),
                "F1": fm["final"].get("f1"),
                "Loss": fm["final"].get("test_loss", fm["final"].get("loss")),
            }
        )

    if not rows:
        print("No metrics found. Run training first.")
        return

    df = pd.DataFrame(rows)
    print("\n=== Centralized vs Federated Comparison ===\n")
    print(df.to_string(index=False))
    print()

    # Bar plot comparison
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
    df_plot = df.set_index("Approach")[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Centralized vs Federated Performance")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Approach")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(plots_dir / "comparison.png", dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {plots_dir / 'comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare centralized and federated results")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    compare(args.config)
