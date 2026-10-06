"""
Privacy-utility trade-off experiment.
Runs federated training with different DP epsilon budgets and records performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from scripts.train_federated import train_federated


def run_privacy_utility_experiment(
    epsilon_values: list,
    config_path: str = "config.yaml",
    use_synthetic: bool = True,
) -> None:
    config = load_config(config_path)
    results = []

    for eps in epsilon_values:
        print(f"\n{'='*60}")
        print(f"Running federated training with target_epsilon={eps}")
        print(f"{'='*60}\n")

        cfg = config.copy()
        cfg["differential_privacy"]["enabled"] = True
        cfg["differential_privacy"]["target_epsilon"] = eps
        cfg["federated"]["num_rounds"] = min(cfg["federated"]["num_rounds"], 5)
        cfg["federated"]["local_epochs"] = min(cfg["federated"]["local_epochs"], 2)

        metrics = train_federated(cfg, use_synthetic=use_synthetic)
        results.append({"epsilon": eps, **metrics})

    df = pd.DataFrame(results)
    print("\n=== Privacy-Utility Trade-off ===\n")
    print(df.to_string(index=False))

    plots_dir = Path(config["outputs"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Target Epsilon (ε)")
    ax1.set_ylabel("Test Accuracy", color=color)
    ax1.plot(df["epsilon"], df["test_accuracy"], marker="o", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Test F1", color=color)
    ax2.plot(df["epsilon"], df["f1"], marker="s", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    fig.savefig(plots_dir / "privacy_utility_tradeoff.png", dpi=300, bbox_inches="tight")
    print(f"\nTrade-off plot saved to {plots_dir / 'privacy_utility_tradeoff.png'}")

    with open(plots_dir / "privacy_utility_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-utility trade-off experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
        help="List of epsilon values to test",
    )
    args = parser.parse_args()

    run_privacy_utility_experiment(args.epsilons, args.config, args.synthetic)
