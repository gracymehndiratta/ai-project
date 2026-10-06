"""Smoke test: run a tiny centralized and federated training loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from scripts.train_centralized import train_centralized
from scripts.train_federated import train_federated


def test_centralized_smoke():
    config = load_config("config.yaml")
    config["centralized"]["epochs"] = 2
    metrics = train_centralized(config, use_synthetic=True)
    assert "accuracy" in metrics
    print("Centralized smoke test passed.")


def test_federated_smoke():
    config = load_config("config.yaml")
    config["federated"]["num_rounds"] = 1
    config["federated"]["local_epochs"] = 1
    config["federated"]["num_clients"] = 2
    config["differential_privacy"]["enabled"] = False
    metrics = train_federated(config, use_synthetic=True)
    assert "accuracy" in metrics
    print("Federated smoke test passed.")


if __name__ == "__main__":
    test_centralized_smoke()
    test_federated_smoke()
