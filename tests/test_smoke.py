"""Smoke test: run a tiny centralized and federated training loop."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from scripts.train_centralized import train_centralized
from scripts.train_federated import train_federated


def smoke_config(tmp_path):
    config = load_config("config.yaml")
    torch.set_num_threads(4)
    config["dataset"].update(synthetic_samples=140, processed_dir=str(tmp_path / "data"))
    for key in config["outputs"]:
        config["outputs"][key] = str(tmp_path / key)
    config["centralized"].update(epochs=1, batch_size=8)
    config["federated"].update(num_rounds=2, local_epochs=1, num_clients=2, batch_size=8)
    return config


def test_centralized_smoke(tmp_path):
    config = smoke_config(tmp_path)
    metrics = train_centralized(config, use_synthetic=True)
    assert "accuracy" in metrics
    print("Centralized smoke test passed.")


def test_federated_smoke(tmp_path):
    config = smoke_config(tmp_path)
    config["differential_privacy"]["enabled"] = False
    metrics = train_federated(config, use_synthetic=True)
    assert "accuracy" in metrics
    import ray
    ray.shutdown()
    print("Federated smoke test passed.")


def test_federated_privacy_smoke(tmp_path):
    config = smoke_config(tmp_path)
    metrics = train_federated(config, use_synthetic=True)
    assert 0 < metrics["privacy_epsilon"] <= config["differential_privacy"]["target_epsilon"]
    import json
    report = json.loads((Path(config["outputs"]["models_dir"]) / "federated_metrics.json").read_text())
    rounds = report["round_history"]
    assert len(rounds) == 2 and rounds[1]["max_epsilon"] > rounds[0]["max_epsilon"]
    assert all("train_accuracy" not in row and "train_loss" not in row for row in rounds)
    import ray
    ray.shutdown()


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_centralized_smoke(root / "centralized")
        test_federated_smoke(root / "federated")
        test_federated_privacy_smoke(root / "private")
