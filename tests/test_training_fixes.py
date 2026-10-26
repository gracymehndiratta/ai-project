"""Regression coverage for image training, grouped evaluation and DP lifecycle."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset
from flwr.common import Context, RecordDict

from scripts.download_data import prepare_metadata
from scripts.train_centralized import train_centralized
from scripts.train_federated import load_data_for_simulation
from src.client.client import SkinLesionClient, client_fn_factory
from src.data.dataset import SkinLesionDataset, partition_data_dirichlet, split_dataset
from src.models.model import SkinLesionCNN
from src.server.strategy import FedAvgWithLogging
from src.utils.config import load_config


class ConstantTransform:
    def __init__(self, value):
        self.value = value

    def __call__(self, image):
        return torch.full((3, 8, 8), self.value, dtype=torch.float32)


@pytest.fixture
def grouped_dataset(tmp_path):
    rows = []
    for label in range(7):
        for lesion in range(20):
            for view in range(2):
                name = f"{label}_{lesion}_{view}"
                rows.append({"image_id": name, "label": str(label), "lesion_id": f"{label}_{lesion}"})
                Image.new("RGB", (8, 8)).save(tmp_path / f"{name}.jpg")
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    return SkinLesionDataset(metadata, tmp_path, ConstantTransform(1))


def test_grouped_splits_keep_every_image_and_independent_transforms(grouped_dataset):
    dataset = grouped_dataset
    train, val, test = split_dataset(dataset, eval_transform=ConstantTransform(2))
    parts = [train, val, test]
    assert sum(len(part) for part in parts) == len(dataset)
    assert len(set().union(*(set(part.indices) for part in parts))) == len(dataset)
    groups = [set(dataset.df.iloc[part.indices].lesion_id) for part in parts]
    assert groups[0].isdisjoint(groups[1] | groups[2]) and groups[1].isdisjoint(groups[2])
    assert all(dataset.df.iloc[part.indices].label.nunique() == 7 for part in parts)
    assert [part.indices for part in split_dataset(dataset)] == [part.indices for part in parts]
    assert train[0][0].eq(1).all() and val[0][0].eq(2).all() and test[0][0].eq(2).all()
    val.dataset.transform = ConstantTransform(3)
    assert train[0][0].eq(1).all() and test[0][0].eq(2).all()


@pytest.mark.parametrize("val_ratio,test_ratio,expected_empty", [(0, .3, 1), (.3, 0, 2), (0, 0, 1)])
def test_zero_sized_holdouts_stay_in_the_correct_split(grouped_dataset, val_ratio, test_ratio, expected_empty):
    parts = split_dataset(grouped_dataset, val_ratio=val_ratio, test_ratio=test_ratio)
    assert len(parts[expected_empty]) == 0
    assert sum(map(len, parts)) == len(grouped_dataset)
    if val_ratio == test_ratio == 0:
        assert len(parts[2]) == 0


@pytest.mark.parametrize("defect", ["missing", "conflicting"])
def test_invalid_lesion_metadata_is_rejected(grouped_dataset, defect):
    if defect == "missing":
        grouped_dataset.df.loc[0, "lesion_id"] = None
    else:
        grouped_dataset.df.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="Lesion IDs|inconsistent"):
        split_dataset(grouped_dataset)


def test_preparation_retains_lesion_group_ids(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    pd.DataFrame({"image_id": ["a", "b"], "dx": ["nv", "nv"],
                  "lesion_id": ["same-lesion", "same-lesion"]}).to_csv(raw / "HAM10000_metadata.csv", index=False)
    prepare_metadata(raw, processed)
    result = pd.read_csv(processed / "HAM10000_metadata.csv")
    assert result.lesion_id.tolist() == ["same-lesion", "same-lesion"]
    assert result.label.tolist() == ["nv", "nv"]


@pytest.mark.parametrize("trainer", [train_centralized, load_data_for_simulation])
def test_real_image_training_requires_group_metadata(tmp_path, trainer):
    config = load_config()
    config["dataset"].update(raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path))
    for key in config["outputs"]:
        config["outputs"][key] = str(tmp_path / key)
    pd.DataFrame({"image_id": ["a", "b"], "label": ["nv", "nv"]}).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)
    with pytest.raises(ValueError, match="requires lesion_id"):
        trainer(config, use_synthetic=False)


def test_federated_validation_uses_eval_transform_without_changing_training(tmp_path, monkeypatch):
    config = load_config()
    config["dataset"].update(processed_dir=str(tmp_path), synthetic_samples=140, image_size=8)
    config["federated"].update(num_clients=2, iid=True)
    monkeypatch.setattr("scripts.train_federated.build_transforms", lambda *a, **k: (ConstantTransform(1), ConstantTransform(2)))
    train, val, test, _ = load_data_for_simulation(config, use_synthetic=True)
    assert all(part[0][0].eq(1).all() for part in train)
    assert all(part[0][0].eq(2).all() for part in val)
    assert test[0][0].eq(2).all()


def test_dirichlet_partition_keeps_nonconsecutive_classes_in_a_subset():
    dataset = TensorDataset(torch.zeros(12, 1), torch.tensor([0] * 4 + [4] * 4 + [6] * 4))
    subset = Subset(dataset, [1, 2, 4, 5, 8, 9, 10])
    partitions = partition_data_dirichlet(subset, num_clients=2)
    assigned = [index for part in partitions for index in part.indices]
    assert sorted(assigned) == sorted(subset.indices) and len(set(assigned)) == len(assigned)


@pytest.mark.parametrize("size", [128, 224])
def test_custom_cnn_accepts_configured_sizes_and_legacy_state(size):
    torch.set_num_threads(4)
    model = SkinLesionCNN(dropout=0).eval()
    # The new pooling operation adds no checkpoint tensors or renumbered keys.
    checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
    restored = SkinLesionCNN(dropout=0).eval()
    restored.load_state_dict(checkpoint, strict=True)
    image = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        output = restored(image)
        assert output.shape == (1, 7) and torch.isfinite(output).all()
        if size == 128:
            legacy_output = restored.classifier(restored.features(image))
            torch.testing.assert_close(output, legacy_output, rtol=0, atol=0)


@pytest.fixture
def private_setup(monkeypatch):
    torch.set_num_threads(4)
    monkeypatch.setattr("src.client.client.build_model", lambda *a, **k: nn.Sequential(nn.Flatten(), nn.Linear(192, 7)))
    config = load_config()
    config["dataset"]["image_size"] = 8
    config["federated"].update(batch_size=14, local_epochs=1, num_rounds=2, num_clients=2)
    config["differential_privacy"]["noise_multiplier"] = 5.0
    dataset = TensorDataset(torch.rand(56, 3, 8, 8), torch.arange(56) % 7)
    return config, dataset


def context(partition=0, state=None):
    return Context(run_id=1, node_id=100 + partition, node_config={"partition-id": partition},
                   state=state if state is not None else RecordDict(), run_config={})


@pytest.mark.parametrize("noise", [5.0, "auto"])
def test_privacy_composes_across_recreated_flower_clients(private_setup, noise):
    config, dataset = private_setup
    config["differential_privacy"]["noise_multiplier"] = noise
    factory = client_fn_factory([dataset, dataset], [dataset, dataset], config, torch.device("cpu"))
    first_context = context()
    first = factory(first_context).numpy_client
    initial = first.get_parameters({})
    updated, _, first_metrics = first.fit(initial, {})
    # Recreate the client and its transported node state as Ray does each round.
    resumed_context = context(state=deepcopy(first_context.state))
    second = factory(resumed_context).numpy_client
    _, _, second_metrics = second.fit(updated, {})
    assert first_metrics["privacy_steps"] == 4 and second_metrics["privacy_steps"] == 8
    assert 0 < first_metrics["epsilon"] < second_metrics["epsilon"] <= config["differential_privacy"]["target_epsilon"]
    assert first_metrics["noise_multiplier"] == second_metrics["noise_multiplier"]
    if noise != "auto":
        assert second_metrics["noise_multiplier"] == noise
    assert "train_loss" not in second_metrics and "train_accuracy" not in second_metrics
    assert list(second.model.state_dict()) == list(first.model.state_dict())
    assert not any(name.startswith("_module.") for name in second.model.state_dict())
    # Another participant and a new run must not inherit this accountant.
    other = factory(context(1)).numpy_client
    assert not other.privacy_state
    fresh = factory(context()).numpy_client
    assert not fresh.privacy_state
    history = json.loads(resumed_context.state["privacy"]["accountant"])["history"]
    assert sum(entry[2] for entry in history) == 8


def test_same_private_client_can_fit_again_without_resetting_or_duplicate_hooks(private_setup):
    config, dataset = private_setup
    client = SkinLesionClient(0, dataset, dataset, config, torch.device("cpu"))
    parameters, _, first = client.fit(client.get_parameters({}), {})
    parameters, _, second = client.fit(parameters, {})
    assert second["privacy_steps"] == 8 and second["epsilon"] > first["epsilon"]
    with pytest.raises(ValueError, match="exceed"):
        client.fit(parameters, {})


def test_private_plan_changes_cannot_silently_reset_accounting(private_setup):
    config, dataset = private_setup
    state = {}
    client = SkinLesionClient(0, dataset, dataset, config, torch.device("cpu"), state)
    parameters, _, _ = client.fit(client.get_parameters({}), {})
    previous = state["accountant"]
    changed = deepcopy(config)
    changed["differential_privacy"]["target_epsilon"] = 4.0
    replacement = SkinLesionClient(0, dataset, dataset, changed, torch.device("cpu"), state)
    with pytest.raises(ValueError, match="configuration changed"):
        replacement.fit(parameters, {})
    assert state["accountant"] == previous


@pytest.mark.parametrize("private", [False, True])
def test_fedavg_ignores_mu_but_fedprox_changes_the_actual_update(private_setup, private):
    config, dataset = private_setup
    config["differential_privacy"]["enabled"] = private
    updates = []
    initial = None
    for aggregation, mu in [("fedavg", 100.0), ("fedprox", 0.0), ("fedprox", 100.0)]:
        current = deepcopy(config)
        current["federated"].update(aggregation=aggregation, fedprox_mu=mu)
        client = SkinLesionClient(0, dataset, dataset, current, torch.device("cpu"))
        if initial is None:
            initial = client.get_parameters({})
        torch.manual_seed(42)
        parameters, _, _ = client.fit(initial, {})
        updates.append(parameters)
    for fedavg, zero_prox in zip(updates[0], updates[1]):
        np.testing.assert_array_equal(fedavg, zero_prox)
    assert any(not np.allclose(plain, prox, rtol=1e-7, atol=1e-9) for plain, prox in zip(updates[0], updates[2]))


def test_failed_federated_round_does_not_look_like_successful_training():
    strategy = FedAvgWithLogging(accept_failures=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        strategy.aggregate_fit(1, [], [RuntimeError("client failed")])
