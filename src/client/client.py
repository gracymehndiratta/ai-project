"""Flower client implementation with optional Differential Privacy and FedProx."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import Config, NDArrays, Scalar
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_transforms
from src.models.model import build_model
from src.utils.training import compute_class_weights, evaluate, get_model_params, set_model_params, train_one_epoch


class SkinLesionClient(fl.client.NumPyClient):
    """A single federated-learning client."""

    def __init__(
        self,
        client_id: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: dict,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        self.num_classes = config["dataset"]["num_classes"]
        self.backbone = config["model"]["backbone"]
        self.dropout = config["model"]["dropout"]
        self.batch_size = config["federated"]["batch_size"]
        self.local_epochs = config["federated"]["local_epochs"]
        self.lr = config["federated"]["learning_rate"]
        self.image_size = config["dataset"]["image_size"]

        self.dp_enabled = config["differential_privacy"]["enabled"]
        self.target_epsilon = config["differential_privacy"]["target_epsilon"]
        self.target_delta = config["differential_privacy"]["target_delta"]
        self.max_grad_norm = config["differential_privacy"]["max_grad_norm"]
        self.noise_multiplier = config["differential_privacy"]["noise_multiplier"]

        self.fedprox_mu = config["federated"].get("fedprox_mu", 0.0)
        self.global_params: Optional[List[torch.Tensor]] = None

        _, self.eval_transform = build_transforms(self.image_size, augment=False)
        self.val_dataset.dataset.transform = self.eval_transform

        self.model = build_model(self.backbone, self.num_classes, self.dropout)
        self.model = self.model.to(self.device)

        if self.dp_enabled:
            self.model = ModuleValidator.fix(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Receive global model parameters from the server."""
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)
        self.global_params = get_model_params(self.model)

    def get_parameters(self, config: Config) -> NDArrays:
        """Return local model parameters to the server."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the local model."""
        self.set_parameters(parameters)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        class_weights = compute_class_weights(self.train_dataset).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        privacy_engine = None
        if self.dp_enabled:
            privacy_engine = PrivacyEngine()
            sample_size = len(self.train_dataset)
            noise_multiplier = self.noise_multiplier
            if noise_multiplier == "auto" or noise_multiplier is None:
                noise_multiplier = self._estimate_noise_multiplier(sample_size)

            self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=self.local_epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
            )

        for epoch in range(self.local_epochs):
            train_loss, train_acc = train_one_epoch(
                self.model,
                train_loader,
                criterion,
                optimizer,
                self.device,
                prox_mu=self.fedprox_mu,
                global_params=self.global_params,
            )

        epsilon = float("inf")
        if privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=self.target_delta)

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "epsilon": float(epsilon),
        }

        return self.get_parameters({}), len(self.train_dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on local validation data."""
        self.set_parameters(parameters)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = evaluate(self.model, val_loader, criterion, self.device)
        return float(val_loss), len(self.val_dataset), {"val_accuracy": float(val_acc)}

    def _estimate_noise_multiplier(self, sample_size: int) -> float:
        """Crude fallback noise multiplier if Opacus epsilon search is not used."""
        # A simple heuristic: larger datasets can tolerate smaller noise for same epsilon.
        return min(1.5, max(0.5, 1000.0 / sample_size))


def client_fn_factory(
    train_datasets: List[Dataset],
    val_datasets: List[Dataset],
    config: dict,
    device: torch.device,
) -> Callable[[str], fl.client.Client]:
    """Build a Flower client factory for simulation."""
    def client_fn(cid: str) -> fl.client.Client:
        return SkinLesionClient(
            client_id=int(cid),
            train_dataset=train_datasets[int(cid)],
            val_dataset=val_datasets[int(cid)],
            config=config,
            device=device,
        ).to_client()

    return client_fn
