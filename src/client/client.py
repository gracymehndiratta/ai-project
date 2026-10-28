"""Flower client implementation with optional Differential Privacy and FedProx."""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from typing import Callable, Dict, List, MutableMapping, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import Config, ConfigRecord, Context, NDArrays, Scalar
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Dataset

from src.models.model import build_model
from src.utils.training import compute_class_weights, evaluate, get_model_params, train_one_epoch


class SkinLesionClient(fl.client.NumPyClient):
    """A single federated-learning client."""

    def __init__(
        self,
        client_id: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: dict,
        device: torch.device,
        privacy_state: Optional[MutableMapping] = None,
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
        self.num_rounds = config["federated"]["num_rounds"]
        self.privacy_state = privacy_state if privacy_state is not None else {}
        self.privacy_engine = None

        aggregation = config["federated"].get("aggregation", "fedavg").lower()
        if aggregation not in {"fedavg", "fedprox"}:
            raise ValueError("Federated aggregation must be fedavg or fedprox")
        self.fedprox_mu = config["federated"].get("fedprox_mu", 0.0) if aggregation == "fedprox" else 0.0
        if self.fedprox_mu < 0:
            raise ValueError("FedProx mu must be nonnegative")
        self.global_params: Optional[List[torch.Tensor]] = None

        self.model = build_model(
            self.backbone,
            self.num_classes,
            dropout=self.dropout,
            freeze_backbone=config["model"].get("freeze_backbone", True),
        )
        self.model = self.model.to(self.device)

        if self.dp_enabled:
            self.model = ModuleValidator.fix(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Receive global model parameters from the server."""
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)
        self.global_params = [p.to(self.device) for p in get_model_params(self.model)]

    def get_parameters(self, config: Config) -> NDArrays:
        """Return local model parameters to the server."""
        return [val.cpu().numpy().copy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the local model."""
        self.set_parameters(parameters)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Private label counts are not a public preprocessing input. Keep the
        # DP loss per-example rather than coupling examples through those counts.
        class_weights = None if self.dp_enabled else compute_class_weights(self.train_dataset, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.dp_enabled:
            optimizer, train_loader = self._make_private(optimizer, train_loader)

        try:
            for epoch in range(self.local_epochs):
                train_loss, train_acc = train_one_epoch(
                    self.model,
                    train_loader,
                    criterion,
                    optimizer,
                    self.device,
                    # In DP mode, add the public proximal gradient after clipping
                    # and noise, immediately before the underlying AdamW step.
                    prox_mu=0.0 if self.dp_enabled else self.fedprox_mu,
                    global_params=self.global_params,
                )
        finally:
            if self.privacy_engine is not None:
                self.privacy_state["accountant"] = json.dumps(self.privacy_engine.accountant.state_dict())
                self.model = self.model.to_standard_module()

        epsilon = float("inf")
        if self.privacy_engine is not None:
            epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)

        metrics = {}
        if self.dp_enabled:
            metrics["epsilon"] = float(epsilon)
        if self.dp_enabled:
            # Unnoised loss/accuracy are private-data queries, not DP outputs.
            metrics.update(noise_multiplier=float(optimizer.noise_multiplier),
                           privacy_steps=sum(entry[2] for entry in self.privacy_engine.accountant.history))
        else:
            metrics.update(train_loss=float(train_loss), train_accuracy=float(train_acc))

        return self.get_parameters({}), len(self.train_dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on local validation data."""
        self.set_parameters(parameters)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = evaluate(self.model, val_loader, criterion, self.device)
        return float(val_loss), len(self.val_dataset), {"val_accuracy": float(val_acc)}

    def _make_private(self, optimizer, train_loader):
        """Restore local accounting and calibrate noise for the entire run."""
        if self.local_epochs < 1 or self.num_rounds < 1:
            raise ValueError("Private training needs positive local epochs and rounds")
        if not 0 < self.target_delta < 1 or self.target_epsilon <= 0:
            raise ValueError("Private training needs epsilon > 0 and 0 < delta < 1")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("Private max_grad_norm must be positive and finite")

        steps_per_round = len(train_loader) * self.local_epochs
        planned_steps = steps_per_round * self.num_rounds
        sample_rate = 1 / len(train_loader)
        plan = json.dumps({"client_id": self.client_id, "samples": len(self.train_dataset),
                           "sample_rate": sample_rate, "steps": planned_steps,
                           "epsilon": self.target_epsilon, "delta": self.target_delta,
                           "noise_multiplier": self.noise_multiplier}, sort_keys=True)
        if "plan" in self.privacy_state and self.privacy_state["plan"] != plan:
            raise ValueError("Privacy configuration changed within a run; refusing to reset accounting")

        self.privacy_engine = PrivacyEngine(accountant="rdp")
        if "accountant" in self.privacy_state:
            state = json.loads(self.privacy_state["accountant"])
            state["history"] = [tuple(entry) for entry in state["history"]]
            self.privacy_engine.accountant.load_state_dict(state)
        completed = sum(entry[2] for entry in self.privacy_engine.accountant.history)
        if completed + steps_per_round > planned_steps:
            raise ValueError("Private training would exceed the calibrated run length")

        if self.noise_multiplier in ("auto", None):
            noise_multiplier = self.privacy_state.get("noise_multiplier")
            if noise_multiplier is None:
                noise_multiplier = get_noise_multiplier(
                    target_epsilon=self.target_epsilon, target_delta=self.target_delta,
                    sample_rate=sample_rate, steps=planned_steps, accountant="rdp",
                )
        else:
            noise_multiplier = float(self.noise_multiplier)
        if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
            raise ValueError("Private noise_multiplier must be positive and finite, or auto")

        self.model, optimizer, train_loader = self.privacy_engine.make_private(
            module=self.model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=noise_multiplier, max_grad_norm=self.max_grad_norm,
        )
        self.privacy_state["plan"] = plan
        self.privacy_state["noise_multiplier"] = float(noise_multiplier)
        if self.fedprox_mu > 0:
            accountant_hook = optimizer.step_hook

            def add_proximal_gradient(dp_optimizer):
                accountant_hook(dp_optimizer)
                with torch.no_grad():
                    for param, global_param in zip(self.model.parameters(), self.global_params):
                        if param.requires_grad and param.grad is not None:
                            param.grad.add_(param - global_param, alpha=self.fedprox_mu)

            optimizer.attach_step_hook(add_proximal_gradient)
        return optimizer, train_loader


def client_fn_factory(
    train_datasets: List[Dataset],
    val_datasets: List[Dataset],
    config: dict,
    device: torch.device,
) -> Callable[[Context], fl.client.Client]:
    """Persist each client's accountant in Flower's node-local, per-run state."""
    def client_fn(context: Context) -> fl.client.Client:
        cid = int(context.node_config["partition-id"])
        if "privacy" not in context.state:
            context.state["privacy"] = ConfigRecord()
        return SkinLesionClient(
            client_id=cid,
            train_dataset=train_datasets[cid],
            val_dataset=val_datasets[cid],
            config=config,
            device=device,
            privacy_state=context.state["privacy"],
        ).to_client()

    return client_fn
