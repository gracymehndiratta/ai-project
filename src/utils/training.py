"""Shared training and evaluation loops."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset or subset."""
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset[idx][1] for idx in dataset.indices]
    elif hasattr(dataset, "df") and hasattr(dataset, "label_col") and hasattr(dataset, "class_to_idx"):
        labels = dataset.df[dataset.label_col].map(dataset.class_to_idx).values.tolist()
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]

    counts = np.bincount(labels)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    prox_mu: float = 0.0,
    global_params: Optional[list] = None,
) -> Tuple[float, float]:
    """Train model for one epoch. Supports FedProx regularization."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # FedProx proximal term
        if prox_mu > 0.0 and global_params is not None:
            prox_term = 0.0
            for param, global_param in zip(model.parameters(), global_params):
                prox_term += torch.norm(param - global_param, p=2) ** 2
            loss += (prox_mu / 2.0) * prox_term

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def get_model_params(model: nn.Module) -> list:
    """Return a copy of model parameters as CPU tensors."""
    return [param.data.cpu().clone() for param in model.parameters()]


def set_model_params(model: nn.Module, params: list) -> None:
    """Set model parameters from a list of tensors."""
    for param, new_param in zip(model.parameters(), params):
        param.data.copy_(new_param.data)
