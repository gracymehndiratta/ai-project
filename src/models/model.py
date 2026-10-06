"""CNN model architectures for multi-class skin lesion classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GroupNormBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU -> MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # GroupNorm is compatible with differential privacy (unlike BatchNorm).
        num_groups = min(groups, out_channels)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class SkinLesionCNN(nn.Module):
    """Lightweight CNN for HAM10000 (128x128 inputs)."""

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            GroupNormBlock(in_channels, 32),
            GroupNormBlock(32, 64),
            GroupNormBlock(64, 128),
            GroupNormBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18Backbone(nn.Module):
    """ResNet18 adapted with GroupNorm for differential-privacy compatibility."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3) -> None:
        super().__init__()
        # Use a GroupNorm variant of ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace BatchNorm layers with GroupNorm
        self.model = self._replace_bn_with_gn(self.model)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def _replace_bn_with_gn(self, module: nn.Module, num_groups: int = 32) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_groups = min(num_groups, child.num_features)
                setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features))
            else:
                self._replace_bn_with_gn(child, num_groups)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(model_name: str, num_classes: int, dropout: float = 0.3) -> nn.Module:
    """Factory function to build the requested model."""
    if model_name.lower() in ("custom_cnn", "skincnn", "skinlesioncnn"):
        return SkinLesionCNN(num_classes=num_classes, dropout=dropout)
    if model_name.lower() in ("resnet18", "resnet"):
        return ResNet18Backbone(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {model_name}")
