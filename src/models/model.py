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


class PretrainedBackbone(nn.Module):
    """
    Wrapper for pretrained torchvision backbones with optional freezing and a
    lightweight classification head.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 7,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = self._build_backbone(backbone_name)
        self.in_features = self._get_in_features(backbone_name)

        # Replace BatchNorm with GroupNorm for DP compatibility
        self.backbone = self._replace_bn_with_gn(self.backbone)

        if freeze_backbone:
            self._freeze_backbone()

        self.classifier = nn.Sequential(
            nn.Flatten() if self.in_features > 1 else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(self.in_features, num_classes),
        )

    def _build_backbone(self, name: str) -> nn.Module:
        name = name.lower()
        if name in ("resnet18", "resnet"):
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Identity()
            return model
        if name in ("efficientnet_b0", "efficientnet"):
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier = nn.Identity()
            return model
        if name in ("mobilenet_v2", "mobilenet"):
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model.classifier = nn.Identity()
            return model
        raise ValueError(f"Unknown pretrained backbone: {name}")

    def _get_in_features(self, name: str) -> int:
        name = name.lower()
        if name in ("resnet18", "resnet"):
            return 512
        if name in ("efficientnet_b0", "efficientnet"):
            return 1280
        if name in ("mobilenet_v2", "mobilenet"):
            return 1280
        raise ValueError(f"Unknown pretrained backbone: {name}")

    def _replace_bn_with_gn(self, module: nn.Module, max_groups: int = 32) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                groups = self._valid_group_count(child.num_features, max_groups)
                setattr(module, name, nn.GroupNorm(num_groups=groups, num_channels=child.num_features))
            else:
                self._replace_bn_with_gn(child, max_groups)
        return module

    def _valid_group_count(self, channels: int, max_groups: int = 32) -> int:
        """Find the largest number of groups that divides channels evenly."""
        for groups in range(min(max_groups, channels), 0, -1):
            if channels % groups == 0:
                return groups
        return 1

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class ResNet18Backbone(nn.Module):
    """ResNet18 adapted with GroupNorm for differential-privacy compatibility."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = self._replace_bn_with_gn(self.model)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def _replace_bn_with_gn(self, module: nn.Module, max_groups: int = 32) -> nn.Module:
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                groups = self._valid_group_count(child.num_features, max_groups)
                setattr(module, name, nn.GroupNorm(num_groups=groups, num_channels=child.num_features))
            else:
                self._replace_bn_with_gn(child, max_groups)
        return module

    def _valid_group_count(self, channels: int, max_groups: int = 32) -> int:
        """Find the largest number of groups that divides channels evenly."""
        for groups in range(min(max_groups, channels), 0, -1):
            if channels % groups == 0:
                return groups
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float = 0.3,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Factory function to build the requested model."""
    name = model_name.lower()

    if name in ("custom_cnn", "skincnn", "skinlesioncnn"):
        return SkinLesionCNN(num_classes=num_classes, dropout=dropout)

    if name in ("resnet18", "resnet"):
        return ResNet18Backbone(num_classes=num_classes, dropout=dropout, freeze_backbone=freeze_backbone)

    if name in ("efficientnet_b0", "efficientnet", "mobilenet_v2", "mobilenet"):
        return PretrainedBackbone(
            backbone_name=name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(f"Unknown model name: {model_name}")
