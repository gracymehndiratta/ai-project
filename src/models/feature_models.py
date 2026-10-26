"""Original pretrained extractors and classifiers for cached-feature experiments.

This intentionally does not use the legacy DP image-model factory: pretrained
normalization and preprocessing must remain unchanged for feature extraction.
"""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF, InterpolationMode


class MultiViewTransform:
    """Canonical center crop plus all eight square symmetries of the full image."""
    def __init__(self, canonical):
        self.canonical = canonical

    def __call__(self, image):
        full = TF.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True)
        full = TF.normalize(TF.to_tensor(full), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        views = [self.canonical(image)]
        for rotation in range(4):
            view = torch.rot90(full, rotation, [-2, -1])
            views.extend([view, view.flip(-1)])
        return torch.stack(views)


def extract_multiview(model, batch):
    """Concatenate canonical features and orientation-averaged full-image features."""
    size, views = batch.shape[:2]
    values = torch.cat([model(chunk) for chunk in batch.flatten(0, 1).split(64)])
    values = values.reshape(size, views, -1)
    return torch.cat([values[:, 0], values[:, 1:].mean(1)], dim=1)


def build_extractor(backbone: str = "efficientnet_b0"):
    if backbone == "panderm_large":
        from src.models.panderm import build_panderm, sha256_file
        model, transform = build_panderm(variant='large')
        return model, transform, 1024, "PanDerm_Large:" + sha256_file("artifacts/panderm_paper/panderm_large.pth")
    if backbone == "panderm_base":
        from src.models.panderm import build_panderm, sha256_file
        model, transform = build_panderm()
        return model, transform, 768, "PanDerm_Base:" + sha256_file("artifacts/panderm_base.pth")
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        dimension = 1280
    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        dimension = 512
    elif backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Identity()
        dimension = 1280
    else:
        raise ValueError(f"Unsupported feature extractor: {backbone}")
    model.requires_grad_(False)
    model.eval()
    return model, weights.transforms(), dimension, str(weights)


def build_mlp(dimension: int, num_classes: int = 7) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dimension, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(64, num_classes),
    )


def load_classifier(folder: str | Path, device: torch.device | str = "cpu"):
    """Load only locally produced experiment artifacts."""
    folder = Path(folder)
    info = json.loads((folder / "metadata.json").read_text())
    if info.get("artifact_type") == "tuned_ensemble":
        classifier = joblib.load(folder / "model.joblib")
    elif info["kind"] == "linear":
        classifier = joblib.load(folder / "model.joblib")
    elif info["kind"] == "mlp":
        classifier = build_mlp(info["feature_dimension"], len(info["class_names"]))
        classifier.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        classifier.to(device).eval()
    elif info["kind"] == "xgboost":
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        classifier.load_model(folder / "model.ubj")
        # Prediction is small and should not reserve a GPU after training.
        classifier.set_params(device="cpu")
    else:
        raise ValueError(f"Unknown classifier kind: {info['kind']}")
    scaler = joblib.load(folder.parent / "scaler.joblib")
    return classifier, scaler, info


def classifier_probabilities(classifier, features: np.ndarray, kind: str, device="cpu") -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(features)
    if kind == "mlp":
        with torch.inference_mode():
            return classifier(torch.as_tensor(features, dtype=torch.float32, device=device)).softmax(1).cpu().numpy()
    return classifier.predict_proba(features)


class ImageFeaturePredictor:
    """Use the exact extraction transform and train-only scaler at inference."""
    def __init__(self, folder: str | Path, device="cpu"):
        self.device = torch.device(device)
        self.classifier, self.scaler, self.info = load_classifier(folder, self.device)
        self.extractor, self.transform, dimension, weight_name = build_extractor(self.info["backbone"])
        self.multiview = self.info.get("feature_views") == "full_d4_center"
        if self.multiview:
            self.transform = MultiViewTransform(self.transform)
            dimension *= 2
        if self.info["weights"] != weight_name or self.info["feature_dimension"] != dimension:
            raise ValueError("Checkpoint and extractor weights/dimensions do not match")
        self.extractor.to(self.device).eval()

    def predict(self, image):
        with torch.inference_mode():
            batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            features = (extract_multiview(self.extractor, batch) if self.multiview else self.extractor(batch)).cpu().numpy()
        features = self.scaler.transform(features).astype(np.float32)
        probabilities = classifier_probabilities(self.classifier, features, self.info["kind"], self.device)[0]
        index = int(probabilities.argmax())
        names = self.info["class_names"]
        return names[index], float(probabilities[index]), dict(zip(names, probabilities.astype(float).tolist()))
