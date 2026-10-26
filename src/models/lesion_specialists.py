"""Seven independently selected binary heads with training-only calibration."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
import torch
from torch import nn


class BinaryMLP(nn.Sequential):
    def __init__(self, dimension=1280):
        super().__init__(nn.Linear(dimension, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4),
                         nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 1))


def binary_logits(model, values):
    values = np.asarray(values, dtype=np.float32)
    if isinstance(model, nn.Module):
        model.eval()
        with torch.inference_mode():
            return model(torch.from_numpy(values)).squeeze(1).numpy().astype(np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), dtype=np.float64)
    return logit(np.clip(model.predict_proba(values)[:, 1], 1e-7, 1 - 1e-7))


class BinarySpecialist:
    def __init__(self, name, model, scaler, calibrator, threshold, config):
        self.name, self.model, self.scaler, self.calibrator = name, model, scaler, calibrator
        self.threshold, self.config = float(threshold), config

    def predict_positive(self, raw_features):
        values = self.scaler.transform(raw_features[:, 1280:2560]).astype(np.float32)
        scores = binary_logits(self.model, values)
        return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


class SpecialistSystem:
    """Normalize calibrated binary scores, optionally adjusted by OOF thresholds.

    The independent binary probabilities and the normalized decision scores are
    different quantities. Threshold adjustment changes class selection and does
    not establish calibrated seven-class posterior probabilities.
    """
    def __init__(self, specialists, strength=0.0):
        self.specialists = specialists
        self.strength = float(strength)

    def binary_probabilities(self, features):
        return np.column_stack([head.predict_positive(features) for head in self.specialists])

    def predict_proba(self, features):
        probabilities = self.binary_probabilities(features)
        thresholds = np.array([head.threshold for head in self.specialists])
        scores = np.clip(probabilities, 1e-12, 1) / np.maximum(thresholds, 1e-6) ** self.strength
        return scores / scores.sum(1, keepdims=True)
