"""Feature-view and serialized inference checks for the tuned heads."""
import json

import joblib
import numpy as np
from PIL import Image
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.feature_models import MultiViewTransform, build_extractor, extract_multiview, load_classifier, classifier_probabilities
from src.utils.tuned_training import Columns, TunedClassifier, class_weights, macro_f1_cost


def test_d4_pooling_and_frozen_extractor():
    torch.set_num_threads(4)
    model, canonical, _, _ = build_extractor()
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (240, 320, 3), dtype=np.uint8))
    transform = MultiViewTransform(canonical)
    first, flipped = transform(image), transform(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    assert first.shape == (9, 3, 224, 224)
    with torch.inference_mode():
        features = extract_multiview(model, torch.stack([first, flipped]))
    assert features.shape == (2, 2560)
    # D4 averaging must be invariant to a horizontal flip of the full image.
    torch.testing.assert_close(features[0, 1280:], features[1, 1280:], rtol=1e-4, atol=1e-5)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_nested_ensemble_roundtrip(tmp_path):
    rng = np.random.default_rng(43)
    features = rng.normal(size=(70, 12)).astype(np.float32)
    labels = np.repeat(np.arange(7), 10)
    scaler = StandardScaler().fit(features[:49])
    scaled = scaler.transform(features).astype(np.float32)
    model = LogisticRegression().fit(scaled[:, :6], labels)
    head = TunedClassifier([model], Columns(0, 6), np.arange(1, 8))
    ensemble = TunedClassifier([head, head], Columns(0, 12))
    folder = tmp_path / "mlp"
    folder.mkdir()
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(ensemble, folder / "model.joblib")
    (folder / "metadata.json").write_text(json.dumps({"kind": "mlp", "artifact_type": "tuned_ensemble"}))
    restored, _, _ = load_classifier(folder)
    np.testing.assert_allclose(classifier_probabilities(restored, scaled, "mlp"), head.predict_proba(scaled))
    assert macro_f1_cost(labels, restored.predict_proba(scaled)) <= 0
    np.testing.assert_array_equal(class_weights(labels, 0), np.ones(7))
    skewed = np.concatenate([np.repeat(np.arange(7), 3), np.zeros(100, dtype=int)])
    for power in [0, 0.5, 1]:
        weights = class_weights(skewed, power)
        np.testing.assert_allclose(weights[skewed].mean(), 1)
