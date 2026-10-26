"""Exercise leakage boundaries, original backbone semantics and saved artifacts."""
from pathlib import Path
import hashlib
import http.server
import json
import os
import threading
import zipfile
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data import full_isic
from src.data.feature_cache import assign_splits, cache_features
from src.data.full_isic import CLASS_NAMES
from src.models.feature_models import build_extractor, load_classifier, classifier_probabilities, ImageFeaturePredictor
from src.utils.feature_training import train_all, score_predictions, resample_training


def make_manifest():
    rows = []
    for label in CLASS_NAMES:
        for group in range(20):
            for image in range(2):
                rows.append({"image_id": f"{label}_{group}_{image}", "label": label,
                             "lesion_id": f"{label}_{group}", "source_partition": "training"})
    return pd.DataFrame(rows)


def test_group_split_and_original_frozen_backbone():
    frame = make_manifest()
    result = assign_splits(frame)
    assert len(result) == len(frame)
    assert result.groupby("lesion_id").split.nunique().eq(1).all()
    assert result.groupby("split").label.nunique().eq(7).all()
    pd.testing.assert_frame_equal(result, assign_splits(frame))
    torch.set_num_threads(4)
    model, transform, dimension, _ = build_extractor()
    assert dimension == 1280 and not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    bn = next(module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))
    original_mean = bn.running_mean.clone()
    with torch.inference_mode():
        features = model(transform(Image.new("RGB", (450, 600))).unsqueeze(0))
    assert features.shape == (1, 1280)
    torch.testing.assert_close(bn.running_mean, original_mean)


def test_cache_and_three_models_reload(tmp_path, monkeypatch):
    torch.set_num_threads(4)
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 90, 60, 80)).save(path)
        paths.append(str(path))
    images = pd.DataFrame({"image_path": paths})
    features, first = cache_features(images, tmp_path / "cache", workers=0, batch_size=2)
    repeated, second = cache_features(images, tmp_path / "cache", workers=0, batch_size=2)
    np.testing.assert_array_equal(features, repeated)
    assert first == second and features.shape == (2, 1280)
    frame = assign_splits(make_manifest())
    rng = np.random.default_rng(42)
    y = frame.label.map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()
    centers = rng.normal(0, 4, (7, 1280))
    values = (centers[y] + rng.normal(0, 0.3, (len(y), 1280))).astype(np.float32)
    values[frame.split == "test"] += 10  # Detect preprocessing fit on held-out data.
    info = {"fingerprint": "synthetic-test-only", "backbone": "efficientnet_b0",
            "weights": first["weights"], "rows": len(frame)}
    output = tmp_path / "outputs/feature_models/efficientnet_b0"
    table = train_all(values, frame, info, output, max_epochs=5, xgb_estimators=10)
    assert set(table.model) == {"linear", "mlp", "xgboost"}
    scaler = joblib.load(output / "scaler.joblib")
    np.testing.assert_allclose(scaler.mean_, values[frame.split == "train"].mean(0), atol=1e-5)
    for kind in table.model:
        model, loaded_scaler, metadata = load_classifier(output / kind)
        predicted = classifier_probabilities(model, loaded_scaler.transform(values[:3]).astype(np.float32), kind)
        assert predicted.shape == (3, 7)
        np.testing.assert_allclose(predicted.sum(axis=1), 1, atol=1e-6)
        metrics = json.loads((output / kind / "metrics.json").read_text())
        assert metrics["test"]["images"] == (frame.split == "test").sum()
        predictor = ImageFeaturePredictor(output / kind)
        predicted_label, confidence, probabilities = predictor.predict(Image.new("RGB", (450, 600)))
        assert predicted_label in CLASS_NAMES and 0 <= confidence <= 1
        assert abs(sum(probabilities.values()) - 1) < 1e-5
    assert json.loads((output / "best_model.json").read_text())["selected_using"] == "validation only"
    original_hashes = {file: hashlib.sha256(file.read_bytes()).hexdigest()
                       for file in output.glob("*/model.*")}
    smote_output = output.with_name("efficientnet_b0_smote")
    smote_table = train_all(values, frame, info, smote_output, max_epochs=3, xgb_estimators=10,
                            resampling="smote", smote_target=36)
    sampling = json.loads((smote_output / "resampling.json").read_text())
    assert sampling["synthetic_rows"] > 0 and not sampling["class_weighted"]
    np.testing.assert_allclose(joblib.load(smote_output / "scaler.joblib").mean_,
                               values[frame.split == "train"].mean(0), atol=1e-5)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in original_hashes.items())
    for kind in smote_table.model:
        metadata = json.loads((smote_output / kind / "metadata.json").read_text())
        assert metadata["training_variant"] == "smote" and not metadata["class_weighted"]
        assert "test_macro_f05" in smote_table.columns
        metrics = json.loads((smote_output / kind / "metrics.json").read_text())
        assert 0 <= metrics["test"]["macro_f05"] <= 1
        assert all("f0.5-score" in metrics["test"]["per_class"][name] for name in CLASS_NAMES)
    assert joblib.load(smote_output / "linear/model.joblib").class_weight is None
    # Exercise the actual app against all three newly saved artifact formats.
    from streamlit.testing.v1 import AppTest
    from src.utils.config import load_config
    root = Path(__file__).resolve().parent.parent
    config = load_config(str(root / "config.yaml"))
    config["outputs"]["models_dir"] = str(tmp_path / "legacy_models")
    monkeypatch.chdir(tmp_path)
    with patch("src.utils.config.load_config", return_value=config):
        app = AppTest.from_file(str(root / "scripts/app.py")).run(timeout=60)
        assert not app.exception, app.exception
        choices = app.sidebar.selectbox[0].options
        assert len(choices) == 6 and sum("SMOTE" in name for name in choices) == 3
        for choice in choices:
            app.sidebar.selectbox[0].select(choice).run(timeout=60)
            assert not app.exception, app.exception
            assert len(app.get("file_uploader")) == 1


def test_smote_reproducibility_and_f05_definition():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(7), np.arange(8, 15))
    features = rng.normal(size=(len(labels), 8)).astype(np.float32)
    original = features.copy()
    sampled, sampled_y, info = resample_training(features, labels, method="smote", target=12)
    np.testing.assert_array_equal(features, original)
    np.testing.assert_array_equal(sampled[:len(labels)], features)
    np.testing.assert_array_equal(np.bincount(sampled_y), [12, 12, 12, 12, 12, 13, 14])
    repeated, repeated_y, _ = resample_training(features, labels, method="smote", target=12)
    np.testing.assert_array_equal(sampled, repeated)
    np.testing.assert_array_equal(sampled_y, repeated_y)
    assert info["synthetic_rows"] == 10 and info["class_weighted"] is False
    # Class 0: precision=2/3, recall=1; class 1: precision=1, recall=1/2.
    probabilities = np.eye(7)[[0, 0, 0, 1]].astype(float)
    scores = score_predictions(np.array([0, 0, 1, 1]), probabilities)
    np.testing.assert_allclose(scores["per_class"]["akiec"]["f0.5-score"], 5/7)
    np.testing.assert_allclose(scores["per_class"]["bcc"]["f0.5-score"], 5/6)
    np.testing.assert_allclose(scores["macro_f05"], (5/7 + 5/6)/7)
    np.testing.assert_allclose(scores["macro_f1"], (0.8 + 2/3)/7)


def test_download_validates_ranges_crc_and_reuses_verified_file(tmp_path, monkeypatch):
    payload = tmp_path / "source.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("image.jpg", os.urandom(1_100_000))
    data = payload.read_bytes()
    etag = hashlib.md5(data).hexdigest()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", f'"{etag}"')
            self.end_headers()

        def do_GET(self):
            assert self.headers["If-Match"] == f'"{etag}"'
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self.wfile.write(data[start:end + 1])

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(full_isic, "BASE_URL", f"http://127.0.0.1:{server.server_port}/")
    try:
        destination = tmp_path / "download"
        receipt = full_isic.download_file("data.zip", destination, workers=2)
        assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
        assert (destination / "data.zip").read_bytes() == data
        assert not (destination / "data.zip.parts").exists()
        assert full_isic.download_file("data.zip", destination, workers=2) == receipt
        # Same-size corruption must be repaired, not trusted based on byte count.
        (destination / "data.zip").write_bytes(b"x" * len(data))
        assert full_isic.download_file("data.zip", destination, workers=2) == receipt
        assert (destination / "data.zip").read_bytes() == data
    finally:
        server.shutdown()
        server.server_close()
