"""Verify delivered checkpoints, preprocessing boundaries and real-image inference."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import ImageFeaturePredictor, load_classifier, classifier_probabilities
    from streamlit.testing.v1 import AppTest
    os.chdir(ROOT)
    torch.set_num_threads(4)
    output = ROOT / "outputs/feature_models/efficientnet_b0_improved"
    experiment = json.loads((output / "experiment.json").read_text())
    frame = pd.read_csv(output / "split_manifest.csv")
    cache = ROOT / f"data/features/efficientnet_b0_{experiment['fingerprint'][:16]}.npy"
    features = np.load(cache, mmap_mode="r")
    scaler = joblib.load(output / "scaler.joblib")
    training = frame.split.eq("train").to_numpy()
    np.testing.assert_allclose(scaler.mean_, features[training].mean(0, dtype=np.float64), atol=1e-10)
    grouped = frame[frame.source_partition.eq("training")].groupby("lesion_id").split.nunique()
    assert grouped.eq(1).all()
    marker = output / "selection_complete.json"
    assert json.loads(marker.read_text())["selected_before_test_evaluation"]
    samples = frame[frame.split.eq("test")].groupby("label", sort=True).head(1)
    checks = []
    for kind in ["linear", "mlp", "xgboost"]:
        folder = output / kind
        assert marker.stat().st_mtime_ns >= (folder / "model.joblib").stat().st_mtime_ns
        assert (folder / "test_predictions.csv").stat().st_mtime_ns >= marker.stat().st_mtime_ns
        head, _, _ = load_classifier(folder)
        cached = classifier_probabilities(head, scaler.transform(features[samples.index]).astype(np.float32), kind)
        saved = pd.read_csv(folder / "test_predictions.csv").set_index("image_id").loc[samples.image_id, CLASS_NAMES].to_numpy()
        np.testing.assert_allclose(cached, saved, atol=1e-6)
        predictor = ImageFeaturePredictor(folder, "cuda" if torch.cuda.is_available() else "cpu")
        assert all(not parameter.requires_grad for parameter in predictor.extractor.parameters())
        for offset, (_, row) in enumerate(samples.iterrows()):
            with Image.open(row.image_path) as image:
                label, _, probabilities = predictor.predict(image)
            values = np.array([probabilities[name] for name in CLASS_NAMES])
            # CUDA convolution rounding depends on batch shape. Trees can cross
            # a split at a nearly equal feature value, amplifying this variation.
            # Check labels strictly and record score variation, rather than
            # claiming bitwise equality across single-image and bulk inference.
            tolerance = 0.01 if kind == "xgboost" else 0.003
            np.testing.assert_allclose(values, cached[offset], atol=tolerance, rtol=0)
            assert label == CLASS_NAMES[cached[offset].argmax()]
            checks.append({"model": kind, "image_id": row.image_id, "prediction": label,
                           "maximum_probability_difference": float(np.abs(values - cached[offset]).max())})
        del predictor
        torch.cuda.empty_cache()
    app = AppTest.from_file(str(ROOT / "scripts/app.py"), default_timeout=60).run()
    assert not app.exception
    options = list(app.sidebar.selectbox[0].options)
    assert len(options) == len(set(options))
    assert len([value for value in options if "Tuned multi-view" in value]) == 3
    report = {"split_images": frame.split.value_counts().to_dict(), "scaler_train_only": True,
              "internal_lesion_disjoint": True, "all_heads_selected_before_test": True,
              "numerical_note": "Batch-shape-dependent CUDA rounding: single-image scores can differ slightly from cached batch scores; all 21 checked predicted labels agree. XGBoost is discontinuous at tree splits.",
              "real_image_checks": checks, "app_choices": options, "app_default": app.sidebar.selectbox[0].value}
    (ROOT / "artifacts/improved-verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
