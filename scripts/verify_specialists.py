"""Audit specialist artifacts, leakage boundaries, metrics and deployed inference."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import ImageFeaturePredictor, load_classifier
    from src.utils.specialist_training import binary_metrics
    from src.utils.feature_training import score_predictions
    os.chdir(ROOT)
    torch.set_num_threads(4)
    output = ROOT / "outputs/feature_models/efficientnet_b0_specialists"
    manifest = pd.read_csv(output / "split_manifest.csv")
    plan = json.loads((output / "experiment.json").read_text())
    feature_path = ROOT / f"data/features/efficientnet_b0_{plan['fingerprint'][:16]}.npy"
    features = np.load(feature_path, mmap_mode="r")
    train = manifest[manifest.split == "train"]
    train_ids = set(train.image_id)
    by_id = manifest.set_index("image_id")
    for fold in json.loads((output / "fold_audit.json").read_text()):
        assert all(set(rows) <= train_ids for rows in fold.values())
        groups = {key: set(by_id.loc[rows].lesion_id) for key, rows in fold.items()}
        assert groups["fit"].isdisjoint(groups["calibration"])
        assert groups["holdout"].isdisjoint(groups["fit"] | groups["calibration"])
        assert groups["outer_train"] == groups["fit"] | groups["calibration"]
    parent = output / "specialists"
    marker = output / "selection_complete.json"
    selection = json.loads(marker.read_text())
    assert selection["model_sha256"] == hashlib.sha256((parent / "model.joblib").read_bytes()).hexdigest()
    system, identity, info = load_classifier(parent)
    assert marker.stat().st_mtime_ns >= (parent / "model.joblib").stat().st_mtime_ns
    probe_rows = manifest[manifest.split == "test"].groupby("label", sort=True).head(1)
    raw_probe = np.asarray(features[probe_rows.index])
    np.testing.assert_array_equal(identity.transform(raw_probe), raw_probe)
    binary_results = pd.read_csv(output / "binary_specialist_metrics.csv")
    metrics = json.loads((parent / "metrics.json").read_text())
    for k, name in enumerate(CLASS_NAMES):
        folder = output / "lesion_models" / name
        head = joblib.load(folder / "model.joblib")
        assert head.name == system.specialists[k].name == name
        np.testing.assert_allclose(head.predict_positive(raw_probe), system.specialists[k].predict_positive(raw_probe), atol=1e-12)
        np.testing.assert_allclose(head.scaler.mean_, features[train.index, 1280:2560].mean(0, dtype=np.float64), atol=1e-10)
        training_predictions = pd.read_csv(folder / "training_oof_predictions.csv")
        assert training_predictions.image_id.tolist() == train.image_id.tolist()
        assert training_predictions.label.tolist() == train.label.eq(name).astype(int).tolist()
    for split in ["validation", "test", "official_validation", "official_test"]:
        records = pd.read_csv(parent / f"{split}_predictions.csv")
        independent = pd.read_csv(parent / f"{split}_binary_probabilities.csv")
        rows = manifest[manifest.split == split]
        assert records.image_id.tolist() == independent.image_id.tolist() == rows.image_id.tolist()
        labels = rows.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
        recomputed = score_predictions(labels, records[CLASS_NAMES].to_numpy())
        for metric in ["accuracy", "macro_f1", "macro_f05", "weighted_f1", "weighted_f05", "balanced_accuracy"]:
            np.testing.assert_allclose(recomputed[metric], metrics[split][metric], atol=1e-12)
        assert (parent / f"{split}_predictions.csv").stat().st_mtime_ns >= marker.stat().st_mtime_ns
        for k, head in enumerate(system.specialists):
            recomputed = binary_metrics((labels == k).astype(int), independent[head.name].to_numpy(), head.threshold)
            saved = binary_results[(binary_results.split == split) & (binary_results.lesion == head.name)].iloc[0]
            for metric in ["accuracy", "precision", "recall", "f1", "f05", "balanced_accuracy", "brier_score"]:
                np.testing.assert_allclose(recomputed[metric], saved[metric], atol=1e-12)
    previous = json.loads((ROOT / "artifacts/specialist-baseline-hashes.json").read_text())
    for name, digest in previous.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    checks = []
    predictor = ImageFeaturePredictor(parent, "cuda" if torch.cuda.is_available() else "cpu")
    cached = system.predict_proba(raw_probe)
    assert all(not parameter.requires_grad for parameter in predictor.extractor.parameters())
    for offset, (_, row) in enumerate(probe_rows.iterrows()):
        with Image.open(row.image_path) as image:
            label, _, probabilities = predictor.predict(image)
        values = np.array([probabilities[name] for name in CLASS_NAMES])
        # Single-image and batched CUDA convolutions have different rounding;
        # tree splits can amplify tiny feature differences. Labels must agree.
        np.testing.assert_allclose(values, cached[offset], atol=0.02, rtol=0)
        assert label == CLASS_NAMES[cached[offset].argmax()]
        checks.append({"image_id": row.image_id, "true_label": row.label, "predicted_label": label,
                       "maximum_score_difference": float(np.abs(values - cached[offset]).max())})
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "scripts/app.py"), default_timeout=60).run()
    assert not app.exception
    options = list(app.sidebar.selectbox[0].options)
    specialist_option = [option for option in options if "Seven lesion specialists" in option]
    assert len(specialist_option) == 1
    app.sidebar.selectbox[0].select(specialist_option[0]).run()
    assert not app.exception
    result = {"all_previous_artifacts_unchanged": len(previous), "lesion_groups_separated": True,
              "calibration_folds_training_only": True, "each_final_scaler_training_only": True,
              "all_heldout_binary_and_combined_metrics_recomputed": True,
              "selection_saved_before_test": True, "real_image_checks": checks,
              "app_model_choices": options, "selected_app_model": app.sidebar.selectbox[0].value}
    (ROOT / "artifacts/specialist-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
