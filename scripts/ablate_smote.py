"""Separate the effects of SMOTE and class weighting on the old feature cache.

Uses only original training images and the fixed validation split. The historical
two runs supply the other cells of this 2x2 experiment. Test labels are not read.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from src.data.full_isic import CLASS_NAMES
    from src.utils.feature_training import train_mlp, train_xgboost, resample_training, score_predictions
    from src.models.feature_models import build_mlp, classifier_probabilities
    from xgboost import XGBClassifier
    torch.set_num_threads(4)
    old = ROOT / "outputs/feature_models/efficientnet_b0"
    frame = pd.read_csv(old / "split_manifest.csv")
    features = np.load(ROOT / "data/features/efficientnet_b0_2af1eff877bf4a7d.npy", mmap_mode="r")
    x = joblib.load(old / "scaler.joblib").transform(features).astype(np.float32)
    y = frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    train, val = np.flatnonzero(frame.split == "train"), np.flatnonzero(frame.split == "validation")
    output = ROOT / "outputs/diagnostics/smote_ablation"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, weighted in [("none", False), ("smote", True)]:
        tx, ty, _ = resample_training(x[train], y[train], method, 500, 42)
        for kind, trainer in [("mlp", train_mlp), ("xgboost", train_xgboost)]:
            folder = output / f"{method}_weighted_{weighted}" / kind
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "validation_metrics.json").exists():
                metrics = json.loads((folder / "validation_metrics.json").read_text())
                rows.append({"model": kind, "smote": method == "smote", "class_weighted": weighted,
                             "validation_macro_f1": metrics["macro_f1"], "validation_macro_f05": metrics["macro_f05"]})
                continue
            details = trainer(tx, ty, x[val], y[val], folder, class_weighted=weighted)
            if kind == "mlp":
                model = build_mlp(1280)
                model.load_state_dict(torch.load(folder / "model.pt", weights_only=True))
                model.eval()
            else:
                model = XGBClassifier()
                model.load_model(folder / "model.ubj")
                model.set_params(device="cpu")
            metrics = score_predictions(y[val], classifier_probabilities(model, x[val], kind))
            (folder / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
            (folder / "details.json").write_text(json.dumps(details, indent=2))
            rows.append({"model": kind, "smote": method == "smote", "class_weighted": weighted,
                         "validation_macro_f1": metrics["macro_f1"], "validation_macro_f05": metrics["macro_f05"]})
    for method, folder, weighted in [(False, old, True), (True, old.with_name("efficientnet_b0_smote"), False)]:
        for kind in ["mlp", "xgboost"]:
            metrics = json.loads((folder / kind / "metrics.json").read_text())["validation"]
            if "macro_f05" not in metrics:
                predictions = pd.read_csv(folder / kind / "validation_predictions.csv")
                metrics = score_predictions(predictions.true_label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy(),
                                            predictions[CLASS_NAMES].to_numpy())
            rows.append({"model": kind, "smote": method, "class_weighted": weighted,
                         "validation_macro_f1": metrics["macro_f1"], "validation_macro_f05": metrics["macro_f05"]})
    result = pd.DataFrame(rows).sort_values(["model", "smote", "class_weighted"])
    result.to_csv(output / "comparison.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
