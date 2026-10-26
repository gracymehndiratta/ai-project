"""Compare saved predictions without retraining or changing baseline artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.data.full_isic import CLASS_NAMES
from src.utils.feature_training import score_predictions
import matplotlib.pyplot as plt


def compare_runs(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    base_info = json.loads((baseline / "experiment.json").read_text())
    new_info = json.loads((candidate / "experiment.json").read_text())
    if base_info.get("training_variant", "class_weighted") != "class_weighted" or new_info.get("training_variant") != "smote":
        raise ValueError("Expected a class-weighted baseline and a SMOTE candidate")
    for key in ["fingerprint", "seed", "weights", "backbone"]:
        if base_info[key] != new_info[key]:
            raise ValueError(f"Runs differ in {key}; cannot claim a matched comparison")
    manifest = pd.read_csv(baseline / "split_manifest.csv")
    pd.testing.assert_frame_equal(manifest, pd.read_csv(candidate / "split_manifest.csv"))
    rows, per_class = [], []
    for run, folder in [("class_weighted", baseline), ("smote", candidate)]:
        for model in ["linear", "mlp", "xgboost"]:
            for split in ["validation", "test", "official_validation", "official_test"]:
                expected = manifest[manifest.split == split]
                if not len(expected):
                    continue
                predictions = pd.read_csv(folder / model / f"{split}_predictions.csv")
                if predictions.image_id.tolist() != expected.image_id.tolist() or predictions.true_label.tolist() != expected.label.tolist():
                    raise ValueError(f"Prediction IDs/labels differ from the frozen split: {folder}/{model}/{split}")
                labels = predictions.true_label.map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()
                scores = score_predictions(labels, predictions[CLASS_NAMES].to_numpy())
                rows.append({"run": run, "model": model, "split": split, **{key: value for key, value in scores.items()
                             if key not in ["per_class", "confusion_matrix"]}})
                for name in CLASS_NAMES:
                    report = scores["per_class"][name]
                    per_class.append({"run": run, "model": model, "split": split, "class": name,
                                      "support": report["support"], "precision": report["precision"], "recall": report["recall"],
                                      "f1": report["f1-score"], "f05": report["f0.5-score"]})
    table = pd.DataFrame(rows)
    table.to_csv(candidate / "comparison_with_baseline.csv", index=False)
    pd.DataFrame(per_class).to_csv(candidate / "per_class_comparison.csv", index=False)
    validation = table[table.split == "validation"].sort_values(["macro_f1", "balanced_accuracy"], ascending=False, kind="stable")
    validation.to_csv(candidate / "validation_ranking.csv", index=False)
    winner = validation.iloc[0]
    selection = {"run": winner.run, "model": winner.model, "selected_using": "validation macro-F1 only",
                 "validation_macro_f1": float(winner.macro_f1), "validation_macro_f05": float(winner.macro_f05)}
    (candidate / "comparison_recommendation.json").write_text(json.dumps(selection, indent=2))
    lines = ["# SMOTE comparison with the original models", "",
             f"Validation-selected model across both runs: **{winner.run} / {winner.model}**.", "",
             "F1 and F0.5 are macro averages across seven classes; weighted and per-class scores are in the CSV files.",
             "The baseline used class weighting. This run uses moderate SMOTE and unweighted losses, comparing two balancing strategies.",
             "The original feature cache, scaler fit data, lesion split, validation images, and test images are unchanged.",
             "Baseline checkpoints and reports were not modified; F0.5 was computed from their saved predictions.", ""]
    for split, title in [("validation", "Internal validation"), ("test", "Internal held-out test"),
                         ("official_validation", "Official validation (additional evaluation only)"),
                         ("official_test", "Official test")]:
        subset = table[table.split == split]
        if subset.empty:
            continue
        lines += [f"## {title}", "", "| Strategy | Model | F1 | F0.5 | Accuracy |", "|---|---|---:|---:|---:|"]
        for row in subset.to_dict("records"):
            lines.append(f"| {row['run']} | {row['model']} | {row['macro_f1']:.4f} | {row['macro_f05']:.4f} | {row['accuracy']:.4f} |")
        lines.append("")
    lines += ["This is a single-seed comparison, not evidence of statistical significance. Model settings were selected on internal validation, not test scores.", ""]
    (candidate / "BASELINE_COMPARISON.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, split, metric, title in zip(axes, ["validation", "test", "test"], ["macro_f1", "macro_f1", "macro_f05"],
                                         ["Validation F1", "Test F1", "Test F0.5"]):
        x = np.arange(3)
        for offset, run, color in [(-0.18, "class_weighted", "#2563eb"), (0.18, "smote", "#ea580c")]:
            values = table[(table.split == split) & (table.run == run)].set_index("model").loc[["linear", "mlp", "xgboost"], metric]
            axis.bar(x + offset, values, width=0.36, label=run, color=color)
        axis.set(xticks=x, xticklabels=["Linear", "MLP", "XGBoost"], ylim=(0, 1), title=title, ylabel="Macro score")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(candidate / "baseline_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(selection, indent=2), flush=True)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="outputs/feature_models/efficientnet_b0")
    parser.add_argument("--candidate", default="outputs/feature_models/efficientnet_b0_smote")
    args = parser.parse_args()
    compare_runs(ROOT / args.baseline, ROOT / args.candidate)
