"""Report every specialist and audit comparisons against all historical heads."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/feature_models/efficientnet_b0_specialists")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from src.data.full_isic import CLASS_NAMES
    from src.utils.feature_training import score_predictions
    output = ROOT / args.output_dir
    manifest = pd.read_csv(output / "split_manifest.csv")
    ids = dict(zip(CLASS_NAMES, range(7)))
    runs = {"Original": (output.with_name("efficientnet_b0"), ["linear", "mlp", "xgboost"]),
            "SMOTE": (output.with_name("efficientnet_b0_smote"), ["linear", "mlp", "xgboost"]),
            "Improved": (output.with_name("efficientnet_b0_improved"), ["linear", "mlp", "xgboost"]),
            "Specialists": (output, ["specialists"])}
    rows, per_class, test_matrices = [], [], {}
    test_frame = manifest[manifest.split == "test"]
    _, test_groups = np.unique(test_frame.lesion_id, return_inverse=True)
    for run, (folder, kinds) in runs.items():
        pd.testing.assert_frame_equal(manifest, pd.read_csv(folder / "split_manifest.csv"))
        for kind in kinds:
            for split in ["validation", "test", "official_validation", "official_test"]:
                predictions = pd.read_csv(folder / kind / f"{split}_predictions.csv")
                expected = manifest[manifest.split == split]
                assert predictions.image_id.tolist() == expected.image_id.tolist()
                assert predictions.true_label.tolist() == expected.label.tolist()
                labels = predictions.true_label.map(ids).to_numpy()
                probabilities = predictions[CLASS_NAMES].to_numpy()
                metrics = score_predictions(labels, probabilities)
                row = {"run": run, "model": kind, "split": split}
                for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_f05", "weighted_f1", "weighted_f05", "log_loss"]:
                    row[metric] = metrics[metric]
                for average in ["macro", "weighted"]:
                    for metric in ["precision", "recall"]:
                        row[f"{average}_{metric}"] = metrics["per_class"][f"{average} avg"][metric]
                rows.append(row)
                for name in CLASS_NAMES:
                    per_class.append({"run": run, "model": kind, "split": split, "lesion": name,
                                      **metrics["per_class"][name]})
                if split == "test":
                    matrices = np.zeros((test_groups.max() + 1, 7, 7), dtype=np.int64)
                    np.add.at(matrices, (test_groups, labels, probabilities.argmax(1)), 1)
                    test_matrices[f"{run}/{kind}"] = matrices
    history = pd.DataFrame(rows)
    history.to_csv(output / "historical_comparison.csv", index=False)
    recommendation = history[history.split == "validation"].sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]
    (output / "recommendation.json").write_text(json.dumps({"run": recommendation.run,
        "model": recommendation.model, "validation_macro_f1": float(recommendation.macro_f1),
        "selected_using": "Internal validation only; all previous models retained"}, indent=2))
    pd.DataFrame(per_class).to_csv(output / "historical_per_class.csv", index=False)
    # Paired resampling of lesions, maintaining correlation of views/images.
    current = test_matrices["Specialists/specialists"]
    rng = np.random.default_rng(42)
    choices = rng.integers(0, len(current), (2000, len(current)))
    counts = np.zeros((len(choices), len(current)), dtype=np.int32)
    for i, sample in enumerate(choices):
        counts[i] = np.bincount(sample, minlength=len(current))

    def bootstrap_f1(matrices):
        cm = (counts @ matrices.reshape(len(matrices), -1)).reshape(-1, 7, 7)
        denominator = cm.sum(-1) + cm.sum(-2)
        return np.divide(2 * np.diagonal(cm, axis1=-2, axis2=-1), denominator,
                         out=np.zeros_like(denominator, dtype=float), where=denominator > 0).mean(-1)

    current_bootstrap = bootstrap_f1(current)
    differences = []
    current_f1 = history[(history.run == "Specialists") & (history.split == "test")].iloc[0].macro_f1
    for label, matrices in test_matrices.items():
        if label == "Specialists/specialists":
            continue
        run, kind = label.split("/")
        previous = history[(history.run == run) & (history.model == kind) & (history.split == "test")].iloc[0]
        differences.append({"baseline": label, "internal_test_f1_change": float(current_f1 - previous.macro_f1),
            "paired_lesion_bootstrap_95_percent_interval": np.quantile(current_bootstrap - bootstrap_f1(matrices), [0.025, 0.975]).tolist(),
            "replicates": 2000})
    (output / "bootstrap_differences.json").write_text(json.dumps(differences, indent=2))
    selections = [json.loads((output / "lesion_models" / name / "selection.json").read_text()) for name in CLASS_NAMES]
    settings = pd.DataFrame([{"lesion": record["lesion"], "family": record["config"]["family"],
        "label_smoothing": record["config"]["smoothing"], "calibration_C": record["calibration_C"],
        "threshold": record["final_threshold"], "training_positives": record["positive_training_images"],
        "training_positive_lesions": record["positive_training_lesions"], "cv_average_precision": record["mean_fold_average_precision"],
        "training_seconds": record["training_seconds"]} for record in selections])
    settings.to_csv(output / "selected_specialists.csv", index=False)
    smoothing_rows = []
    for name in CLASS_NAMES:
        trials = json.loads((output / "lesion_models" / name / "trials.json").read_text())
        for trial in trials:
            if trial["config"]["family"] == "mlp":
                smoothing_rows.append({"lesion": name, "smoothing": trial["config"]["smoothing"],
                    "cv_average_precision": trial["mean_fold_average_precision"],
                    "oof_selection_f1": trial["oof_selection_metrics"]["f1"],
                    "oof_selection_f05": trial["oof_selection_metrics"]["f05"],
                    "oof_selection_log_loss": trial["oof_selection_metrics"]["log_loss"]})
    smoothing = pd.DataFrame(smoothing_rows)
    smoothing.to_csv(output / "neural_smoothing_comparison.csv", index=False)
    binary = pd.read_csv(output / "binary_specialist_metrics.csv")
    combined = pd.read_csv(output / "combined_per_class_metrics.csv")
    binary_melanoma_recall = float(binary[(binary.split == "official_test") & (binary.lesion == "mel")].iloc[0].recall)
    combined_melanoma_recall = float(combined[(combined.split == "official_test") & (combined.lesion == "mel")].iloc[0].recall)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    palette = {"Original": "#94a3b8", "SMOTE": "#f59e0b", "Improved": "#2563eb", "Specialists": "#16a34a"}
    for ax, split in zip(axes, ["test", "official_test"]):
        table = history[history.split == split].reset_index(drop=True)
        bars = ax.barh(np.arange(len(table)), table.macro_f1, color=[palette[run] for run in table.run])
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.set(yticks=np.arange(len(table)), yticklabels=[f"{row.run}: {row.model}" for _, row in table.iterrows()],
               xlim=(0, 0.9), xlabel="Macro F1", title="Internal test (1,492 images)" if split == "test" else "Official test (1,512 images)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    metrics = json.loads((output / "specialists/metrics.json").read_text())
    for split in ["test", "official_test"]:
        cm = np.array(metrics[split]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="Predicted", ylabel="Actual", title=f"Seven specialists: {split}")
        fig.tight_layout()
        fig.savefig(output / "specialists" / f"{split}_confusion_matrix.png", dpi=150)
        plt.close(fig)
    selection = json.loads((output / "selection_complete.json").read_text())["selected_fusion"]
    lines = ["# Seven calibrated lesion specialists", "",
        f"Validation-selected recommendation across all runs: **{recommendation.run} / {recommendation.model}**, validation macro F1 **{recommendation.macro_f1:.4f}**. The specialist system's validation F1 is **{metrics['validation']['macro_f1']:.4f}**.",
        "The specialist system does not outperform the strongest previous model on overall macro F1 in this experiment. It remains available for comparison, with every result below retained.", "",
        "All seven independently selected binary specialists are trained. The complete 11,720-image release and all historical split assignments are retained; each final head fits on all 7,014 training images.", "",
        "## Selection and calibration", "",
        "For each lesion, 11 candidates (four logistic regressions, three neural networks and four XGBoost models) are compared using three lesion-grouped training folds. Every candidate fold separates base fitting, calibration and scoring lesions. Scalers are fitted on base-fitting rows only. Mean fold average precision selects the classifier, with OOF F1 as tie-breaker; calibration regularization is selected by OOF log loss.",
        "Each lesion's neural candidates independently test smoothing 0, 0.03 and 0.10, shrinking binary targets towards that fit-fold's training prevalence. This avoids adding the same artificial positive fraction to common and rare classes. Linear and tree models do not use neural label smoothing; their regularization and calibration are selected separately.",
        "After selection, the winner generates logits on independent training lesion folds to fit its sigmoid calibrator, then the base model is refitted on all 7,014 training images. Per-lesion thresholds maximize F1 on calibrated training OOF scores. OOF figures are selection diagnostics, not a new unbiased performance estimate.",
        f"Fusion strength selected using internal validation only: **{selection['strength']}**. For each image, scores are proportional to calibrated binary probability divided by its training-selected threshold raised to this strength. Row normalization alone does not change the winning class. These normalized decision scores are not calibrated multiclass posterior probabilities.",
        "All seven models, calibration parameters, thresholds and the fusion strength were saved before either test set was evaluated. No metadata or new images were added. Test sets have been reported previously, so this is not a fresh prospective confirmation.", "",
        "## Selected specialist settings", "", "| Lesion | Family | Label smoothing | Calibration C | Binary threshold | Training positives / lesions |", "|---|---|---:|---:|---:|---:|"]
    for _, row in settings.iterrows():
        lines.append(f"| {row.lesion} | {row.family} | {row.label_smoothing:.2f} | {row.calibration_C:g} | {row.threshold:.4f} | {row.training_positives} / {row.training_positive_lesions} |")
    lines += ["", "### Per-lesion neural smoothing comparison", "",
        "These are training cross-validation selection diagnostics, not test results. Each lesion's best neural smoothing was chosen independently. A non-neural final winner does not use label smoothing.", "",
        "| Lesion | Neural CV AP: smoothing 0 | Neural CV AP: 0.03 | Neural CV AP: 0.10 | Best neural smoothing |", "|---|---:|---:|---:|---:|"]
    for name in CLASS_NAMES:
        table = smoothing[smoothing.lesion == name].set_index("smoothing")
        best_smoothing = table.cv_average_precision.idxmax()
        lines.append(f"| {name} | {table.loc[0, 'cv_average_precision']:.4f} | {table.loc[0.03, 'cv_average_precision']:.4f} | {table.loc[0.10, 'cv_average_precision']:.4f} | {best_smoothing:.2f} |")
    lines += ["", "## Combined seven-class comparison", "",
        "F1 and F0.5 here are macro averages across the seven mutually exclusive diagnoses. Precision and recall are also macro averages. All nine historical model runs are retained and recomputed from their saved predictions."]
    for split in ["test", "official_test", "validation", "official_validation"]:
        lines += ["", f"### {split}", "", "| Run | Model | Accuracy | Macro F1 | Macro F0.5 | Precision | Recall | Balanced accuracy |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for _, row in history[history.split == split].iterrows():
            lines.append(f"| {row.run} | {row.model} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.macro_f05:.4f} | {row.macro_precision:.4f} | {row.macro_recall:.4f} | {row.balanced_accuracy:.4f} |")
    lines += ["", "## Each specialist's independent binary decisions", "",
        "These thresholds can independently identify zero, one or several classes for an image. This table therefore differs from the combined argmax classifier. Binary accuracy counts all other diagnoses as negatives and can be high for rare classes; use F1, precision, recall, specificity and balanced accuracy alongside it."]
    for split in ["test", "official_test"]:
        lines += ["", f"### {split}", "", "| Lesion | Model | Accuracy | F1 | F0.5 | Precision | Recall | Specificity | Balanced accuracy | ROC AUC | Average precision |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, row in binary[binary.split == split].iterrows():
            lines.append(f"| {row.lesion} | {row.family} | {row.accuracy:.4f} | {row.f1:.4f} | {row.f05:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.specificity:.4f} | {row.balanced_accuracy:.4f} | {row.roc_auc:.4f} | {row.average_precision:.4f} |")
    lines += ["", "## Per-class results after the combined decision", "",
        "Each image receives exactly one diagnosis here. One-vs-rest accuracy is distinct from the overall seven-class accuracy above."]
    for split in ["test", "official_test"]:
        lines += ["", f"### {split}", "", "| Lesion | Precision | Recall | F1 | F0.5 | One-vs-rest accuracy | Support |", "|---|---:|---:|---:|---:|---:|---:|"]
        for _, row in combined[combined.split == split].iterrows():
            lines.append(f"| {row['lesion']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {row['f0.5-score']:.4f} | {row['one_vs_rest_accuracy']:.4f} | {int(row['support'])} |")
    lines += ["", "## Files and uncertainty", "",
        f"The seven independent binary decisions and the final argmax solve different decision problems. For example, official-test melanoma recall is {binary_melanoma_recall:.4f} at its binary threshold but {combined_melanoma_recall:.4f} after the seven-class winner is selected. This shows where binary detection gains are lost through class competition; it does not establish that a particular smoothing or calibration choice caused the overall regression. More models and score normalization do not guarantee better combined classification.",
        "binary_specialist_metrics.csv contains every specialist's metrics for all four held-out sets, including confusion counts, ROC AUC, average precision, Brier score, log loss, and F2. combined_per_class_metrics.csv contains the mutually exclusive per-class results. historical_comparison.csv and historical_per_class.csv cover all prior models. bootstrap_differences.json gives paired internal-test macro-F1 differences with 2,000 lesion-bootstrap replicates; an interval crossing zero does not establish a clear improvement.",
        "Rare diagnoses have few independent lesions. Outer folds preserve lesion groups but cannot assert patient disjointness because patient identifiers are unavailable. Official external sets lack grouping IDs in the downloaded files. These are centralized research models with no differential-privacy or clinical-validation claim.", "",
        "## Reproduce", "", "```powershell", ".venv\\Scripts\\python.exe scripts/train_specialists.py", ".venv\\Scripts\\python.exe scripts/report_specialists.py", "```", "",
        "Completed lesion checkpoints are resumable when the dataset fingerprint and protocol match. All previous trained artifacts remain in their original folders."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("COMBINED TEST RESULTS\n" + history[history.split.isin(["test", "official_test"])].to_string(index=False))
    print("\nSPECIALIST SETTINGS\n" + settings.to_string(index=False))
    print("\nBINARY OFFICIAL TEST RESULTS\n" + binary[binary.split == "official_test"].to_string(index=False))


if __name__ == "__main__":
    main()
