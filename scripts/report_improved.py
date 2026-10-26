"""Audit and report finalized model comparisons, including lesion-bootstrap uncertainty."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from src.data.full_isic import CLASS_NAMES
    from src.utils.feature_training import score_predictions
    output = ROOT / "outputs/feature_models/efficientnet_b0_improved"
    runs = {"Original": output.with_name("efficientnet_b0"), "SMOTE": output.with_name("efficientnet_b0_smote"), "Improved": output}
    manifest = pd.read_csv(output / "split_manifest.csv")
    for folder in runs.values():
        previous = pd.read_csv(folder / "split_manifest.csv")
        pd.testing.assert_frame_equal(manifest, previous)
    rows, per_class, differences = [], [], []
    manifest_test = manifest[manifest.split == "test"].set_index("image_id")
    ids = dict(zip(CLASS_NAMES, range(7)))

    def group_confusions(predictions):
        y = predictions.true_label.map(ids).to_numpy()
        predicted = predictions[CLASS_NAMES].to_numpy().argmax(1)
        groups = manifest_test.loc[predictions.image_id].lesion_id.to_numpy()
        _, inverse = np.unique(groups, return_inverse=True)
        matrices = np.zeros((inverse.max() + 1, 7, 7), dtype=np.int64)
        np.add.at(matrices, (inverse, y, predicted), 1)
        return matrices

    def macro_from_cm(cm):
        denominator = cm.sum(-1) + cm.sum(-2)
        scores = np.divide(2 * np.diagonal(cm, axis1=-2, axis2=-1), denominator,
                           out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
        return scores.mean(-1)

    for kind in ["linear", "mlp", "xgboost"]:
        predictions_by_run = {}
        for name, folder in runs.items():
            metrics = json.loads((folder / kind / "metrics.json").read_text())
            for split in ["validation", "test", "official_validation", "official_test"]:
                predictions = pd.read_csv(folder / kind / f"{split}_predictions.csv")
                expected = manifest[manifest.split == split]
                assert predictions.image_id.tolist() == expected.image_id.tolist()
                assert predictions.true_label.tolist() == expected.label.tolist()
                recomputed = score_predictions(predictions.true_label.map(ids).to_numpy(), predictions[CLASS_NAMES].to_numpy())
                assert abs(recomputed["macro_f1"] - metrics[split]["macro_f1"]) < 1e-12
                rows.append({"run": name, "model": kind, "split": split,
                             **{metric: recomputed[metric] for metric in ["macro_f1", "macro_f05", "accuracy", "balanced_accuracy"]}})
                for label in CLASS_NAMES:
                    per_class.append({"run": name, "model": kind, "split": split, "class": label, **recomputed["per_class"][label]})
                if split == "test":
                    predictions_by_run[name] = predictions
        current = group_confusions(predictions_by_run["Improved"])
        rng = np.random.default_rng(42)
        samples = rng.integers(0, len(current), (2000, len(current)))
        current_scores = macro_from_cm(current[samples].sum(1))
        for previous in ["Original", "SMOTE"]:
            earlier = group_confusions(predictions_by_run[previous])
            delta = current_scores - macro_from_cm(earlier[samples].sum(1))
            differences.append({"model": kind, "compared_with": previous,
                "test_f1_difference": float(macro_from_cm(current.sum(0)) - macro_from_cm(earlier.sum(0))),
                "lesion_bootstrap_95_percent_interval": np.quantile(delta, [0.025, 0.975]).tolist(), "replicates": 2000})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "historical_comparison.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "per_class_comparison.csv", index=False)
    (output / "bootstrap_differences.json").write_text(json.dumps(differences, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#94a3b8", "#f59e0b", "#16a34a"]
    for ax, split in zip(axes, ["test", "official_test"]):
        table = comparison[comparison.split == split]
        for index, name in enumerate(runs):
            scores = table[table.run == name].set_index("model").loc[["linear", "mlp", "xgboost"], "macro_f1"]
            bars = ax.bar(np.arange(3) + (index - 1) * 0.25, scores, width=0.25, label=name, color=colors[index])
            ax.bar_label(bars, fmt="%.3f", fontsize=8, rotation=90, padding=3)
        ax.set(xticks=np.arange(3), xticklabels=["Linear", "Neural network", "XGBoost"], ylim=(0, 0.9),
               ylabel="Macro F1", title="Internal test" if split == "test" else "Official challenge test")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "historical_comparison.png", dpi=170)
    plt.close(fig)
    for kind in ["linear", "mlp", "xgboost"]:
        metrics = json.loads((output / kind / "metrics.json").read_text())
        for split in ["test", "official_test"]:
            cm = np.array(metrics[split]["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(cm, cmap="Blues")
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   xlabel="Predicted", ylabel="Actual", title=f"Improved {kind}: {split}")
            fig.tight_layout()
            fig.savefig(output / kind / f"{split}_confusion_matrix.png", dpi=150)
            plt.close(fig)
    best = json.loads((output / "best_model.json").read_text())["model"]
    lines = ["# Improved frozen EfficientNet training", "", f"Validation-selected model: **{best}**.", "",
        "All three improve macro F1 on the official challenge test set compared with both prior runs. Internal-test results are mixed: the new MLP remains below its original F1, and linear remains below its SMOTE F1. This is not an across-the-board improvement on every metric or split.", "",
        "All 11,720 release images are present: 7,014 training, 1,509 internal validation, 1,492 internal test, 193 official validation and 1,512 official test.",
        "Training, validation and internal test are lesion-disjoint. Official sets lack lesion IDs, so cross-set lesion disjointness is not asserted.", "",
        "## What changed", "",
        "EfficientNet-B0 remains frozen. Each image contributes the canonical ImageNet crop and eight square symmetries of the resized full image. The latter are averaged, then concatenated with canonical features (2,560 values). This keeps full-image information and reduces orientation sensitivity.",
        "Scaler and optional PCA are fitted on the 7,014 real training images only. No SMOTE is used. A predeclared grid separately selects representation, class-weight strength and regularization for each model. MLP uses label smoothing, stronger weight decay, early stopping and independent seeds. XGBoost early-stops on negative macro F1, matching selection. A three-candidate same-family ensemble is retained only if its validation macro F1 improves on the best single candidate.",
        "A small validation-selected class-prior adjustment targets macro F1; these output scores are not calibrated clinical probabilities. The linear baseline remains a single logistic regression. Only MLP/XGBoost can use same-family ensembles.", "",
        "All three configurations were saved before any new test prediction was evaluated (selection_complete.json). Test labels never choose hyperparameters, ensemble members or the recommended model.", "",
        "## Comparison", "", "F1 and F0.5 are macro averages over seven classes. F0.5 gives precision greater weight.", "",
        "| Model | Run | Validation F1 | Internal test F1 | Internal test F0.5 | Official test F1 | Official test F0.5 |", "|---|---|---:|---:|---:|---:|---:|"]
    for kind in ["linear", "mlp", "xgboost"]:
        for name in runs:
            table = comparison[(comparison.model == kind) & (comparison.run == name)].set_index("split")
            lines.append(f"| {kind} | {name} | {table.loc['validation', 'macro_f1']:.4f} | {table.loc['test', 'macro_f1']:.4f} | {table.loc['test', 'macro_f05']:.4f} | {table.loc['official_test', 'macro_f1']:.4f} | {table.loc['official_test', 'macro_f05']:.4f} |")
    lines += ["", "## Why SMOTE regressed", "",
        "The previous comparison changed both oversampling and class weighting, so attributing all losses to SMOTE alone was unsupported. SMOTE also only raised rare classes to 500 while leaving 4,690 nevi; it did not fully balance training. Interpolation adds vectors, not new lesion appearances. The old XGBoost stopping criterion optimized log loss rather than macro F1.",
        "The controlled 2x2 validation-only ablation in outputs/diagnostics/smote_ablation/comparison.csv separates oversampling and weighting while keeping the old representations, hyperparameter grids and seeds fixed. Restoring class weights with SMOTE recovers XGBoost validation F1 from 0.4767 to 0.5124 (original weighted: 0.5146), and raises MLP from 0.5574 to 0.5672 (original weighted: 0.5581). The earlier attribution to SMOTE alone was wrong: removing weighting explains most of the XGBoost validation regression. These experiments explain this split, not a universal claim that SMOTE is harmful.", "",
        "## Uncertainty", "",
        "bootstrap_differences.json reports paired 2,000-replicate internal-test confidence intervals, resampling lesions rather than correlated images. Against the original runs, all three internal-test intervals include zero; a clear improvement over those originals is not established on this internal split. XGBoost versus SMOTE has an interval above zero. Improvements are not guaranteed on a new hospital population. Repeated validation tuning can overfit, and these historical test sets have been reported before; a future untouched external dataset is needed for a fresh confirmation.",
        "Real-image deployment checks cover one internal-test image per true class for each head (21 checks). All predicted labels agree with cached evaluation. CUDA convolution rounding can vary by batch shape, causing small score differences; observed maxima were about 0.0004 (linear), 0.0014 (MLP), and 0.0083 (XGBoost, where tiny feature changes can cross tree splits). See artifacts/improved-verification.json.", "",
        "## Reproduce", "", "```powershell", ".venv\\Scripts\\python.exe scripts/train_improved.py", ".venv\\Scripts\\python.exe scripts/ablate_smote.py", ".venv\\Scripts\\python.exe scripts/report_improved.py", "```", "",
        "All historical checkpoints are preserved. This is centralized research training with a frozen ImageNet backbone, not differential privacy or federated training."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
