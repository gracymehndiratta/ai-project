"""Fair centralized comparison of linear, MLP, and XGBoost classifiers."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, fbeta_score, log_loss, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from src.data.full_isic import CLASS_NAMES
from src.models.feature_models import build_mlp, classifier_probabilities, load_classifier


def score_predictions(labels, probabilities):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(labels), 7) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid class probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Class probabilities must be between zero and one")
    # CSV round trips preserve float32 values with tiny row-sum error. Normalize
    # validated near-unit rows in float64 before log-loss/AUC calculations.
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    result = {
        "images": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=range(7), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=range(7), average="weighted", zero_division=0)),
        "macro_f05": float(fbeta_score(labels, predictions, beta=0.5, labels=range(7), average="macro", zero_division=0)),
        "weighted_f05": float(fbeta_score(labels, predictions, beta=0.5, labels=range(7), average="weighted", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=range(7))),
        "per_class": classification_report(labels, predictions, labels=range(7), target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=range(7)).tolist(),
    }
    per_class_f05 = fbeta_score(labels, predictions, beta=0.5, labels=range(7), average=None, zero_division=0)
    for name, value in zip(CLASS_NAMES, per_class_f05):
        result["per_class"][name]["f0.5-score"] = float(value)
    result["per_class"]["macro avg"]["f0.5-score"] = result["macro_f05"]
    result["per_class"]["weighted avg"]["f0.5-score"] = result["weighted_f05"]
    if len(np.unique(labels)) == 7:
        result["macro_roc_auc_ovr"] = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    return result


def _validation_key(result):
    return result["macro_f1"], result["balanced_accuracy"]


def train_linear(train_x, train_y, val_x, val_y, folder, seed=42, class_weighted=True):
    trials = []
    best, best_score = None, (-1, -1)
    with threadpool_limits(limits=4):
        for strength in [0.01, 0.1, 1.0]:
            model = LogisticRegression(C=strength, class_weight="balanced" if class_weighted else None, max_iter=1500,
                                       solver="lbfgs", random_state=seed)
            model.fit(train_x, train_y)
            score = score_predictions(val_y, model.predict_proba(val_x))
            trials.append({"C": strength, "validation_macro_f1": score["macro_f1"],
                           "iterations": int(model.n_iter_.max())})
            print(f"Linear C={strength}: validation macro-F1={score['macro_f1']:.4f}", flush=True)
            if _validation_key(score) > best_score:
                best, best_score = model, _validation_key(score)
    joblib.dump(best, folder / "model.joblib")
    return {"trials": trials, "selected_C": best.C, "trainable_parameters": int(best.coef_.size + best.intercept_.size)}


def train_mlp(train_x, train_y, val_x, val_y, folder, seed=42, max_epochs=150, patience=20, class_weighted=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp(train_x.shape[1]).to(device)
    train_data = TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=generator)
    weights = None
    if class_weighted:
        weights = torch.as_tensor(compute_class_weight("balanced", classes=np.arange(7), y=train_y),
                                  dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.5)
    val_tensor = torch.as_tensor(val_x, device=device)
    best_score, best_state, stale, best_epoch = (-1, -1), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_total = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(labels)
        model.eval()
        with torch.inference_mode():
            probabilities = model(val_tensor).softmax(1).cpu().numpy()
        score = score_predictions(val_y, probabilities)
        scheduler.step(score["macro_f1"])
        history.append({"epoch": epoch, "train_loss": loss_total / len(train_y),
                        "validation_macro_f1": score["macro_f1"], "validation_accuracy": score["accuracy"]})
        if _validation_key(score) > best_score:
            best_score = _validation_key(score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale, best_epoch = 0, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"MLP epoch {epoch}: loss={history[-1]['train_loss']:.4f}, validation macro-F1={score['macro_f1']:.4f}", flush=True)
        if stale >= patience:
            break
    torch.save(best_state, folder / "model.pt")
    parameters = sum(p.numel() for p in model.parameters())
    del loader, model, val_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"architecture": [train_x.shape[1], 256, 64, 7], "trainable_parameters": parameters,
            "dropout": [0.4, 0.25], "optimizer": "AdamW", "learning_rate": 1e-3,
            "weight_decay": 1e-3, "device": str(device), "best_epoch": best_epoch, "history": history}


def train_xgboost(train_x, train_y, val_x, val_y, folder, seed=42, estimators=1200, class_weighted=True):
    device = "cuda" if torch.cuda.is_available() and xgb.build_info().get("USE_CUDA") else "cpu"
    sample_weights = compute_sample_weight("balanced", train_y) if class_weighted else None
    best, best_score, trials = None, (-1, -1), []
    for depth in [3, 5]:
        model = xgb.XGBClassifier(
            n_estimators=estimators, max_depth=depth, learning_rate=0.04,
            min_child_weight=5, subsample=0.85, colsample_bytree=0.7,
            reg_lambda=5.0, reg_alpha=0.1, objective="multi:softprob", num_class=7,
            tree_method="hist", device=device, n_jobs=4, max_bin=128,
            eval_metric="mlogloss", early_stopping_rounds=50, random_state=seed,
        )
        print(f"Training XGBoost depth={depth} on {device}", flush=True)
        model.fit(train_x, train_y, sample_weight=sample_weights, eval_set=[(val_x, val_y)], verbose=False)
        # Use CPU prediction to avoid repeated implicit GPU transfers for NumPy inputs.
        model.set_params(device="cpu")
        score = score_predictions(val_y, model.predict_proba(val_x))
        trials.append({"max_depth": depth, "best_iteration": int(model.best_iteration),
                       "validation_macro_f1": score["macro_f1"], "validation_loss_history": model.evals_result()})
        print(f"XGBoost depth={depth}: validation macro-F1={score['macro_f1']:.4f}, best round={model.best_iteration}", flush=True)
        if _validation_key(score) > best_score:
            best, best_score = model, _validation_key(score)
    best.save_model(folder / "model.ubj")
    return {"device": device, "trials": trials, "selected_max_depth": best.max_depth,
            "selected_iteration": int(best.best_iteration), "tree_method": "hist"}


def resample_training(train_x, train_y, method="none", target=500, seed=42):
    """Only original training rows are accepted here; holdouts never enter SMOTE."""
    if method not in ("none", "smote") or target < 1:
        raise ValueError("Resampling must be none/smote and the target must be positive")
    counts = np.bincount(train_y, minlength=7)
    info = {"method": method, "original_counts": dict(zip(CLASS_NAMES, counts.astype(int).tolist())),
            "original_rows": len(train_y), "random_seed": seed,
            "class_weighted": method == "none"}
    if method == "smote":
        import imblearn
        from imblearn.over_sampling import SMOTE
        strategy = {i: target for i, count in enumerate(counts) if count < target}
        if strategy and min(counts[i] for i in strategy) < 2:
            raise ValueError("SMOTE needs at least two real training examples in each targeted class")
        neighbors = min(5, min(counts[i] for i in strategy) - 1) if strategy else 5
        if strategy:
            with threadpool_limits(limits=4):
                train_x, train_y = SMOTE(sampling_strategy=strategy, k_neighbors=int(neighbors),
                                        random_state=seed).fit_resample(train_x, train_y)
        info.update(target_per_minority_class=target, k_neighbors=int(neighbors), imbalanced_learn_version=imblearn.__version__,
                    note="Synthetic feature vectors only; no synthetic images or holdout resampling. Class weights disabled.")
    train_x = np.asarray(train_x, dtype=np.float32)
    info.update(resampled_counts=dict(zip(CLASS_NAMES, np.bincount(train_y, minlength=7).astype(int).tolist())),
                resampled_rows=len(train_y), synthetic_rows=len(train_y) - info["original_rows"],
                training_features_sha256=hashlib.sha256(train_x.tobytes()).hexdigest(),
                training_labels_sha256=hashlib.sha256(np.asarray(train_y, dtype=np.int64).tobytes()).hexdigest())
    return train_x, train_y, info


def train_all(features, manifest, cache_info, output_dir, seed=42, max_epochs=150, xgb_estimators=1200,
              resampling="none", smote_target=500):
    """Fit preprocessing on train only; choose settings without accessing test labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = output_dir / "experiment.json"
    variant = "smote" if resampling == "smote" else "class_weighted"
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved["fingerprint"] != cache_info["fingerprint"] or saved["seed"] != seed:
            raise ValueError("Output directory belongs to a different dataset/split. Choose --output-dir.")
        if saved.get("training_variant", "class_weighted") != variant:
            raise ValueError("Output directory belongs to a different training variant. Choose --output-dir.")
        if resampling == "smote" and saved.get("resampling", {}).get("target_per_minority_class") != smote_target:
            raise ValueError("Output directory belongs to a different SMOTE target. Choose --output-dir.")
    if not np.isfinite(features).all():
        raise ValueError("Non-finite features")
    train_idx = np.flatnonzero(manifest.split == "train")
    val_idx = np.flatnonzero(manifest.split == "validation")
    labels = manifest.label.map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()
    if pd.isna(labels).any() or set(labels[train_idx]) != set(range(7)):
        raise ValueError("Training data must contain all seven known classes")
    scaler = StandardScaler().fit(features[train_idx])
    joblib.dump(scaler, output_dir / "scaler.joblib")
    scaled = scaler.transform(features).astype(np.float32)
    train_x, train_y, sampling_info = resample_training(scaled[train_idx], labels[train_idx], resampling, smote_target, seed)
    (output_dir / "resampling.json").write_text(json.dumps(sampling_info, indent=2))
    print(f"Training strategy: {variant}; {sampling_info['original_rows']} real training rows + "
          f"{sampling_info['synthetic_rows']} synthetic feature vectors", flush=True)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    experiment = {**cache_info, "seed": seed, "class_names": CLASS_NAMES,
                  "training_variant": variant, "resampling": sampling_info,
                  "selection_metric": "validation macro-F1; balanced accuracy breaks ties",
                  "training_mode": "centralized, cached frozen features; no DP or federated claims",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "split_images": manifest.split.value_counts().astype(int).to_dict(),
                  "training_lesions": int(manifest[manifest.source_partition == "training"].lesion_id.nunique()),
                  "external_grouping_note": "Official external sets have no lesion IDs in the supplied grouping file; no cross-set lesion-disjointness claim.",
                  "versions": {"torch": torch.__version__, "xgboost": xgb.__version__}}
    existing.write_text(json.dumps(experiment, indent=2))
    trainers = {"linear": train_linear, "mlp": train_mlp, "xgboost": train_xgboost}
    summary = []
    for kind, trainer in trainers.items():
        folder = output_dir / kind
        folder.mkdir(exist_ok=True)
        start = time.monotonic()
        kwargs = {"max_epochs": max_epochs} if kind == "mlp" else {"estimators": xgb_estimators} if kind == "xgboost" else {}
        details = trainer(train_x, train_y, scaled[val_idx], labels[val_idx], folder, seed,
                          class_weighted=sampling_info["class_weighted"], **kwargs)
        info = {"format_version": 1, "kind": kind, "feature_dimension": int(features.shape[1]),
                "class_names": CLASS_NAMES, "backbone": cache_info["backbone"], "weights": cache_info["weights"],
                "fingerprint": cache_info["fingerprint"], "seed": seed, "training_seconds": time.monotonic() - start,
                "training_variant": variant, "class_weighted": sampling_info["class_weighted"],
                "details": details}
        (folder / "metadata.json").write_text(json.dumps(info, indent=2))
        # Reload the saved artifact; metrics verify the actual deliverable.
        classifier, _, _ = load_classifier(folder)
        results = {}
        for split in ["validation", "test", "official_validation", "official_test"]:
            indices = np.flatnonzero(manifest.split == split)
            if not len(indices):
                continue
            probabilities = classifier_probabilities(classifier, scaled[indices], kind)
            results[split] = score_predictions(labels[indices], probabilities)
            records = pd.DataFrame(probabilities, columns=CLASS_NAMES)
            records.insert(0, "true_label", manifest.iloc[indices].label.to_numpy())
            records.insert(0, "image_id", manifest.iloc[indices].image_id.to_numpy())
            records.to_csv(folder / f"{split}_predictions.csv", index=False)
            cm = np.array(results[split]["confusion_matrix"])
            fig, axis = plt.subplots(figsize=(7, 6))
            image = axis.imshow(cm, cmap="Blues")
            axis.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                     xlabel="Predicted", ylabel="Actual", title=f"{kind}: {split}")
            for y in range(7):
                for x in range(7):
                    axis.text(x, y, str(cm[y, x]), ha="center", va="center", color="white" if cm[y, x] > cm.max()/2 else "black")
            fig.colorbar(image, ax=axis)
            fig.tight_layout()
            fig.savefig(folder / f"{split}_confusion_matrix.png", dpi=150)
            plt.close(fig)
        (folder / "metrics.json").write_text(json.dumps(results, indent=2))
        row = {"model": kind, "training_seconds": info["training_seconds"]}
        for split, metrics in results.items():
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_f05", "weighted_f05"]:
                row[f"{split}_{metric}"] = metrics[metric]
        summary.append(row)
        pd.DataFrame(summary).to_csv(output_dir / "comparison.csv", index=False)
        print(f"Completed {kind}: validation macro-F1={results['validation']['macro_f1']:.4f}; test macro-F1={results['test']['macro_f1']:.4f}", flush=True)
    table = pd.DataFrame(summary)
    best = table.sort_values(["validation_macro_f1", "validation_balanced_accuracy"], ascending=False).iloc[0]["model"]
    (output_dir / "best_model.json").write_text(json.dumps({"model": best, "selected_using": "validation only"}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, split in zip(axes, ["validation", "test"]):
        axis.bar(table.model, table[f"{split}_macro_f1"], color=["#2563eb", "#7c3aed", "#059669"])
        axis.set(ylim=(0, 1), ylabel="Macro F1", title=f"{split.title()} (same split for all models)")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Frozen-feature model comparison", "", f"Validation-selected model: **{best}**.", "",
             f"Training strategy: **{variant}**. Real training rows: {sampling_info['original_rows']}; synthetic training feature vectors: {sampling_info['synthetic_rows']}.",
             "All models use the same original pretrained backbone, feature cache, train-only scaler, and lesion-disjoint internal split.",
             "These are centralized research models; no differential-privacy guarantee is claimed.", "",
             "F1 and F0.5 below are macro averages over all seven classes. F0.5 emphasizes precision. Weighted and per-class scores are also saved in metrics.json.", "",
             "| Model | Validation F1 | Validation F0.5 | Test F1 | Test F0.5 | Test accuracy |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['validation_macro_f1']:.4f} | {row['validation_macro_f05']:.4f} | {row['test_macro_f1']:.4f} | {row['test_macro_f05']:.4f} | {row['test_accuracy']:.4f} |")
    if "official_test_macro_f1" in table:
        lines += ["", "## Additional official challenge sets", "",
                  "These sets were evaluated after training and were not used to select model settings.", "",
                  "| Model | Official validation F1 | Official validation F0.5 | Official test F1 | Official test F0.5 | Official test accuracy |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in summary:
            lines.append(f"| {row['model']} | {row['official_validation_macro_f1']:.4f} | {row['official_validation_macro_f05']:.4f} | {row['official_test_macro_f1']:.4f} | {row['official_test_macro_f05']:.4f} | {row['official_test_accuracy']:.4f} |")
    lines += ["", "All 10,015 HAM10000 images are included across the internal training, validation, and test sets.",
              "The additional official validation/test sets are evaluated separately; their lesion grouping metadata is unavailable.",
              "No hyperparameters were selected using either test set.",
              "These are initial baselines from one fixed split/seed, not an exhaustive tuning study or a clinical validation."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return table
