"""Group-separated selection of seven lesion specialists and calibrated fusion."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, fbeta_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import xgboost as xgb

from src.data.full_isic import CLASS_NAMES
from src.models.lesion_specialists import BinaryMLP, BinarySpecialist, SpecialistSystem, binary_logits
from src.utils.feature_training import score_predictions


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"images": len(labels), "positives": int(labels.sum()), "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f05": float(fbeta_score(labels, predictions, beta=0.5, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    # On a tie prefer the more conservative threshold.
    index = np.flatnonzero(values >= values.max() - 1e-12)[-1]
    return float(np.clip(thresholds[index], 1e-6, 1 - 1e-6))


def binary_weights(labels, power):
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("Every training fold needs positive and negative lesions")
    weights = (len(labels) / (2 * counts)) ** power
    return weights / weights[labels].mean()


def fit_binary(values, labels, config, seed):
    weights = binary_weights(labels, config["weight_power"])
    if config["family"] == "linear":
        model = LogisticRegression(C=config["C"], class_weight=dict(enumerate(weights)),
                                   solver="lbfgs", max_iter=1000, random_state=seed)
        with threadpool_limits(limits=4):
            model.fit(values, labels)
        return model
    if config["family"] == "xgboost":
        model = xgb.XGBClassifier(n_estimators=config["trees"], max_depth=config["depth"],
            learning_rate=0.04, min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=10, reg_alpha=0.5, max_bin=128, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss", n_jobs=4, random_state=seed,
            device="cuda" if torch.cuda.is_available() and xgb.build_info().get("USE_CUDA") else "cpu")
        model.fit(values, labels, sample_weight=weights[labels], verbose=False)
        model.set_params(device="cpu")
        return model
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BinaryMLP(values.shape[1]).to(device)
    x = torch.as_tensor(values, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(weights[labels], dtype=torch.float32, device=device)
    # Smooth towards this fold's training prevalence, preserving its marginal
    # class prior rather than injecting 5% positives into a 1% minority class.
    targets = y * (1 - config["smoothing"]) + config["smoothing"] * float(labels.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0.00005)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(config["epochs"]):
        model.train()
        for indices in torch.randperm(len(labels), device=device, generator=generator).split(128):
            optimizer.zero_grad(set_to_none=True)
            losses = nn.functional.binary_cross_entropy_with_logits(model(x[indices]).squeeze(1),
                targets[indices], reduction="none")
            loss = (losses * sample_weights[indices]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        schedule.step()
    model.cpu().eval()
    return model


def candidate_grid(epochs=45, trees=400):
    candidates = []
    for power in [0.5, 1.0]:
        for strength in [0.001, 0.01]:
            candidates.append({"family": "linear", "C": strength, "weight_power": power, "smoothing": 0.0})
    for smoothing in [0.0, 0.03, 0.10]:
        candidates.append({"family": "mlp", "epochs": epochs, "weight_power": 0.5, "smoothing": smoothing})
    for power in [0.5, 1.0]:
        for depth in [2, 3]:
            candidates.append({"family": "xgboost", "trees": trees, "depth": depth,
                               "weight_power": power, "smoothing": 0.0})
    return candidates


def fit_calibrator(scores, labels, strength):
    # No balancing here: calibration estimates the original observed prevalence.
    model = LogisticRegression(C=strength, max_iter=500, solver="lbfgs")
    model.fit(np.asarray(scores).reshape(-1, 1), labels)
    return model


def build_folds(train_frame, folds=3, seed=42):
    """Outer held-out folds never enter a candidate's fit or calibration fit."""
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = train_frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    groups = train_frame.lesion_id.to_numpy()
    result = []
    for index, (outer_train, outer_holdout) in enumerate(splitter.split(train_frame, labels, groups)):
        unique = train_frame.iloc[outer_train][["lesion_id", "label"]].drop_duplicates("lesion_id")
        fit_groups, calibration_groups = train_test_split(unique, test_size=0.2,
            stratify=unique.label, random_state=seed + 100 + index)
        fit = np.flatnonzero(train_frame.lesion_id.isin(fit_groups.lesion_id))
        calibration = np.flatnonzero(train_frame.lesion_id.isin(calibration_groups.lesion_id))
        assert not (set(groups[fit]) & set(groups[calibration]))
        assert not ((set(groups[fit]) | set(groups[calibration])) & set(groups[outer_holdout]))
        for rows in [fit, calibration, outer_holdout, outer_train]:
            if set(labels[rows]) != set(range(7)):
                raise ValueError("A grouped fold lost a lesion class")
        result.append({"fit": fit, "calibration": calibration, "holdout": outer_holdout, "outer_train": outer_train})
    coverage = np.zeros(len(train_frame), dtype=int)
    for fold in result:
        coverage[fold["holdout"]] += 1
    assert (coverage == 1).all()
    return result


def train_specialists(features, manifest, cache_info, output_dir, epochs=45, trees=400):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv("outputs/feature_models/efficientnet_b0_improved/split_manifest.csv")
    pd.testing.assert_frame_equal(manifest.reset_index(drop=True), baseline.reset_index(drop=True))
    expected = {"train": 7014, "validation": 1509, "test": 1492, "official_validation": 193, "official_test": 1512}
    if manifest.split.value_counts().to_dict() != expected:
        raise ValueError("All original images and split sizes must be retained")
    if not manifest[manifest.source_partition == "training"].groupby("lesion_id").split.nunique().eq(1).all():
        raise ValueError("Lesion leakage")
    train_rows = np.flatnonzero(manifest.split == "train")
    train_frame = manifest.iloc[train_rows].reset_index(drop=True)
    train_x = np.asarray(features[train_rows, 1280:2560], dtype=np.float32)
    train_y = train_frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    folds = build_folds(train_frame)
    grid = candidate_grid(epochs, trees)
    plan = {"protocol_version": 1, "fingerprint": cache_info["fingerprint"], "seed": 42,
        "feature_views": "full_d4_center", "training_images": len(train_rows),
        "feature_representation": "1280-dimensional averaged full-image D4 features",
        "class_names": CLASS_NAMES, "candidates_per_lesion": grid, "outer_folds": 3,
        "candidate_protocol": "Fit on 80% of outer-training lesions; calibrate on remaining 20%; score on wholly held-out outer fold.",
        "candidate_selection": "Mean outer-fold average precision; pooled OOF F1 breaks ties. Calibration C chosen by OOF log loss.",
        "calibration_C": [0.1, 1.0, 10.0],
        "final_refit": "Refit base head on all 7014 training images; sigmoid calibrator fitted to base-head OOF logits from training-only lesion folds.",
        "thresholds": "Maximum F1 on training OOF calibrated scores, never validation or test labels.",
        "smoothing": "MLP only: independently select 0, 0.03 or 0.10 per lesion; targets shrink towards fit-fold prevalence. Linear/tree heads use regularization instead.",
        "fusion_strengths": [0.0, 0.25, 0.5, 0.75, 1.0],
        "fusion": "p(class)/OOF_binary_F1_threshold**strength, normalize rows; strength chosen by existing internal validation macro F1.",
        "test_policy": "Finalize all seven heads, calibrators, thresholds and fusion before test evaluation.",
        "limitations": "Candidate OOF scores are selection diagnostics, not unbiased final performance estimates. Historical tests were previously reported."}
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["protocol_sha256"] = digest
    existing = output / "experiment.json"
    if existing.exists() and json.loads(existing.read_text())["protocol_sha256"] != digest:
        raise ValueError("Different specialist protocol already exists here. Choose a new output directory.")
    existing.write_text(json.dumps(plan, indent=2))
    manifest.to_csv(output / "split_manifest.csv", index=False)
    train_frame.assign(outer_fold=-1).to_csv(output / "training_folds.csv", index=False)
    fold_frame = train_frame.copy()
    for k, fold in enumerate(folds):
        fold_frame.loc[fold["holdout"], "outer_fold"] = k
    fold_frame.to_csv(output / "training_folds.csv", index=False)
    (output / "fold_audit.json").write_text(json.dumps([{key: train_frame.iloc[value].image_id.tolist()
        for key, value in fold.items()} for fold in folds], indent=2))
    # No globally fitted preprocessing is used in folds. Each candidate owns a
    # scaler fitted on exactly that fold's base-model fitting images.
    prepared = []
    for fold in folds:
        scaler = StandardScaler().fit(train_x[fold["fit"]])
        prepared.append({key: scaler.transform(train_x[fold[key]]).astype(np.float32)
                         for key in ["fit", "calibration", "holdout"]})
    specialists, selections = [], []
    for lesion_index, name in enumerate(CLASS_NAMES):
        folder = output / "lesion_models" / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            if json.loads((folder / "complete.json").read_text())["protocol_sha256"] != digest:
                raise ValueError("Specialist checkpoint protocol mismatch")
            specialists.append(joblib.load(folder / "model.joblib"))
            selections.append(json.loads((folder / "selection.json").read_text()))
            print(f"Resuming completed specialist: {name}", flush=True)
            continue
        start = time.monotonic()
        binary_y = (train_y == lesion_index).astype(int)
        trials, best = [], None
        for candidate_index, config in enumerate(grid):
            oof = {strength: np.empty(len(train_y)) for strength in plan["calibration_C"]}
            fold_aps = []
            for k, (fold, values) in enumerate(zip(folds, prepared)):
                model = fit_binary(values["fit"], binary_y[fold["fit"]], config, 42 + lesion_index * 10 + k)
                cal_scores = binary_logits(model, values["calibration"])
                hold_scores = binary_logits(model, values["holdout"])
                fold_aps.append(float(average_precision_score(binary_y[fold["holdout"]], expit(hold_scores))))
                for strength in plan["calibration_C"]:
                    calibrator = fit_calibrator(cal_scores, binary_y[fold["calibration"]], strength)
                    oof[strength][fold["holdout"]] = calibrator.predict_proba(hold_scores.reshape(-1, 1))[:, 1]
                del model
            losses = {strength: float(log_loss(binary_y, np.column_stack([1 - probability, probability]), labels=[0, 1]))
                      for strength, probability in oof.items()}
            calibration_strength = min(losses, key=losses.get)
            probability = oof[calibration_strength]
            threshold = f1_threshold(binary_y, probability)
            diagnostics = binary_metrics(binary_y, probability, threshold)
            record = {"candidate": candidate_index, "config": config, "calibration_C": calibration_strength,
                      "fold_average_precision": fold_aps, "mean_fold_average_precision": float(np.mean(fold_aps)),
                      "calibration_losses": losses, "oof_selection_metrics": diagnostics}
            trials.append(record)
            (folder / "trials.json").write_text(json.dumps(trials, indent=2))
            key = (record["mean_fold_average_precision"], diagnostics["f1"], -diagnostics["log_loss"])
            if best is None or key > best[0]:
                best = key, record, probability.copy()
            print(f"{name} candidate {candidate_index+1}/{len(grid)} {config}: CV AP={key[0]:.4f}, OOF F1={diagnostics['f1']:.4f}", flush=True)
        _, chosen, _ = best
        config = chosen["config"]
        # Obtain independent base-model logits for calibration of a full-data
        # refit, matching the standard ensemble=False CV calibration pattern.
        oof_logits = np.empty(len(train_y))
        for k, fold in enumerate(folds):
            scaler = StandardScaler().fit(train_x[fold["outer_train"]])
            model = fit_binary(scaler.transform(train_x[fold["outer_train"]]).astype(np.float32),
                               binary_y[fold["outer_train"]], config, 42 + lesion_index * 10 + k)
            oof_logits[fold["holdout"]] = binary_logits(model, scaler.transform(train_x[fold["holdout"]]).astype(np.float32))
        calibrator = fit_calibrator(oof_logits, binary_y, chosen["calibration_C"])
        oof_probability = calibrator.predict_proba(oof_logits.reshape(-1, 1))[:, 1]
        threshold = f1_threshold(binary_y, oof_probability)
        scaler = StandardScaler().fit(train_x)
        model = fit_binary(scaler.transform(train_x).astype(np.float32), binary_y, config, 42 + lesion_index * 10)
        head = BinarySpecialist(name, model, scaler, calibrator, threshold, config)
        joblib.dump(head, folder / "model.joblib")
        selection = {"lesion": name, **chosen, "final_threshold": threshold,
            "training_seconds": time.monotonic() - start,
            "calibrator_coefficient": float(calibrator.coef_[0, 0]), "calibrator_intercept": float(calibrator.intercept_[0]),
            "training_images": len(train_y), "positive_training_images": int(binary_y.sum()),
            "positive_training_lesions": int(train_frame[binary_y == 1].lesion_id.nunique())}
        (folder / "selection.json").write_text(json.dumps(selection, indent=2))
        pd.DataFrame({"image_id": train_frame.image_id, "label": binary_y,
                      "base_oof_logit": oof_logits, "calibrated_oof_score": oof_probability}).to_csv(folder / "training_oof_predictions.csv", index=False)
        (folder / "complete.json").write_text(json.dumps({"protocol_sha256": digest}, indent=2))
        specialists.append(joblib.load(folder / "model.joblib"))
        selections.append(selection)
        print(f"SELECTED {name}: {config}; calibration C={chosen['calibration_C']}; threshold={threshold:.4f}", flush=True)
    # Validation chooses only five predeclared global fusion strengths.
    val_rows = np.flatnonzero(manifest.split == "validation")
    val_y = manifest.iloc[val_rows].label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    system = SpecialistSystem(specialists)
    fusion_trials = []
    for strength in plan["fusion_strengths"]:
        system.strength = strength
        metrics = score_predictions(val_y, system.predict_proba(np.asarray(features[val_rows])))
        fusion_trials.append({"strength": strength, "macro_f1": metrics["macro_f1"],
                              "macro_f05": metrics["macro_f05"], "balanced_accuracy": metrics["balanced_accuracy"]})
    selected = max(fusion_trials, key=lambda row: (row["macro_f1"], row["balanced_accuracy"], -row["strength"]))
    system.strength = selected["strength"]
    folder = output / "specialists"
    folder.mkdir(exist_ok=True)
    joblib.dump(FunctionTransformer(validate=False), output / "scaler.joblib")
    joblib.dump(system, folder / "model.joblib")
    metadata = {"format_version": 1, "artifact_type": "tuned_ensemble", "kind": "specialists",
        "feature_dimension": 2560, "feature_views": "full_d4_center", "class_names": CLASS_NAMES,
        "backbone": cache_info["backbone"], "weights": cache_info["weights"], "fingerprint": cache_info["fingerprint"],
        "training_variant": "calibrated_specialists", "protocol_sha256": digest,
        "details": {"specialists": selections, "fusion_trials": fusion_trials, "selected_fusion": selected,
                    "score_semantics": "Normalized class-selection scores, not calibrated multiclass posterior probabilities"}}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "selection_complete.json").write_text(json.dumps({"selected_before_test_evaluation": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "selected_fusion": selected,
        "model_sha256": hashlib.sha256((folder / "model.joblib").read_bytes()).hexdigest()}, indent=2))
    system = joblib.load(folder / "model.joblib")
    combined_metrics, binary_rows, final_class_rows = {}, [], []
    for split in ["validation", "test", "official_validation", "official_test"]:
        indices = np.flatnonzero(manifest.split == split)
        labels = manifest.iloc[indices].label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
        raw = np.asarray(features[indices])
        binary_probabilities = system.binary_probabilities(raw)
        probabilities = system.predict_proba(raw)
        metrics = score_predictions(labels, probabilities)
        for average in ["macro", "weighted"]:
            metrics[f"{average}_precision"] = metrics["per_class"][f"{average} avg"]["precision"]
            metrics[f"{average}_recall"] = metrics["per_class"][f"{average} avg"]["recall"]
        combined_metrics[split] = metrics
        records = pd.DataFrame(probabilities, columns=CLASS_NAMES)
        records.insert(0, "true_label", manifest.iloc[indices].label.to_numpy())
        records.insert(0, "image_id", manifest.iloc[indices].image_id.to_numpy())
        records.to_csv(folder / f"{split}_predictions.csv", index=False)
        binary_records = pd.DataFrame(binary_probabilities, columns=CLASS_NAMES)
        binary_records.insert(0, "true_label", manifest.iloc[indices].label.to_numpy())
        binary_records.insert(0, "image_id", manifest.iloc[indices].image_id.to_numpy())
        binary_records.to_csv(folder / f"{split}_binary_probabilities.csv", index=False)
        for k, head in enumerate(specialists):
            result = binary_metrics((labels == k).astype(int), binary_probabilities[:, k], head.threshold)
            binary_rows.append({"split": split, "lesion": head.name, "family": head.config["family"],
                "label_smoothing": head.config["smoothing"], "calibration_C": selections[k]["calibration_C"], **result})
            matrix = np.array(metrics["confusion_matrix"])
            tp, positives, predicted = matrix[k, k], matrix[k].sum(), matrix[:, k].sum()
            final_class_rows.append({"split": split, "lesion": head.name, **metrics["per_class"][head.name],
                "one_vs_rest_accuracy": float((len(labels) - positives - predicted + 2 * tp) / len(labels))})
    (folder / "metrics.json").write_text(json.dumps(combined_metrics, indent=2))
    pd.DataFrame(binary_rows).to_csv(output / "binary_specialist_metrics.csv", index=False)
    pd.DataFrame(final_class_rows).to_csv(output / "combined_per_class_metrics.csv", index=False)
    row = {"model": "specialists"}
    for split, metrics in combined_metrics.items():
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_f05", "weighted_f1", "weighted_f05", "macro_precision", "macro_recall"]:
            row[f"{split}_{metric}"] = metrics[metric]
    pd.DataFrame([row]).to_csv(output / "comparison.csv", index=False)
    (output / "best_model.json").write_text(json.dumps({"model": "specialists", "selected_using": "training folds + internal validation only"}, indent=2))
    print(json.dumps(row, indent=2), flush=True)
    return row
