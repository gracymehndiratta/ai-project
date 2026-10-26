"""Bounded validation-only search on frozen, multi-view image representations.

All preprocessing is fitted on real training images. No synthetic examples,
test-label selection, or changes to the historical splits are permitted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb

from src.data.full_isic import CLASS_NAMES
from src.models.feature_models import build_mlp, load_classifier, classifier_probabilities
from src.utils.feature_training import score_predictions


class Columns:
    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def transform(self, features):
        return np.asarray(features[:, self.start:self.stop], dtype=np.float32)


class TunedClassifier:
    """Serializable head ensemble; preprocessing and score adjustment travel with it."""
    def __init__(self, models, projection, factors=None):
        self.models = models
        self.projection = projection
        self.factors = np.ones(7) if factors is None else np.asarray(factors)

    def predict_proba(self, features):
        values = np.asarray(self.projection.transform(features), dtype=np.float32)
        predictions = []
        for model in self.models:
            if isinstance(model, nn.Module):
                model.eval()
                with torch.inference_mode():
                    predictions.append(model(torch.from_numpy(values)).softmax(1).numpy())
            else:
                predictions.append(model.predict_proba(values))
        probabilities = np.mean(predictions, axis=0).astype(np.float64) * self.factors
        return probabilities / probabilities.sum(1, keepdims=True)


def macro_f1_cost(labels, probabilities):
    # XGBoost sklearn callable metrics are minimized. Use negative macro F1.
    return -float(f1_score(labels, probabilities.reshape(-1, 7).argmax(1),
                           labels=np.arange(7), average="macro", zero_division=0))


def class_weights(labels, power):
    weights = (len(labels) / (7 * np.bincount(labels, minlength=7))) ** power
    # Mean sample weight stays at one so regularization remains comparable.
    return weights / weights[np.asarray(labels)].mean()


def train_network(train_x, train_y, val_x, val_y, power, decay, seed, epochs=120):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_mlp(train_x.shape[1]).to(device)
    x, y = torch.as_tensor(train_x, device=device), torch.as_tensor(train_y, device=device)
    validation = torch.as_tensor(val_x, device=device)
    weights = torch.as_tensor(class_weights(train_y, power), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    generator = torch.Generator(device=device).manual_seed(seed)
    best, best_score, stale, history = None, -1, 0, []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(y), device=device, generator=generator)
        total = 0
        for indices in permutation.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            probability = model(validation).softmax(1).cpu().numpy()
        score = -macro_f1_cost(val_y, probability)
        history.append({"epoch": epoch + 1, "train_loss": total / len(y), "validation_macro_f1": score})
        scheduler.step(score)
        if score > best_score:
            best, best_score, stale = deepcopy(model).cpu(), score, 0
        else:
            stale += 1
        if stale >= 18:
            break
    return best, history


def train_tuned(features, manifest, cache_info, output_dir, epochs=120, trees=1000):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists():
        old = json.loads((output / "experiment.json").read_text())
        if old["fingerprint"] != cache_info["fingerprint"]:
            raise ValueError("Refusing to mix different feature caches")
    manifest.to_csv(output / "split_manifest.csv", index=False)
    train = np.flatnonzero(manifest.split == "train")
    val = np.flatnonzero(manifest.split == "validation")
    labels = manifest.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    if len(features) != len(manifest) or not np.isfinite(features).all():
        raise ValueError("Invalid features")
    groups = manifest[manifest.source_partition == "training"].groupby("lesion_id").split.nunique()
    if not groups.eq(1).all():
        raise ValueError("Lesion leakage")
    scaler = StandardScaler().fit(features[train])
    joblib.dump(scaler, output / "scaler.joblib")
    scaled = scaler.transform(features).astype(np.float32)
    train_x, train_y = scaled[train], labels[train]
    val_x, val_y = scaled[val], labels[val]
    projections = {"canonical": Columns(0, 1280), "pooled": Columns(1280, 2560), "combined": Columns(0, 2560)}
    with threadpool_limits(limits=4):
        projections["pca256"] = make_pipeline(PCA(n_components=256, svd_solver="randomized", random_state=42),
                                               StandardScaler()).fit(train_x)
    arrays = {name: (np.asarray(project.transform(train_x), dtype=np.float32),
                     np.asarray(project.transform(val_x), dtype=np.float32)) for name, project in projections.items()}
    plan = {**cache_info, "seed": 42, "training_variant": "tuned_multiview", "class_names": CLASS_NAMES,
            "selection_metric": "validation macro F1", "split_images": manifest.split.value_counts().to_dict(),
            "linear_grid": {"representations": list(projections), "C": [0.001, 0.01, 0.1], "weight_power": [0, 0.5, 1]},
            "mlp_grid": {"representations": ["pooled", "combined", "pca256"], "weight_power": [0, 0.5, 1], "weight_decay": [0.01]},
            "xgboost_grid": {"representations": ["pooled", "combined", "pca256"], "weight_power": [0.5, 1], "max_depth": [3, 5]},
            "decision_prior_powers": [-0.2, 0, 0.2],
            "ensemble": "MLP/XGBoost only: equal probability average of the three strongest validation candidates, retained only if better than the single head. Linear remains one logistic regression.",
            "test_policy": "Do not evaluate any test set until all three heads and their choices are finalized.",
            "limitations": "Repeated use of this validation split can overfit; historical test sets have already been reported. No pristine prospective evaluation claim."}
    (output / "experiment.json").write_text(json.dumps(plan, indent=2))
    selected = {}
    trials = []
    frequencies = np.bincount(train_y, minlength=7) / len(train_y)

    def consider(kind, model, representation, config, details=None):
        wrapper = TunedClassifier([model], projections[representation])
        raw = wrapper.predict_proba(val_x)
        choices = []
        for power in plan["decision_prior_powers"]:
            factors = frequencies ** power
            probability = raw * factors
            probability /= probability.sum(1, keepdims=True)
            score = -macro_f1_cost(val_y, probability)
            choices.append((score, -abs(power), power, factors))
        score, _, power, factors = max(choices, key=lambda choice: choice[:2])
        wrapper.factors = factors
        record = {"kind": kind, "representation": representation, **config,
                  "decision_prior_power": power, "validation_macro_f1": score, "details": details}
        trials.append(record)
        (output / "trials.json").write_text(json.dumps(trials, indent=2))
        print(json.dumps({key: value for key, value in record.items() if key != "details"}), flush=True)
        candidates[kind].append((score, wrapper, record))
        candidates[kind].sort(key=lambda item: item[0], reverse=True)
        candidates[kind] = candidates[kind][:3]

    candidates = {name: [] for name in ("linear", "mlp", "xgboost")}
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for representation, (x, v) in arrays.items():
            for power in [0, 0.5, 1]:
                for strength in [0.001, 0.01, 0.1]:
                    model = LogisticRegression(C=strength, max_iter=1200, solver="lbfgs",
                                               class_weight=dict(enumerate(class_weights(train_y, power))), random_state=42)
                    model.fit(x, train_y)
                    consider("linear", model, representation, {"C": strength, "weight_power": power})
    timings = {"linear": time.monotonic() - start}
    start = time.monotonic()
    for representation in ["pooled", "combined", "pca256"]:
        x, v = arrays[representation]
        for power in [0, 0.5, 1]:
            model, history = train_network(x, train_y, v, val_y, power, 0.01, 42, epochs)
            consider("mlp", model, representation, {"weight_power": power, "weight_decay": 0.01, "seed": 42}, history)
    # Two independent seeds for the strongest MLP configuration reduce run variance.
    winner = candidates["mlp"][0][2]
    representation = winner["representation"]
    x, v = arrays[representation]
    for seed in [43, 44]:
        model, history = train_network(x, train_y, v, val_y, winner["weight_power"], 0.01, seed, epochs)
        consider("mlp", model, representation, {"weight_power": winner["weight_power"], "weight_decay": 0.01, "seed": seed}, history)
    timings["mlp"] = time.monotonic() - start
    start = time.monotonic()
    for representation in ["pooled", "combined", "pca256"]:
        x, v = arrays[representation]
        for power in [0.5, 1]:
            for depth in [3, 5]:
                model = xgb.XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=0.04,
                    min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=10, reg_alpha=0.5,
                    objective="multi:softprob", num_class=7, tree_method="hist", max_bin=128,
                    device="cuda" if torch.cuda.is_available() else "cpu", n_jobs=4,
                    eval_metric=macro_f1_cost, callbacks=[xgb.callback.EarlyStopping(rounds=100,
                        metric_name="macro_f1_cost", data_name="validation_0", maximize=False, min_delta=0.0005)], random_state=42)
                model.fit(x, train_y, sample_weight=class_weights(train_y, power)[train_y], eval_set=[(v, val_y)], verbose=False)
                model.set_params(device="cpu")
                consider("xgboost", model, representation, {"weight_power": power, "max_depth": depth,
                         "best_iteration": int(model.best_iteration)})
    timings["xgboost"] = time.monotonic() - start
    # Finalize every artifact before opening either test split.
    for kind, options in candidates.items():
        single_score, single, record = options[0]
        # Members may use different projections and score adjustments, so combine wrappers.
        ensemble = TunedClassifier([option[1] for option in options], Columns(0, 2560))
        ensemble_score = -macro_f1_cost(val_y, ensemble.predict_proba(val_x))
        use_ensemble = kind != "linear" and ensemble_score > single_score
        selected[kind] = ensemble if use_ensemble else single
        folder = output / kind
        folder.mkdir(exist_ok=True)
        joblib.dump(selected[kind], folder / "model.joblib")
        metadata = {"format_version": 1, "artifact_type": "tuned_ensemble", "kind": kind,
            "feature_dimension": int(features.shape[1]), "feature_views": "full_d4_center",
            "backbone": cache_info["backbone"], "weights": cache_info["weights"], "fingerprint": cache_info["fingerprint"],
            "class_names": CLASS_NAMES, "training_variant": "tuned_multiview", "seed": 42,
            "training_seconds": timings[kind], "details": {"best_single": record,
                "ensemble_used": use_ensemble, "ensemble_validation_macro_f1": ensemble_score,
                "members": [option[2] for option in options] if use_ensemble else [record]}}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "selection_complete.json").write_text(json.dumps({"selected_before_test_evaluation": True,
        "validation_scores": {kind: -macro_f1_cost(val_y, head.predict_proba(val_x)) for kind, head in selected.items()}}, indent=2))
    rows = []
    for kind in selected:
        folder = output / kind
        head, _, _ = load_classifier(folder)
        row = {"model": kind, "training_seconds": timings[kind]}
        metrics = {}
        for split in ["validation", "test", "official_validation", "official_test"]:
            indices = np.flatnonzero(manifest.split == split)
            probabilities = classifier_probabilities(head, scaled[indices], kind)
            metrics[split] = score_predictions(labels[indices], probabilities)
            predictions = pd.DataFrame(probabilities, columns=CLASS_NAMES)
            predictions.insert(0, "true_label", manifest.iloc[indices].label.to_numpy())
            predictions.insert(0, "image_id", manifest.iloc[indices].image_id.to_numpy())
            predictions.to_csv(folder / f"{split}_predictions.csv", index=False)
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_f05", "weighted_f1", "weighted_f05"]:
                row[f"{split}_{metric}"] = metrics[split][metric]
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / "comparison.csv", index=False)
    best = result.sort_values("validation_macro_f1", ascending=False).iloc[0].model
    (output / "best_model.json").write_text(json.dumps({"model": best, "selected_using": "validation only"}, indent=2))
    print(result.to_string(index=False), flush=True)
    return result
