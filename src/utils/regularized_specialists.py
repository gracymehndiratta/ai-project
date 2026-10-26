"""Controlled SMOTE ablation with grouped early stopping and frozen test policy."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import xgboost as xgb

from src.data.full_isic import CLASS_NAMES
from src.models.lesion_specialists import BinaryMLP, BinarySpecialist, SpecialistSystem, binary_logits
from src.utils.specialist_training import (binary_metrics, binary_weights, build_folds,
    fit_calibrator, f1_threshold)
from src.utils.feature_training import score_predictions


def resample_fit(values, labels, enabled, seed):
    """Bound interpolation to 3x the minority, never force full balance."""
    counts = np.bincount(labels, minlength=2)
    minority = int(counts.argmin())
    target = max(int(counts[minority]), min(int(counts[minority] * 3),
                 int(np.ceil(counts.max() * 0.5))))
    if enabled and target > counts[minority]:
        sampler = SMOTE(sampling_strategy={minority: target},
                        k_neighbors=min(3, int(counts.min()) - 1), random_state=seed)
        values, labels = sampler.fit_resample(values, labels)
    return values.astype(np.float32), labels


class PatientStopping(xgb.callback.TrainingCallback):
    """Allow at least 200 rounds; remember the best round from the beginning."""
    def __init__(self):
        self.best, self.best_round = float('inf'), 0

    def after_iteration(self, model, epoch, evals_log):
        metric = next(iter(evals_log['validation_0'].values()))[-1]
        if metric < self.best - 1e-6:
            self.best, self.best_round = metric, epoch
        model.set_attr(best_iteration=str(self.best_round), best_score=str(self.best))
        return epoch + 1 >= 200 and epoch - self.best_round >= 100


def negative_ap(labels, probabilities):
    return -float(average_precision_score(labels, probabilities))


def fit_head(values, labels, config, smote, seed, stop=None, iterations=None):
    original_count = len(labels)
    prevalence = float(labels.mean())
    values, labels = resample_fit(values, labels, smote, seed)
    weights = binary_weights(labels, 0.5)
    details = {'original_images': original_count, 'resampled_images': len(labels),
               'class_counts_after': np.bincount(labels, minlength=2).tolist()}
    if config['family'] == 'linear':
        model = LogisticRegression(C=config['C'], class_weight=dict(enumerate(weights)),
                                   max_iter=1500, random_state=seed)
        with threadpool_limits(limits=4):
            model.fit(values, labels)
        return model, {**details, 'best_iterations': int(model.n_iter_[0]), 'curve': []}
    if config['family'] == 'xgboost':
        model = xgb.XGBClassifier(n_estimators=iterations or 1200, max_depth=config['depth'],
            learning_rate=.04, min_child_weight=5, subsample=.85, colsample_bytree=.8,
            reg_lambda=15, reg_alpha=.75, max_bin=128, tree_method='hist',
            objective='binary:logistic', eval_metric=negative_ap, n_jobs=4, random_state=seed,
            callbacks=[PatientStopping()] if stop is not None else None,
            device='cuda' if torch.cuda.is_available() else 'cpu')
        model.fit(values, labels, sample_weight=weights[labels],
                  eval_set=[stop] if stop is not None else None, verbose=False)
        curve = model.evals_result()['validation_0']['negative_ap'] if stop is not None else []
        best = int(model.best_iteration + 1) if stop is not None else int(iterations)
        if stop is not None:
            model._Booster = model.get_booster()[:best]
        model.set_params(device='cpu', callbacks=None)
        return model, {**details, 'best_iterations': best, 'curve': curve}
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BinaryMLP(values.shape[1]).to(device)
    x = torch.as_tensor(values, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights[labels], dtype=torch.float32, device=device)
    target = y * .97 + .03 * prevalence
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=.02)
    generator = torch.Generator(device=device).manual_seed(seed)
    best, best_epoch, state, curve = -1., 0, None, []
    for epoch in range(iterations or 180):
        model.train()
        for indices in torch.randperm(len(y), device=device, generator=generator).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[indices]).squeeze(1),
                    target[indices], reduction='none')
            (loss * w[indices]).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if stop is not None:
            model.eval()
            with torch.inference_mode():
                score = model(torch.as_tensor(stop[0], device=device)).squeeze(1).cpu().numpy()
            ap = float(average_precision_score(stop[1], score))
            curve.append(ap)
            if ap > best + 1e-6:
                best, best_epoch = ap, epoch
                state = copy.deepcopy(model.state_dict())
            if epoch + 1 >= 40 and epoch - best_epoch >= 30:
                break
    if stop is not None:
        model.load_state_dict(state)
    model.cpu().eval()
    return model, {**details, 'best_iterations': best_epoch + 1 if stop is not None else int(iterations),
                   'curve': curve}


def grid():
    return [{'family': 'linear', 'C': c, 'weight_power': .5, 'smoothing': 0.}
            for c in [.0003, .001]] + [
        {'family': 'xgboost', 'depth': depth, 'weight_power': .5, 'smoothing': 0.}
        for depth in [2, 3]] + [
        {'family': 'mlp', 'weight_decay': .02, 'weight_power': .5, 'smoothing': .03}]


def train_variant(features, manifest, cache_info, output, smote):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = {'version': 1, 'seed': 42, 'smote': smote, 'grid': grid(),
        'fingerprint': cache_info['fingerprint'], 'folds': 3,
        'smote_policy': 'Fit fold only, after scaling; at most 3x minority and 50% majority; k=3; recompute sqrt weights after resampling',
        'stopping': 'Lesion-separated inner 20% stop set: XGB max1200/min200/patience100; NN max180/min40/patience30. Restore best AP checkpoint.',
        'refit': 'For each outer fold choose iterations on inner fit/stop, refit outer train at that budget. Final all7014 refit at median budget.',
        'selection': 'Training OOF positive F1 primary; within .01 of best prefer smaller train-OOF F1 gap, then AP. No tests used.',
        'calibration': 'Sigmoid C=1 on training OOF logits; threshold max training OOF F1. Selection diagnostics are optimistic.',
        'fusion_strengths': [0., .25, .5, .75, 1.],
        'fusion': 'Five strengths selected by existing validation macro F1',
        'limitation': 'Previously viewed tests are reused for comparison, not a fresh blind benchmark; no patient-level grouping available.'}
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan['protocol_sha256'] = digest
    if (output / 'experiment.json').exists():
        assert json.loads((output / 'experiment.json').read_text())['protocol_sha256'] == digest
    (output / 'experiment.json').write_text(json.dumps(plan, indent=2))
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    rows = np.flatnonzero(manifest.split == 'train')
    frame = manifest.iloc[rows].reset_index(drop=True)
    x = np.asarray(features[rows, 1280:2560], dtype=np.float32)
    y = frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    folds = build_folds(frame)
    (output / 'fold_audit.json').write_text(json.dumps([
        {key: frame.iloc[value].image_id.tolist() for key, value in fold.items()} for fold in folds], indent=2))
    heads, selections = [], []
    for index, name in enumerate(CLASS_NAMES):
        folder = output / 'lesion_models' / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'complete.json').exists():
            assert json.loads((folder / 'complete.json').read_text())['protocol_sha256'] == digest
            heads.append(joblib.load(folder / 'model.joblib'))
            selections.append(json.loads((folder / 'selection.json').read_text()))
            continue
        binary_y = (y == index).astype(int)
        trials, candidates = [], []
        for candidate_index, config in enumerate(grid()):
            oof = np.empty(len(y))
            fitting_logits, fitting_labels, budgets, diagnostics = [], [], [], []
            for k, fold in enumerate(folds):
                seed = 42 + index * 10 + k
                fit_rows, stop_rows, outer = fold['fit'], fold['calibration'], fold['outer_train']
                scaler = StandardScaler().fit(x[fit_rows])
                if config['family'] != 'linear':
                    _, stopping = fit_head(scaler.transform(x[fit_rows]).astype(np.float32), binary_y[fit_rows],
                        config, smote, seed, stop=(scaler.transform(x[stop_rows]).astype(np.float32), binary_y[stop_rows]))
                    budget = stopping['best_iterations']
                else:
                    stopping, budget = {}, None
                scaler = StandardScaler().fit(x[outer])
                model, refit = fit_head(scaler.transform(x[outer]).astype(np.float32), binary_y[outer],
                                        config, smote, seed, iterations=budget)
                oof[fold['holdout']] = binary_logits(model, scaler.transform(x[fold['holdout']]).astype(np.float32))
                fitting_logits.append(binary_logits(model, scaler.transform(x[outer]).astype(np.float32)))
                fitting_labels.append(binary_y[outer])
                budgets.append(refit['best_iterations'])
                diagnostics.append({'fold': k, 'stopping': stopping, 'refit': refit})
            calibrator = fit_calibrator(oof, binary_y, 1.)
            probability = calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]
            threshold = f1_threshold(binary_y, probability)
            cv = binary_metrics(binary_y, probability, threshold)
            fit_probability = calibrator.predict_proba(np.concatenate(fitting_logits).reshape(-1, 1))[:, 1]
            fitting = binary_metrics(np.concatenate(fitting_labels), fit_probability, threshold)
            record = {'candidate': candidate_index, 'config': config, 'oof': cv, 'fit': fitting,
                'gap': fitting['f1'] - cv['f1'], 'budgets': budgets, 'folds': diagnostics}
            trials.append(record)
            candidates.append((record, oof.copy(), calibrator, threshold))
            (folder / 'trials.json').write_text(json.dumps(trials, indent=2))
            print(f"{output.name} {name} {config}: OOF F1={cv['f1']:.4f}, fit={fitting['f1']:.4f}, budgets={budgets}", flush=True)
        best_f1 = max(item[0]['oof']['f1'] for item in candidates)
        eligible = [item for item in candidates if item[0]['oof']['f1'] >= best_f1 - .01]
        chosen, logits, calibrator, threshold = min(eligible,
            key=lambda item: (max(0., item[0]['gap']), -item[0]['oof']['average_precision']))
        budget = int(np.median(chosen['budgets']))
        scaler = StandardScaler().fit(x)
        model, refit = fit_head(scaler.transform(x).astype(np.float32), binary_y,
                               chosen['config'], smote, 42 + index * 10, iterations=budget)
        head = BinarySpecialist(name, model, scaler, calibrator, threshold, chosen['config'])
        joblib.dump(head, folder / 'model.joblib')
        selection = {'lesion': name, **chosen, 'final_budget': budget, 'final_refit': refit,
                     'final_threshold': threshold, 'calibration_C': 1.}
        (folder / 'selection.json').write_text(json.dumps(selection, indent=2))
        pd.DataFrame({'image_id': frame.image_id, 'label': binary_y, 'base_oof_logit': logits,
            'calibrated_oof_score': calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]}).to_csv(
                folder / 'training_oof_predictions.csv', index=False)
        (folder / 'complete.json').write_text(json.dumps({'protocol_sha256': digest}))
        heads.append(head)
        selections.append(selection)
        print(f"SELECTED {name}: {chosen['config']}, budget={budget}", flush=True)
    system = SpecialistSystem(heads)
    val_rows = np.flatnonzero(manifest.split == 'validation')
    val_y = manifest.iloc[val_rows].label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    fusion_trials = []
    for strength in plan['fusion_strengths']:
        system.strength = strength
        metric = score_predictions(val_y, system.predict_proba(np.asarray(features[val_rows])))
        fusion_trials.append({'strength': strength, 'macro_f1': metric['macro_f1']})
    selected = max(fusion_trials, key=lambda row: row['macro_f1'])
    system.strength = selected['strength']
    folder = output / 'specialists'
    folder.mkdir(exist_ok=True)
    joblib.dump(FunctionTransformer(validate=False), output / 'scaler.joblib')
    joblib.dump(system, folder / 'model.joblib')
    metadata = {'format_version': 1, 'artifact_type': 'tuned_ensemble', 'kind': 'specialists',
        'feature_dimension': 2560, 'feature_views': 'full_d4_center', 'class_names': CLASS_NAMES,
        'backbone': cache_info['backbone'], 'weights': cache_info['weights'],
        'fingerprint': cache_info['fingerprint'], 'training_variant': 'specialists_smote' if smote else 'specialists_regularized',
        'details': {'selected_fusion': selected, 'fusion_trials': fusion_trials}}
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'selection_complete.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_sha256': hashlib.sha256((folder / 'model.joblib').read_bytes()).hexdigest(),
        'selected_before_test_evaluation': True, 'selected_fusion': selected}, indent=2))
    evaluate(system, features, manifest, output)
    (output / 'best_model.json').write_text(json.dumps({'model': 'specialists', 'selected_using': 'Training OOF + validation only'}))


def evaluate(system, features, manifest, output):
    metrics, binary_rows, class_rows = {}, [], []
    for split in ['train', 'validation', 'test', 'official_validation', 'official_test']:
        rows = np.flatnonzero(manifest.split == split)
        labels = manifest.iloc[rows].label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
        raw = np.asarray(features[rows])
        binary = system.binary_probabilities(raw)
        probabilities = system.predict_proba(raw)
        metric = score_predictions(labels, probabilities)
        for average in ['macro', 'weighted']:
            for key in ['precision', 'recall']:
                metric[f'{average}_{key}'] = metric['per_class'][f'{average} avg'][key]
        metrics[split] = metric
        for suffix, values in [('predictions', probabilities), ('binary_probabilities', binary)]:
            records = pd.DataFrame(values, columns=CLASS_NAMES)
            records.insert(0, 'true_label', manifest.iloc[rows].label.to_numpy())
            records.insert(0, 'image_id', manifest.iloc[rows].image_id.to_numpy())
            records.to_csv(output / 'specialists' / f'{split}_{suffix}.csv', index=False)
        for k, head in enumerate(system.specialists):
            binary_rows.append({'split': split, 'lesion': head.name, 'family': head.config['family'],
                'label_smoothing': head.config['smoothing'],
                **binary_metrics((labels == k).astype(int), binary[:, k], head.threshold)})
            class_rows.append({'split': split, 'lesion': head.name, **metric['per_class'][head.name]})
    (output / 'specialists' / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(binary_rows).to_csv(output / 'binary_specialist_metrics.csv', index=False)
    pd.DataFrame(class_rows).to_csv(output / 'combined_per_class_metrics.csv', index=False)
    row = {'model': 'specialists'}
    for split, metric in metrics.items():
        for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_f05', 'weighted_f1',
                    'weighted_f05', 'macro_precision', 'macro_recall']:
            row[f'{split}_{key}'] = metric[key]
    pd.DataFrame([row]).to_csv(output / 'comparison.csv', index=False)
    print(json.dumps(row, indent=2), flush=True)
