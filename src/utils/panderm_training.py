"""A fixed retrieval baseline and three heads on frozen PanDerm embeddings."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import xgboost as xgb

from src.data.full_isic import CLASS_NAMES
from src.models.feature_models import build_mlp, classifier_probabilities, load_classifier
from src.models.panderm import sha256_file
from src.models.panderm_baseline import PanDermNearestNeighbor
from src.utils.feature_training import score_predictions
from src.utils.regularized_specialists import PatientStopping
from src.utils.tuned_training import TunedClassifier, class_weights, macro_f1_cost


def key(metric):
    return metric['macro_f1'], metric['balanced_accuracy']


def fit_network(train_x, train_y, val_x, val_y, power, seed=42):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_mlp(train_x.shape[1]).to(device)
    x = torch.as_tensor(train_x, device=device)
    y = torch.as_tensor(train_y, device=device)
    val = torch.as_tensor(val_x, device=device)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights(train_y, power),
        dtype=torch.float32, device=device), label_smoothing=.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=.02)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=.5)
    generator = torch.Generator(device=device).manual_seed(seed)
    best, state, best_epoch, history = (-1., -1.), None, 0, []
    for epoch in range(1, 181):
        model.train()
        total = 0.
        for rows in torch.randperm(len(y), device=device, generator=generator).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[rows]), y[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        model.eval()
        with torch.inference_mode():
            probabilities = model(val).softmax(1).cpu().numpy()
            fit_probabilities = model(x).softmax(1).cpu().numpy()
        metric = score_predictions(val_y, probabilities)
        fitting = -macro_f1_cost(train_y, fit_probabilities)
        history.append({'epoch': epoch, 'train_loss': total / len(y),
            'train_macro_f1': fitting, 'validation_macro_f1': metric['macro_f1']})
        schedule.step(metric['macro_f1'])
        if key(metric) > best:
            best, best_epoch = key(metric), epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f'NN weights^{power} epoch {epoch}: train F1={fitting:.4f}, validation F1={metric["macro_f1"]:.4f}', flush=True)
        if epoch >= 40 and epoch - best_epoch >= 30:
            break
    model.cpu().load_state_dict(state)
    model.eval()
    return model, {'weight_power': power, 'best_epoch': best_epoch, 'epochs_observed': len(history),
        'history': history, 'architecture': [train_x.shape[1], 256, 64, 7],
        'parameters': sum(p.numel() for p in model.parameters()), 'validation_macro_f1': best[0]}


def train(output='outputs/feature_models/panderm_base_comparison', large=False):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if large:
        cache = Path('data/features/panderm_large_72dfc47de5e585b2.npy')
        receipt = json.loads(cache.with_suffix('.json').read_text())
        receipt.update(embeddings_path=str(cache), checkpoint_path='artifacts/panderm_paper/panderm_large.pth',
            manifest_path=str(cache.with_name(cache.stem+'_manifest.csv')), trainable_parameters=0)
    else:
        receipt = json.loads(Path('outputs/diagnostics/panderm_embeddings/extraction_receipt.json').read_text())
    dimension = 1024 if large else 768
    backbone = 'panderm_large' if large else 'panderm_base'
    assert receipt['complete'] and receipt['trainable_parameters'] == 0
    assert sha256_file(receipt['embeddings_path']) == receipt['embeddings_sha256']
    assert sha256_file(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
    frame = pd.read_csv(receipt['manifest_path'])
    pd.testing.assert_frame_equal(frame, pd.read_csv('outputs/feature_models/efficientnet_b0_improved/split_manifest.csv'))
    features = np.load(receipt['embeddings_path'], mmap_mode='r')
    assert features.shape == (11720, dimension) and np.isfinite(features).all()
    if large:
        paper = pd.read_csv('artifacts/panderm_paper/metadata/HAM10000_clean/ISIC2018_splits/HAM_clean.csv')
        mapping = paper.set_index('image_id').split.replace({'val': 'validation'})
        frame['project_split'] = frame.split
        internal = frame.image_id.isin(mapping.index)
        frame.loc[internal, 'split'] = frame.loc[internal, 'image_id'].map(mapping)
        assert frame.groupby('lesion_id').split.nunique().max() <= 2
        used = frame[frame.split.isin(['train', 'validation', 'test'])]
        assert used.groupby('lesion_id').split.nunique().eq(1).all()
        assert frame.split.value_counts()[['train','validation','test']].tolist() == [8207,575,1232]
    labels = frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    train_rows, val_rows = np.flatnonzero(frame.split == 'train'), np.flatnonzero(frame.split == 'validation')
    checkpoint = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=True)
    head_keys = [k for k in checkpoint if k.startswith('head.')]
    assert not head_keys
    plain = {'status': 'not_applicable', 'reason': 'Official checkpoint is a frozen vision encoder, with no learned classification head or text encoder. Output is 768 features, not seven lesion probabilities. A separately labelled 1-nearest-neighbor baseline is supplied, not claimed to be the bare model.',
        'checkpoint_tensors': len(checkpoint), 'classifier_keys': head_keys, 'embedding_shape': list(features.shape),
        'classification_metrics': None}
    if not large:
        (output / 'plain_panderm.json').write_text(json.dumps(plain, indent=2))
    del checkpoint
    plan = {'version': 2, 'seed': 42, 'fingerprint': receipt['fingerprint'],
        'backbone': backbone, 'checkpoint_sha256': receipt['checkpoint_sha256'],
        'heads': ['paper', 'linear', 'mlp', 'xgboost'] if large else ['nearest', 'linear', 'mlp', 'xgboost'], 'training_images': len(train_rows),
        'nearest': {'neighbors': 1, 'metric': 'cosine', 'preprocessing': 'Per-vector L2 normalization, no fitted scaling',
            'fit': 'Store only 7014 original training embeddings and their labels; no optimization or hyperparameter selection',
            'train_metric': 'Leave-one-lesion-out retrieval; all images of the queried lesion are excluded',
            'scores': 'Hard neighbor votes; not calibrated probabilities'},
        'linear': {'C': [.0003, .001, .003, .01, .1], 'weight_power': [.5, 1.]},
        'mlp': {'architecture': [dimension, 256, 64, 7], 'weight_power': [.5, 1.], 'weight_decay': .02,
            'dropout': [.4, .25], 'smoothing': .03, 'max_epochs': 180, 'min_observed_epochs': 40, 'patience': 30},
        'xgboost': {'depth': [2, 3], 'weight_power': [.5, 1.], 'max_rounds': 1200,
            'min_observed_rounds': 200, 'patience': 100, 'learning_rate': .04, 'lambda': 15., 'alpha': .75},
        'selection': 'Existing internal validation macro F1; balanced accuracy breaks ties. Restore best checkpoints. Freeze all heads before tests.',
        'resampling': 'None; train-only class weighting, no SMOTE',
        'representation': f'Canonical frozen {dimension}-dimensional cache; train-only StandardScaler for custom heads, unchanged raw features for author probe',
        'limits': 'Previously inspected test sets are comparative benchmarks, not fresh blind validation. No patient-level or cross-release independence claim.'}
    if large:
        plan.pop('nearest')
        plan['xgboost']['stopping_metric'] = 'macro_f1_cost; default logloss metric disabled'
        plan['split_protocol'] = 'Exact released HAM_clean.csv train8207/val575/test1232; unknown trainisic row excluded. External official sets unchanged.'
        plan['paper'] = {'recipe': 'Unchanged author logistic_regression.py, raw embeddings, C=71.68, LBFGS max_iter1000, seed0',
            'source': 'outputs/diagnostics/panderm_paper_reference/model.joblib',
            'model_sha256': sha256_file('outputs/diagnostics/panderm_paper_reference/model.joblib'),
            'split_csv_sha256': sha256_file('artifacts/panderm_paper/metadata/HAM10000_clean/ISIC2018_splits/HAM_clean.csv')}
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan['protocol_sha256'] = digest
    if (output / 'experiment.json').exists():
        assert json.loads((output / 'experiment.json').read_text())['protocol_sha256'] == digest
    (output / 'experiment.json').write_text(json.dumps(plan, indent=2))
    hashes_path = output / 'previous_artifact_hashes.json'
    if not hashes_path.exists():
        paths = [p for p in output.parent.rglob('*') if p.is_file() and output not in p.parents]
        hashes_path.write_text(json.dumps({str(p): sha256_file(p) for p in paths}, indent=2))
    frame.to_csv(output / 'split_manifest.csv', index=False)
    scaler = StandardScaler().fit(features[train_rows])
    scaled = scaler.transform(features).astype(np.float32)
    # Each serialized model owns its fitted preprocessing; parent scaler is identity.
    joblib.dump(FunctionTransformer(validate=False), output / 'scaler.joblib')
    x, y, val_x, val_y = scaled[train_rows], labels[train_rows], scaled[val_rows], labels[val_rows]
    all_selections = {}
    for kind in plan['heads']:
        folder = output / kind
        folder.mkdir(exist_ok=True)
        if (folder / 'selection.json').exists():
            all_selections[kind] = json.loads((folder / 'selection.json').read_text())
            continue
        trials, best, chosen, winner = [], (-1., -1.), None, None
        if kind == 'paper':
            reference = Path('outputs/diagnostics/panderm_paper_reference')
            protocol = json.loads((reference / 'protocol.json').read_text())
            assert protocol['checkpoint_sha256'] == receipt['checkpoint_sha256']
            assert protocol['released_csv_sha256'] == plan['paper']['split_csv_sha256']
            winner = joblib.load(reference / 'model.joblib')
            metric = score_predictions(val_y, winner.predict_proba(np.asarray(features[val_rows])))
            expected = json.loads((reference / 'metrics.json').read_text())['released_val']
            assert abs(metric['macro_f1'] - expected['macro_f1']) < 1e-12
            chosen, best = plan['paper'], key(metric)
            trials = [{'config': chosen, 'validation_macro_f1': best[0]}]
        elif kind == 'nearest':
            winner = PanDermNearestNeighbor(np.asarray(features[train_rows]), y,
                frame.iloc[train_rows].lesion_id.to_numpy())
            metric = score_predictions(val_y, winner.predict_proba(np.asarray(features[val_rows])))
            chosen = plan['nearest']
            best = key(metric)
            trials = [{'config': chosen, 'validation_macro_f1': best[0]}]
        elif kind == 'linear':
            for power in plan['linear']['weight_power']:
                weights = class_weights(y, power)
                for c in plan['linear']['C']:
                    model = LogisticRegression(C=c, class_weight=dict(enumerate(weights)), max_iter=2000, random_state=42)
                    with threadpool_limits(limits=4):
                        model.fit(x, y)
                    metric = score_predictions(val_y, model.predict_proba(val_x))
                    trial = {'C': c, 'weight_power': power, 'iterations': int(model.n_iter_[0]),
                             'validation_macro_f1': metric['macro_f1'], 'validation_balanced_accuracy': metric['balanced_accuracy']}
                    trials.append(trial)
                    if key(metric) > best:
                        best, chosen, winner = key(metric), trial, model
                    print(f'Linear C={c}, weights^{power}: validation F1={metric["macro_f1"]:.4f}', flush=True)
        elif kind == 'mlp':
            for power in plan['mlp']['weight_power']:
                model, trial = fit_network(x, y, val_x, val_y, power)
                metric = score_predictions(val_y, classifier_probabilities(model, val_x, 'mlp'))
                trials.append(trial)
                if key(metric) > best:
                    best, chosen, winner = key(metric), trial, model
        else:
            for power in plan['xgboost']['weight_power']:
                weights = class_weights(y, power)
                for depth in plan['xgboost']['depth']:
                    model = xgb.XGBClassifier(n_estimators=1200, max_depth=depth, learning_rate=.04,
                        min_child_weight=5, reg_lambda=15., reg_alpha=.75, subsample=.85, colsample_bytree=.8,
                        max_bin=128, tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
                        objective='multi:softprob', num_class=7, eval_metric=macro_f1_cost,
                        callbacks=[PatientStopping()], n_jobs=4, random_state=42)
                    if large:
                        model.set_params(disable_default_eval_metric=True)
                    model.fit(x, y, sample_weight=weights[y], eval_set=[(val_x, val_y)], verbose=False)
                    best_round = int(model.best_iteration + 1)
                    history = model.evals_result()
                    model._Booster = model.get_booster()[:best_round]
                    model.set_params(device='cpu', callbacks=None)
                    metric = score_predictions(val_y, model.predict_proba(val_x))
                    trial = {'depth': depth, 'weight_power': power, 'best_round': best_round,
                        'history': history, 'validation_macro_f1': metric['macro_f1']}
                    trials.append(trial)
                    if key(metric) > best:
                        best, chosen, winner = key(metric), trial, model
                    print(f'XGB depth={depth}, weights^{power}: best round={best_round}, validation F1={metric["macro_f1"]:.4f}', flush=True)
        wrapped = winner if kind in ['nearest', 'paper'] else TunedClassifier([winner], scaler)
        joblib.dump(wrapped, folder / 'model.joblib')
        metadata = {'format_version': 1, 'artifact_type': 'tuned_ensemble', 'kind': kind,
            'backbone': backbone, 'weights': ('PanDerm_Large:' if large else 'PanDerm_Base:') + receipt['checkpoint_sha256'],
            'feature_dimension': dimension, 'feature_views': 'canonical', 'class_names': CLASS_NAMES,
            'fingerprint': receipt['fingerprint'], 'training_variant': 'panderm_frozen_heads',
            'protocol_sha256': digest, 'details': chosen}
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        selection = {'selected': chosen, 'trials': trials, 'validation_macro_f1': best[0],
            'model_sha256': sha256_file(folder / 'model.joblib')}
        (folder / 'selection.json').write_text(json.dumps(selection, indent=2))
        all_selections[kind] = selection
        print(f'SELECTED {kind}: validation F1={best[0]:.4f}', flush=True)
    best_kind = max(all_selections, key=lambda k: all_selections[k]['validation_macro_f1'])
    (output / 'best_model.json').write_text(json.dumps({'model': best_kind, 'selected_using': 'internal validation macro F1 only'}))
    (output / 'selection_complete.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
        'models': {k: v['model_sha256'] for k, v in all_selections.items()}, 'selected_before_test_evaluation': True}, indent=2))
    summary, classes = [], []
    for kind in plan['heads']:
        folder = output / kind
        model, _, _ = load_classifier(folder)
        metrics = {}
        row = {'model': kind}
        for split in ['train', 'validation', 'test', 'official_validation', 'official_test']:
            rows = np.flatnonzero(frame.split == split)
            probability = (model.predict_proba(np.asarray(features[rows]), excluded_groups=frame.iloc[rows].lesion_id.to_numpy())
                if kind == 'nearest' and split == 'train' else model.predict_proba(np.asarray(features[rows])))
            metric = score_predictions(labels[rows], probability)
            for average in ['macro', 'weighted']:
                for name in ['precision', 'recall']:
                    metric[f'{average}_{name}'] = metric['per_class'][f'{average} avg'][name]
            metrics[split] = metric
            records = pd.DataFrame(probability, columns=CLASS_NAMES)
            records.insert(0, 'true_label', frame.iloc[rows].label.to_numpy())
            records.insert(0, 'image_id', frame.iloc[rows].image_id.to_numpy())
            records.to_csv(folder / f'{split}_predictions.csv', index=False)
            for name in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_f05', 'weighted_f1',
                         'weighted_f05', 'macro_precision', 'macro_recall']:
                row[f'{split}_{name}'] = metric[name]
            for name in CLASS_NAMES:
                classes.append({'model': kind, 'split': split, 'lesion': name, **metric['per_class'][name]})
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        summary.append(row)
        print(f'FINAL {kind}: train F1={metrics["train"]["macro_f1"]:.4f}, validation={metrics["validation"]["macro_f1"]:.4f}, official test={metrics["official_test"]["macro_f1"]:.4f}', flush=True)
    pd.DataFrame(summary).to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(classes).to_csv(output / 'per_class_statistics.csv', index=False)
    previous = json.loads(hashes_path.read_text())
    assert all(sha256_file(p) == digest for p, digest in previous.items())
    print(f'Preserved {len(previous)} previous artifacts unchanged.', flush=True)
