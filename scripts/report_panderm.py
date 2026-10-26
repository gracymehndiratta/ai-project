"""Verify saved PanDerm classifiers and compare with every historical run."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def markdown(frame):
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def main():
    os.chdir(ROOT)
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import ImageFeaturePredictor, load_classifier
    from src.models.panderm import sha256_file
    from src.utils.feature_training import score_predictions
    torch.set_num_threads(4)
    output = Path('outputs/feature_models/panderm_base_comparison')
    receipt = json.loads(Path('outputs/diagnostics/panderm_embeddings/extraction_receipt.json').read_text())
    frame = pd.read_csv(output / 'split_manifest.csv')
    pd.testing.assert_frame_equal(frame, pd.read_csv(receipt['manifest_path']))
    features = np.load(receipt['embeddings_path'], mmap_mode='r')
    labels = frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    train_rows = np.flatnonzero(frame.split == 'train')
    assert frame[frame.source_partition == 'training'].groupby('lesion_id').split.nunique().eq(1).all()
    marker_path = output / 'selection_complete.json'
    marker = json.loads(marker_path.read_text())
    records, per_class, models = [], [], {}
    for kind in ['nearest', 'linear', 'mlp', 'xgboost']:
        folder = output / kind
        model, identity, info = load_classifier(folder)
        models[kind] = model
        assert info['backbone'] == 'panderm_base' and info['fingerprint'] == receipt['fingerprint']
        assert marker['models'][kind] == sha256_file(folder / 'model.joblib')
        metrics = json.loads((folder / 'metrics.json').read_text())
        if kind == 'nearest':
            assert len(model.features) == len(train_rows) == 7014
            np.testing.assert_array_equal(model.labels, labels[train_rows])
            np.testing.assert_array_equal(model.groups, frame.iloc[train_rows].lesion_id.to_numpy())
            np.testing.assert_allclose(model.features, model.normalize(features[train_rows]))
        else:
            np.testing.assert_allclose(model.projection.mean_, features[train_rows].mean(0, dtype=np.float64), atol=1e-10)
        selection = json.loads((folder / 'selection.json').read_text())
        if kind == 'mlp':
            assert all(trial['epochs_observed'] >= 40 for trial in selection['trials'])
        if kind == 'xgboost':
            assert model.models[0].get_booster().num_boosted_rounds() == selection['selected']['best_round']
        for split in ['train', 'validation', 'test', 'official_validation', 'official_test']:
            rows = np.flatnonzero(frame.split == split)
            saved = pd.read_csv(folder / f'{split}_predictions.csv')
            assert saved.image_id.tolist() == frame.iloc[rows].image_id.tolist()
            assert saved.true_label.tolist() == frame.iloc[rows].label.tolist()
            if split in ['test', 'official_test']:
                assert marker_path.stat().st_mtime_ns <= (folder / f'{split}_predictions.csv').stat().st_mtime_ns
            values = np.asarray(features[rows])
            predicted = (model.predict_proba(values, excluded_groups=frame.iloc[rows].lesion_id.to_numpy())
                if kind == 'nearest' and split == 'train' else model.predict_proba(values))
            np.testing.assert_allclose(predicted, saved[CLASS_NAMES].to_numpy(), atol=1e-7)
            measured = score_predictions(labels[rows], predicted)
            for metric in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_f05', 'weighted_f1', 'weighted_f05']:
                np.testing.assert_allclose(measured[metric], metrics[split][metric], atol=1e-12)
            row = {'run': 'PanDerm', 'model': kind, 'split': split,
                **{k: measured[k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_f05', 'weighted_f1', 'weighted_f05']},
                'macro_precision': measured['per_class']['macro avg']['precision'],
                'macro_recall': measured['per_class']['macro avg']['recall']}
            records.append(row)
            for name in CLASS_NAMES:
                per_class.append({'model': kind, 'split': split, 'lesion': name, **measured['per_class'][name]})
    current = pd.DataFrame(records)
    historical = pd.read_csv('outputs/diagnostics/regularized_specialists/all_models_comparison.csv')
    history = pd.concat([historical, current], ignore_index=True)
    history.to_csv(output / 'historical_comparison.csv', index=False)
    pd.DataFrame(per_class).to_csv(output / 'per_class_statistics.csv', index=False)
    recommendation = history[history.split == 'validation'].sort_values(['macro_f1', 'balanced_accuracy'], ascending=False).iloc[0]
    (output / 'recommendation.json').write_text(json.dumps({'run': recommendation.run,
        'model': recommendation.model, 'validation_macro_f1': recommendation.macro_f1,
        'selected_using': 'Internal validation only; all earlier experiments retained'}, indent=2))
    hashes = json.loads((output / 'previous_artifact_hashes.json').read_text())
    assert all(sha256_file(path) == digest for path, digest in hashes.items())
    # Verify the actual image path using exactly the stored checkpoint/transform.
    probes = frame[frame.split == 'test'].groupby('label', sort=True).head(1)
    image_checks = []
    predictor = ImageFeaturePredictor(output / 'linear', 'cuda' if torch.cuda.is_available() else 'cpu')
    assert all(not p.requires_grad for p in predictor.extractor.parameters())
    for index, row in probes.iterrows():
        with Image.open(row.image_path) as image:
            tensor = predictor.transform(image.convert('RGB')).unsqueeze(0).to(predictor.device)
        with torch.inference_mode():
            embedding = predictor.extractor(tensor).cpu().numpy()
        np.testing.assert_allclose(embedding[0], features[index], atol=1e-4, rtol=1e-4)
        for kind, model in models.items():
            actual = model.predict_proba(embedding)
            expected = model.predict_proba(np.asarray(features[index:index+1]))
            np.testing.assert_allclose(actual, expected, atol=.02, rtol=0)
            assert actual.argmax(1)[0] == expected.argmax(1)[0]
            image_checks.append({'image_id': row.image_id, 'model': kind,
                'label': CLASS_NAMES[actual.argmax(1)[0]], 'max_probability_delta': float(np.abs(actual-expected).max())})
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'scripts/app.py'), default_timeout=90).run()
    assert not app.exception
    options = list(app.sidebar.selectbox[0].options)
    panderm_options = [option for option in options if option.startswith('panderm_base /')]
    assert len(panderm_options) == 4 and len(options) == len(set(options))
    for option in panderm_options:
        app.sidebar.selectbox[0].select(option).run()
        assert not app.exception
    (output / 'verification.json').write_text(json.dumps({'passed': True,
        'previous_artifacts_unchanged': len(hashes), 'scalers_train_only': True,
        'nearest_references_train_only': True, 'nearest_train_scores_lesion_excluded': True,
        'all_predictions_and_metrics_recomputed': True, 'selection_before_test': True,
        'real_image_checks': image_checks, 'app_options': panderm_options}, indent=2))
    # Paired lesion bootstrap compares the validation-selected PanDerm head with
    # the previous validation-selected EfficientNet linear model on internal test.
    chosen = json.loads((output / 'best_model.json').read_text())['model']
    baseline = Path('outputs/feature_models/efficientnet_b0_improved/linear/test_predictions.csv')
    ours = pd.read_csv(output / chosen / 'test_predictions.csv')
    other = pd.read_csv(baseline)
    assert ours.image_id.tolist() == other.image_id.tolist()
    test_frame = frame[frame.split == 'test']
    _, groups = np.unique(test_frame.lesion_id, return_inverse=True)
    truth = test_frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    matrices = []
    for saved in [other, ours]:
        matrix = np.zeros((groups.max()+1, 7, 7), dtype=np.int64)
        np.add.at(matrix, (groups, truth, saved[CLASS_NAMES].to_numpy().argmax(1)), 1)
        matrices.append(matrix)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        sample = rng.integers(0, len(matrices[0]), len(matrices[0]))
        fs = []
        for matrix in matrices:
            cm = matrix[sample].sum(0)
            fs.append(float(np.mean(2 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1))))
        differences.append(fs[1] - fs[0])
    interval = np.quantile(differences, [.025, .975]).tolist()
    (output / 'paired_bootstrap.json').write_text(json.dumps({'panDerm_head': chosen,
        'internal_test_macro_f1_difference_CI95': interval, 'resampling_unit': 'lesion',
        'replicates': 2000, 'limitation': 'Fixed-model comparison, excluding training/selection uncertainty'}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for kind in ['nearest', 'linear', 'mlp', 'xgboost']:
        values = current[current.model == kind].set_index('split')
        axes[0].plot(['Train*', 'Validation', 'Internal test', 'Official test'],
            values.loc[['train', 'validation', 'test', 'official_test'], 'macro_f1'], marker='o', label=kind)
    axes[0].set(ylabel='Macro F1', ylim=(0, 1), title='Frozen PanDerm heads: generalization')
    axes[0].legend()
    official = current[current.split == 'official_test'].set_index('model')
    positions = np.arange(4)
    for i, metric in enumerate(['macro_f1', 'macro_f05', 'accuracy']):
        axes[1].bar(positions + (i-1)*.25, official.loc[['nearest','linear','mlp','xgboost'], metric], .25, label=metric)
    axes[1].set_xticks(positions, ['Nearest', 'Linear', 'Small NN', 'XGBoost'])
    axes[1].set(ylim=(0, 1), title='Official test: same 1,512 images')
    axes[1].legend()
    fig.text(.01, .005, '*Nearest baseline training score excludes every image of the query lesion; other training scores are resubstitution.', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    lines = ['# PanDerm backbone comparison', '',
        '**Bare PanDerm cannot classify seven lesions by itself:** the verified official checkpoint contains 186 encoder tensors and no classification head or text encoder. It produces 768-dimensional embeddings. No accuracy/F1 is assigned to the bare encoder.', '',
        'For a fourth measurable baseline, fixed cosine 1-nearest-neighbor retrieval uses only original training embeddings/labels and learns no parameters. It is not the bare model or zero-shot classification. Its votes are not calibrated confidence estimates.', '',
        'All four methods use the same frozen PanDerm Base features and existing splits. The backbone was never fine-tuned. The 7,014 fitting images are used for supervised training; all 11,720 images remain assigned to their original partitions. No SMOTE, metadata fusion or ensembles of multiple heads are used.', '',
        'Linear: validation-selected L2 multinomial logistic regression. Small NN: 768→256→64→7, 214,407 parameters, dropout0.4/0.25, AdamW weight decay0.02, label smoothing0.03; up to180 epochs, observe at least40, patience30, best checkpoint restored. XGBoost: depth2/3, lambda15/alpha0.75, up to1200 rounds, observe at least200, patience100, best validation macro-F1 checkpoint restored. Class weighting and scaling use training data only.', '',
        '## Official-test results', '', markdown(current[current.split == 'official_test'][['model', 'accuracy', 'macro_f1', 'macro_f05', 'macro_precision', 'macro_recall', 'balanced_accuracy']]), '',
        '## F1 across splits', '', markdown(current[['model', 'split', 'macro_f1', 'macro_f05', 'accuracy']]), '',
        'Nearest training scores use leave-one-lesion-out retrieval; reporting exact-self retrieval would trivially give 100% and mislead. Training scores for supervised heads are evaluated on original fitting images.', '',
        '## Per-class official-test statistics', '', markdown(pd.DataFrame(per_class).query("split == 'official_test'")[['model', 'lesion', 'precision', 'recall', 'f1-score', 'f0.5-score', 'support']]), '',
        '## Historical comparison on official test', '', markdown(history[history.split == 'official_test'][['run', 'model', 'accuracy', 'macro_f1', 'macro_f05']]), '',
        f'Validation-selected PanDerm method: **{chosen}**. Overall validation-selected recommendation: **{recommendation.run}/{recommendation.model}**, validation macro F1={recommendation.macro_f1:.4f}.', '',
        f'Paired internal-test lesion bootstrap, PanDerm winner minus previous EfficientNet linear: 95% interval [{interval[0]:.4f}, {interval[1]:.4f}]. This excludes training/selection uncertainty.', '',
        '## Verification and limitations', '',
        f'Passed: train-only scaling/retrieval references, all saved metrics and predictions independently recomputed, best tree count restored, all heads frozen before test reporting, {len(image_checks)} actual image/model checks, all four app choices load, and {len(hashes)} historical artifacts unchanged.', '',
        'The original tests have been inspected repeatedly during earlier experiments, so this is a comparative benchmark rather than a fresh blind evaluation. Repeated validation selection can overfit. Lesion separation is established within the training collection; patient-level and cross-release independence are not established. No independent audit of PanDerm pretraining overlap with these images has been performed. These results do not establish clinical usefulness.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(markdown(current[current.split == 'official_test'][['model', 'accuracy', 'macro_f1', 'macro_f05', 'macro_precision', 'macro_recall']]), flush=True)
    print('Verification passed.', flush=True)


if __name__ == '__main__':
    main()
