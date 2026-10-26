"""Compare all preserved experiments and verify the regularized SMOTE ablation."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import os
    os.chdir(ROOT)
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import load_classifier, classifier_probabilities, ImageFeaturePredictor
    from src.utils.feature_training import score_predictions
    from src.utils.specialist_training import binary_metrics
    torch.set_num_threads(4)
    parent = Path('outputs/feature_models')
    out = Path('outputs/diagnostics/regularized_specialists')
    manifest = pd.read_csv(parent / 'efficientnet_b0_improved/split_manifest.csv')
    train = manifest[manifest.split == 'train']
    labels = train.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    multiview = np.load('data/features/efficientnet_b0_ccb90bc07d54e7c0.npy', mmap_mode='r')
    canonical = np.load('data/features/efficientnet_b0_2af1eff877bf4a7d.npy', mmap_mode='r')
    runs = [('Original', 'efficientnet_b0', ['linear', 'mlp', 'xgboost']),
            ('Previous SMOTE', 'efficientnet_b0_smote', ['linear', 'mlp', 'xgboost']),
            ('Improved multiclass', 'efficientnet_b0_improved', ['linear', 'mlp', 'xgboost']),
            ('Previous specialists', 'efficientnet_b0_specialists', ['specialists']),
            ('Regularized', 'efficientnet_b0_specialists_regularized', ['specialists']),
            ('Regularized + SMOTE', 'efficientnet_b0_specialists_smote', ['specialists'])]
    history, binaries, per_class, selection_rows = [], [], [], []
    for title, directory, kinds in runs:
        folder = parent / directory
        pd.testing.assert_frame_equal(manifest, pd.read_csv(folder / 'split_manifest.csv'))
        for kind in kinds:
            classifier, scaler, info = load_classifier(folder / kind)
            features = multiview if info.get('feature_views') == 'full_d4_center' else canonical
            x = scaler.transform(np.asarray(features[manifest.split == 'train'])).astype(np.float32)
            train_p = classifier_probabilities(classifier, x, kind)
            train_metrics = score_predictions(labels, train_p)
            metric_dict = {'train': train_metrics}
            if kind == 'specialists':
                raw = np.asarray(multiview[manifest.split == 'train'])
                train_binary = classifier.binary_probabilities(raw)
                for k, head in enumerate(classifier.specialists):
                    binaries.append({'run': title, 'split': 'train', 'lesion': head.name,
                        'family': head.config['family'], **binary_metrics((labels == k).astype(int), train_binary[:, k], head.threshold)})
                saved_binary = pd.read_csv(folder / 'binary_specialist_metrics.csv')
                for _, row in saved_binary[saved_binary.split != 'train'].iterrows():
                    binaries.append({'run': title, **row.to_dict()})
            for split in ['validation', 'test', 'official_validation', 'official_test']:
                predictions = pd.read_csv(folder / kind / f'{split}_predictions.csv')
                expected = manifest[manifest.split == split]
                assert predictions.image_id.tolist() == expected.image_id.tolist()
                assert predictions.true_label.tolist() == expected.label.tolist()
                metric_dict[split] = score_predictions(predictions.true_label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy(),
                                                      predictions[CLASS_NAMES].to_numpy())
                if kind == 'specialists':
                    independent = pd.read_csv(folder / kind / f'{split}_binary_probabilities.csv')
                    assert independent.image_id.tolist() == expected.image_id.tolist()
                    binary_truth = expected.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
                    for k, head in enumerate(classifier.specialists):
                        recomputed = binary_metrics((binary_truth == k).astype(int), independent[head.name].to_numpy(), head.threshold)
                        saved = saved_binary[(saved_binary.split == split) & (saved_binary.lesion == head.name)].iloc[0]
                        for metric in ['f1', 'f05', 'accuracy', 'precision', 'recall', 'balanced_accuracy']:
                            np.testing.assert_allclose(recomputed[metric], saved[metric], atol=1e-12)
            for split, metric in metric_dict.items():
                record = {'run': title, 'model': kind, 'split': split,
                    **{key: metric[key] for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_f05', 'weighted_f1', 'weighted_f05']},
                    'macro_precision': metric['per_class']['macro avg']['precision'],
                    'macro_recall': metric['per_class']['macro avg']['recall']}
                history.append(record)
                for name in CLASS_NAMES:
                    per_class.append({'run': title, 'model': kind, 'split': split, 'lesion': name,
                                      **metric['per_class'][name]})
            print(f'{title}/{kind}: train F1={train_metrics["macro_f1"]:.4f}, official F1={metric_dict["official_test"]["macro_f1"]:.4f}', flush=True)
    history = pd.DataFrame(history)
    binary = pd.DataFrame(binaries)
    history.to_csv(out / 'all_models_comparison.csv', index=False)
    binary.to_csv(out / 'all_specialist_statistics.csv', index=False)
    pd.DataFrame(per_class).to_csv(out / 'all_models_per_class.csv', index=False)
    gap = binary.pivot(index=['run', 'lesion'], columns='split', values='f1').reset_index()
    gap['train_minus_validation'] = gap.train - gap.validation
    gap['train_minus_internal_test'] = gap.train - gap.test
    gap['train_minus_official_test'] = gap.train - gap.official_test
    gap.to_csv(out / 'specialist_f1_gaps.csv', index=False)
    # Verify grouped fit/stop/evaluation boundaries and saved checkpoints.
    verified = []
    for title, directory, _ in runs[-2:]:
        folder = parent / directory
        model_path = folder / 'specialists/model.joblib'
        marker = json.loads((folder / 'selection_complete.json').read_text())
        assert marker['model_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest()
        for split in ['test', 'official_test']:
            assert (folder / 'selection_complete.json').stat().st_mtime_ns <= (folder / 'specialists' / f'{split}_predictions.csv').stat().st_mtime_ns
        audit = json.loads((folder / 'fold_audit.json').read_text())
        for fold in audit:
            groups = {key: set(train[train.image_id.isin(ids)].lesion_id) for key, ids in fold.items()}
            assert groups['fit'].isdisjoint(groups['calibration'])
            assert groups['holdout'].isdisjoint(groups['outer_train'])
            assert groups['fit'] | groups['calibration'] == groups['outer_train']
        for name in CLASS_NAMES:
            lesion_folder = folder / 'lesion_models' / name
            head = joblib.load(lesion_folder / 'model.joblib')
            np.testing.assert_allclose(head.scaler.mean_, np.asarray(multiview[manifest.split == 'train', 1280:2560]).mean(0, dtype=np.float64), atol=1e-8)
            selection = json.loads((lesion_folder / 'selection.json').read_text())
            oof = pd.read_csv(lesion_folder / 'training_oof_predictions.csv')
            assert oof.image_id.tolist() == train.image_id.tolist()
            assert oof.label.tolist() == (train.label == name).astype(int).tolist()
            if head.config['family'] == 'xgboost':
                assert head.model.get_booster().num_boosted_rounds() == selection['final_budget']
            for trial in json.loads((lesion_folder / 'trials.json').read_text()):
                for details in trial['folds']:
                    curve = details['stopping'].get('curve', [])
                    if curve:
                        assert len(curve) >= (200 if trial['config']['family'] == 'xgboost' else 40)
            selection_rows.append({'run': title, 'lesion': name, 'family': head.config['family'],
                'config': json.dumps(head.config), 'iterations': selection['final_budget'],
                'training_oof_f1': selection['oof']['f1'], 'added_smote_vectors':
                selection['final_refit']['resampled_images'] - selection['final_refit']['original_images']})
        verified.append(directory)
    hashes = json.loads((out / 'previous_artifact_hashes.json').read_text())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    image_checks = []
    from PIL import Image
    probes = manifest[manifest.split == 'test'].groupby('label', sort=True).head(1)
    for title, directory, _ in runs[-2:]:
        predictor = ImageFeaturePredictor(parent / directory / 'specialists', 'cuda' if torch.cuda.is_available() else 'cpu')
        cached = predictor.classifier.predict_proba(np.asarray(multiview[probes.index]))
        for k, (_, row) in enumerate(probes.iterrows()):
            with Image.open(row.image_path) as image:
                label, _, probabilities = predictor.predict(image)
            values = np.array([probabilities[name] for name in CLASS_NAMES])
            np.testing.assert_allclose(values, cached[k], atol=.02, rtol=0)
            assert label == CLASS_NAMES[cached[k].argmax()]
            image_checks.append({'run': title, 'image_id': row.image_id, 'prediction': label,
                                 'max_score_delta': float(np.abs(values - cached[k]).max())})
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'scripts/app.py'), default_timeout=60).run()
    assert not app.exception
    options = list(app.sidebar.selectbox[0].options)
    assert len(options) >= 12 and len(options) == len(set(options))
    for suffix in [' / Early stopping + weighting', ' / Early stopping + SMOTE + weighting']:
        option = next(option for option in options if option.endswith(suffix))
        app.sidebar.selectbox[0].select(option).run()
        assert not app.exception
    (out / 'verification.json').write_text(json.dumps({'passed': True, 'runs': verified,
        'previous_artifacts_unchanged': len(hashes), 'group_boundaries': True, 'scalers_train_only': True,
        'best_tree_budgets_restored': True, 'selection_markers_before_tests': True,
        'real_image_checks': image_checks, 'app_options': options}, indent=2))
    pd.DataFrame(selection_rows).to_csv(out / 'selected_specialists.csv', index=False)
    candidate = history[history.split == 'validation'].sort_values(['macro_f1', 'balanced_accuracy'], ascending=False).iloc[0]
    (out / 'recommendation.json').write_text(json.dumps({'run': candidate.run, 'model': candidate.model,
        'validation_macro_f1': candidate.macro_f1, 'selected_using': 'Internal validation only'}, indent=2))
    # Paired lesion bootstrap for the SMOTE effect on internal test, with fixed models.
    test_frame = manifest[manifest.split == 'test']
    truth = test_frame.label.map(dict(zip(CLASS_NAMES, range(7)))).to_numpy()
    _, groups = np.unique(test_frame.lesion_id, return_inverse=True)
    matrices = []
    for _, directory, _ in runs[-2:]:
        pred = pd.read_csv(parent / directory / 'specialists/test_predictions.csv')[CLASS_NAMES].to_numpy().argmax(1)
        cm = np.zeros((groups.max() + 1, 7, 7), dtype=np.int64)
        np.add.at(cm, (groups, truth, pred), 1)
        matrices.append(cm)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        sample = rng.integers(0, len(matrices[0]), len(matrices[0]))
        values = []
        for cm in matrices:
            total = cm[sample].sum(0)
            values.append(float(np.mean(2 * total.diagonal() / np.maximum(total.sum(0) + total.sum(1), 1))))
        differences.append(values[1] - values[0])
    ci = np.quantile(differences, [.025, .975]).tolist()
    (out / 'smote_paired_bootstrap.json').write_text(json.dumps({'internal_test_delta_macro_f1_CI95': ci,
        'resampling_unit': 'lesion', 'samples': 2000, 'limitation': 'Fixed models; does not include training/selection uncertainty'}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    combined = history[history.model == 'specialists']
    for _, group in combined.groupby('run', sort=False):
        name = group.run.iloc[0]
        values = group.set_index('split').loc[['train', 'validation', 'test', 'official_test'], 'macro_f1']
        axes[0].plot(['Train', 'Validation', 'Internal test', 'Official test'], values, marker='o', label=name)
    axes[0].set(ylabel='Macro F1', title='Seven-class decision performance', ylim=(0, 1))
    axes[0].legend(fontsize=8)
    colors = ['#8a8a8a', '#2979b8', '#e09124']
    for i, title in enumerate(['Previous specialists', 'Regularized', 'Regularized + SMOTE']):
        values = gap[gap.run == title].set_index('lesion').loc[CLASS_NAMES, 'train_minus_official_test']
        axes[1].bar(np.arange(7) + (i - 1) * .25, values, .25, label=title, color=colors[i])
    axes[1].set_xticks(np.arange(7), CLASS_NAMES)
    axes[1].set(ylabel='Training F1 minus official test F1', title='Independent specialist generalization gaps')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'overfitting_comparison.png', dpi=160)
    plt.close(fig)
    # Avoid a reporting-only dependency on tabulate.
    def table(frame):
        columns = frame.columns.tolist()
        rows = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        for _, row in frame.iterrows():
            rows.append('| ' + ' | '.join(f'{value:.4f}' if isinstance(value, float) else str(value) for value in row) + ' |')
        return '\n'.join(rows)
    lines = ['# Regularization and SMOTE experiment', '',
        'All previous artifacts are preserved byte for byte. All 11,720 images retain the original splits; fitting uses 7,014 images. No demographics were added.', '',
        '## Applied action plan', '',
        '- Keep lesion groups together; fit every scaler and SMOTE sampler only on the fitting partition.',
        '- Compare identical candidate grids and seeds with and without conservative SMOTE. Interpolate at most 3x the minority count, capped at half the majority count; recompute square-root class weights after resampling.',
        '- Tune iterations on a separate inner stopping partition: XGBoost up to 1,200 rounds, at least 200 observed, 100-round patience; neural networks up to 180 epochs, at least 40 observed, 30-epoch patience. Restore the best AP checkpoint, including an earlier checkpoint when appropriate.',
        '- Moderate tree constraints: depths 2/3, child weight 5, lambda15, alpha0.75, subsample0.85, column sample0.8. Neural network keeps 256/64 units and dropout0.4/0.25, weight decay0.02 and smoothing0.03. Linear C0.0003/0.001 candidates.',
        '- Refit outer-fold heads at the selected budget; obtain held-out training OOF logits. Choose heads by OOF positive F1; within 0.01 choose a smaller fitting-OOF gap. Refit the final head on all original fitting images at the median budget.',
        '- Fit unweighted sigmoid calibration and binary thresholds on training OOF scores. Select fusion strength only on internal validation. Freeze artifacts before reporting test performance.', '',
        '## All historical model results on the official test', '',
        table(history[history.split == 'official_test'][['run', 'model', 'accuracy', 'macro_f1', 'macro_f05', 'macro_precision', 'macro_recall']]), '',
        '## Combined system F1 across splits', '',
        table(combined[['run', 'split', 'macro_f1', 'macro_f05', 'accuracy']]), '',
        '## Each independent specialist', '',
        table(binary[binary.split == 'official_test'][['run', 'lesion', 'family', 'f1', 'f05', 'precision', 'recall', 'accuracy', 'balanced_accuracy']]), '',
        '## Specialist F1 gaps', '', table(gap), '',
        f'Validation-selected recommendation across every retained model: **{candidate.run}/{candidate.model}** (macro F1 {candidate.macro_f1:.4f}).', '',
        f'Paired lesion bootstrap: internal-test macro F1 SMOTE-minus-no-SMOTE 95% interval {ci[0]:.4f} to {ci[1]:.4f}. Fixed-model interval excludes training/selection uncertainty.', '',
        '## Interpretation and limits', '',
        'Training metrics are inference on original fitting images, never synthetic vectors. Training OOF scores are used for selection/calibration/thresholds and are optimistic selection diagnostics. Inner stopping AP is a ranking metric, so patient stopping can retain strong separation while training F1 remains high. Moderate regularization cannot guarantee a small gap with only 51 distinct training dermatofibroma lesions.', '',
        'A narrower training/test gap is not itself a win: held-out F1 must remain useful. Binary accuracy is dominated by negatives for rare classes. Independent specialist metrics differ from seven-class argmax metrics. Normalized fusion scores are decision scores, not calibrated joint probabilities.', '',
        'These test sets have been inspected in earlier experiments; results are repeated comparisons rather than a fresh blind benchmark. Lesion grouping covers the training collection; patient grouping and cross-release lesion independence are not established. No test-label-driven follow-up tuning was performed.', '',
        f'Verification passed; {len(hashes)} historical artifacts unchanged. See CSV files for all models, all classes, selected budgets, and train/held-out gaps.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(table(history[history.split == 'official_test'][['run', 'model', 'accuracy', 'macro_f1', 'macro_f05']]), flush=True)
    print(table(gap), flush=True)
    print('Verification passed.', flush=True)


if __name__ == '__main__':
    main()
