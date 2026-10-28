"""Safeguards use small fixtures, independent of historical local checkpoints."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.full_isic import CLASS_NAMES
from src.ui.catalog import benchmark_models, rank_models, saved_models
from src.utils.prediction_audit import audit_predictions
from src.utils.research_integrity import require_exploratory_opt_in


def model(name, score, stage='original', **info):
    return {'id': name, 'name': name, 'family_id': name, 'stage': stage,
            'info': info, 'neural': False,
            'metrics': {'official_test': {'macro_f1': score, 'images': 1512}}}


def test_test_scores_and_preferred_flags_cannot_change_selection():
    models = [model('B', .99), model('A', .01), model('C', .9, 'adapted')]
    assert [m['id'] for m in benchmark_models(models)] == ['A', 'B']
    changed = deepcopy(models)
    changed[0]['metrics']['official_test']['macro_f1'] = 0
    changed[1]['metrics']['official_test']['macro_f1'] = 1
    changed[0]['preferred_for_benchmark'] = True
    changed[0]['family_id'] = 'A'
    models[0]['family_id'] = 'A'
    assert [m['id'] for m in benchmark_models(changed)] == [m['id'] for m in benchmark_models(models)]
    assert [m['id'] for m in rank_models(models)] == ['A', 'B', 'C']


@pytest.mark.parametrize('info', [{'data_contaminated_for_test_evaluation': True},
                                {'excluded_from_clean_model_ranking': True},
                                {'training_variant': 'joint_refinement'}])
def test_default_excludes_contamination_even_if_stage_is_wrong(info):
    assert benchmark_models([model('bad', 1, **info)]) == []


def test_archive_requires_explicit_stage_and_is_not_score_ordered():
    models = [model('Z', 1, 'adapted'), model('A', 0, 'adapted')]
    assert benchmark_models(models) == []
    assert [m['id'] for m in benchmark_models(models, 'adapted')] == ['A', 'Z']


def test_catalog_without_artifacts_is_empty(tmp_path):
    assert saved_models(tmp_path) == []


def test_dashboard_without_checkpoints_renders_notice(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.ui import dashboard
    monkeypatch.setattr(dashboard, 'saved_models', lambda: [])
    monkeypatch.setattr(dashboard, 'federated_panel', lambda: None)
    def render():
        from src.ui.dashboard import main
        main()
    app = AppTest.from_function(render).run()
    assert not app.exception
    assert any('No eligible local model artifacts' in item.value for item in app.info)


@pytest.mark.parametrize('module', ['scripts.refine_joint_cohorts', 'scripts.adapt_six_models',
                                  'scripts.adapt_panderm_once'])
def test_historical_entrypoints_refuse_before_loading_or_writing(module, tmp_path, monkeypatch):
    import importlib
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='disabled'):
        importlib.import_module(module).main()
    assert list(tmp_path.iterdir()) == []


def test_opt_in_emits_explicit_limit():
    with pytest.warns(UserWarning, match='cannot establish'):
        require_exploratory_opt_in(True)


@pytest.fixture
def evidence(tmp_path):
    truth = pd.DataFrame({'image_id': [f'new-{i}' for i in range(7)], 'label': CLASS_NAMES,
                          'lesion_id': [f'new-lesion-{i}' for i in range(7)],
                          'image_sha256': [f'{i:064x}' for i in range(7)]})
    seen = pd.DataFrame({'image_id': ['old-1'], 'label': ['mel'], 'lesion_id': ['old-lesion'],
                         'image_sha256': ['f' * 64]})
    predictions = pd.DataFrame(np.eye(7), columns=CLASS_NAMES)
    predictions.insert(0, 'image_id', truth.image_id)
    paths = [tmp_path / n for n in ['predictions.csv', 'truth.csv', 'exposure.csv']]
    for frame, path in zip([predictions, truth, seen], paths):
        frame.to_csv(path, index=False)
    return paths


def test_recalculation_aligns_ids_and_hashes_inputs(evidence):
    predictions = pd.read_csv(evidence[0]).iloc[::-1]
    predictions.to_csv(evidence[0], index=False)
    report = audit_predictions(*evidence, require_disjoint=True)
    assert report['metrics']['accuracy'] == 1
    assert report['metrics']['macro_f1'] == 1
    assert report['declared_image_and_lesion_disjoint']
    assert not report['identity_evidence_complete']['patient_id']
    assert not report['independent_generalization_established']
    assert len(report['input_sha256']) == 3


@pytest.mark.parametrize('key', ['image_id', 'lesion_id', 'image_sha256', 'patient_id'])
def test_overlap_across_releases_rejected(evidence, key):
    cohort = pd.read_csv(evidence[1], dtype=str)
    exposure = pd.read_csv(evidence[2], dtype=str)
    if key == 'patient_id':
        cohort[key] = 'patient-1'
    exposure.loc[0, key] = cohort.loc[0, key]
    cohort.to_csv(evidence[1], index=False)
    exposure.to_csv(evidence[2], index=False)
    with pytest.raises(ValueError, match='Disjoint evaluation rejected'):
        audit_predictions(*evidence, require_disjoint=True)


def test_missing_identity_evidence_is_not_a_pass(evidence):
    cohort = pd.read_csv(evidence[1]).drop(columns='lesion_id')
    cohort.to_csv(evidence[1], index=False)
    assert not audit_predictions(*evidence)['declared_image_and_lesion_disjoint']
    with pytest.raises(ValueError, match='missing'):
        audit_predictions(*evidence, require_disjoint=True)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'labels', 'negative', 'nan', 'sum'])
def test_invalid_prediction_evidence_is_rejected(evidence, problem):
    table = pd.read_csv(evidence[0])
    if problem == 'missing': table = table.iloc[:-1]
    if problem == 'duplicate': table.loc[1, 'image_id'] = table.loc[0, 'image_id']
    if problem == 'labels': table['true_label'] = 'mel'
    if problem == 'negative': table.loc[0, CLASS_NAMES[:2]] = [-.1, 1.1]
    if problem == 'nan': table.loc[0, CLASS_NAMES[0]] = float('nan')
    if problem == 'sum': table.loc[0, CLASS_NAMES] = .5
    table.to_csv(evidence[0], index=False)
    with pytest.raises(ValueError):
        audit_predictions(*evidence)


def test_reported_metrics_must_match_predictions(evidence, tmp_path):
    receipt = tmp_path / 'reported.json'
    metrics = audit_predictions(*evidence)['metrics']
    receipt.write_text(json.dumps({'test': metrics}))
    audit_predictions(*evidence, reported_metrics=receipt, split='test')
    metrics['macro_f1'] = .9399
    receipt.write_text(json.dumps({'test': metrics}))
    with pytest.raises(ValueError, match='macro_f1'):
        audit_predictions(*evidence, reported_metrics=receipt, split='test')
