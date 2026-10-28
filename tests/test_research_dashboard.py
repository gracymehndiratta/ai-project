from io import BytesIO
from pathlib import Path
import pytest
from PIL import Image
from src.ui.results import all_models,top_six,research,CLASS_LABELS
from src.ui.uploads import decode_image,quality_issue


@pytest.mark.skipif(not Path('outputs/diagnostics/six_model_adaptation/active_models.json').exists(), reason='Historical local experiment artifacts required')
def test_saved_dropdown_catalog_includes_all_families_in_alphabetical_order():
    from src.ui.catalog import saved_models,benchmark_models
    models=saved_models()
    assert models == sorted(models,key=lambda m:(m['name'].casefold(),m['id']))
    assert len({m['id'] for m in models})==len(models)
    assert len({m['selection_label'] for m in models})==len(models)
    assert all(m['manifest_path'].is_file() for m in models)
    assert all('evaluation scores never used' in m['selection_basis'] for m in models)
    assert {'efficientnet_b0','panderm_large','panderm_base'} <= {m['info']['backbone'] for m in models}
    assert {'linear','mlp','xgboost','specialists','cnn'} <= {m['info']['kind'] for m in models}
    assert any(m['stage']=='adapted' for m in models)
    expected={p.parent.resolve() for p in Path('outputs/feature_models').glob('*/*/metadata.json')}
    assert expected <= {m['folder'].resolve() for m in models}
    for stage in ['original','adapted']:
        selected=benchmark_models(models,stage)
        assert len(selected)==6 and sum(m['neural'] for m in selected)<=2
        assert not any(m.get('reference') for m in selected)
        assert len({m['family_id'] for m in selected})==6
        assert selected==sorted(selected,key=lambda m:(m['name'].casefold(),m['id']))
        if stage=='adapted': assert all(m['stage']=='adapted' for m in selected)


@pytest.mark.skipif(not Path('outputs/diagnostics/six_model_adaptation/active_models.json').exists(), reason='Historical local experiment artifacts required')
def test_adapted_comparison_preserves_six_families_and_marks_exposure():
    from src.ui.results import adapted_models,local_research,exposure_label
    selected=adapted_models()
    assert len(selected)==6 and sum(m['neural'] for m in selected)==2
    assert {m['id'] for m in selected}=={'large_linear','large_nn128','large_nn192','large_xgb','base_linear','base_xgb'}
    assert all('Fitting data' in exposure_label(m,'official_test') for m in selected)
    assert all(m['metrics']['test']['images']==1232 for m in selected)
    assert all(m['folder']!=old['folder'] for m,old in zip(selected,adapted_models('original')))
    assert 'Original training overlap' in exposure_label(next(m for m in selected if m['id']=='base_linear'),'validation')
    refs=local_research()
    assert len(refs)==1 and not refs[0]['adapted_to_test_data']
    assert refs[0]['metrics']['official_test']['images']==1512


@pytest.mark.skipif(not Path('outputs/diagnostics/six_model_adaptation/active_models.json').exists(), reason='Historical local experiment artifacts required')
def test_top_six_matches_recorded_ranking_and_limits_neural_heads():
    selected=top_six(all_models())
    assert len(selected)==6
    assert sum(model['neural'] for model in selected)==2
    assert len({model['name'] for model in selected})==6
    assert selected==sorted(selected,key=lambda m:(m['name'].casefold(),m['id']))
    assert all(model['metrics']['official_test']['images']==1512 for model in selected)
    assert all(not model['info'].get('data_contaminated_for_test_evaluation') for model in selected)


def test_paper_missing_metrics_are_not_invented():
    papers=research()
    assert len(papers)==5
    assert all(paper['url'].startswith('https://') for paper in papers)
    assert all('f05' not in paper['metrics'] for paper in papers)
    pan=next(paper for paper in papers if paper['id']=='panderm2025')
    assert pan['metrics']['weighted_f1']==92.6 and 'macro_f1' not in pan['metrics']
    disputed=next(paper for paper in papers if paper['id']=='musthafa2024')
    assert disputed['metrics']['accuracy']==97.858
    assert disputed['additional_results'][0]['metrics']['accuracy']==76
    assert disputed['additional_results'][0]['metrics']['macro_f1']==48
    for paper in papers:
        if paper['per_class']: assert set(paper['per_class'])==set(CLASS_LABELS)


def test_upload_rejects_corrupt_oversized_but_accepts_small_images():
    with pytest.raises(ValueError,match='readable'): decode_image(b'not an image')
    with pytest.raises(ValueError,match='10 MB'): decode_image(b'x'*(10*1024*1024+1))
    data=BytesIO();Image.new('RGB',(32,32)).save(data,format='PNG')
    assert decode_image(data.getvalue()).size==(32,32)


def test_solid_colour_is_rejected_even_when_rgb_channel_values_differ():
    image=Image.new('RGB',(224,224),(200,100,20))
    data=BytesIO();image.save(data,format='PNG')
    assert quality_issue(decode_image(data.getvalue())) is not None
