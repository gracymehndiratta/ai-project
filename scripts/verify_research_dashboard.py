"""Audit the six-model board and exercise the real Streamlit uploader/inference."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def main():
    os.chdir(ROOT)
    import json
    import shutil
    from io import BytesIO
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from src.ui.results import all_models,top_six,CLASS_LABELS,research,adapted_models,local_research
    from src.ui.dashboard import classify_image
    from src.ui.uploads import decode_image
    from src.data.full_isic import CLASS_NAMES
    from src.utils.feature_training import score_predictions
    from src.models.panderm import sha256_file
    torch.set_num_threads(4)
    out=Path('artifacts/ui_qa');out.mkdir(parents=True,exist_ok=True)
    from src.ui.catalog import saved_models,benchmark_models
    selected=benchmark_models(saved_models());assert len(selected)==6 and sum(m['neural'] for m in selected)==2
    ref=local_research()[0]
    reference={'name':ref['name'],'folder':Path(ref['source_folder']),
        'info':json.loads((Path(ref['source_folder'])/'metadata.json').read_text(encoding='utf-8')),
        'metrics':ref['metrics']}
    baseline_ids=None;checks=[]
    for model in selected+[reference]:
        prediction_folder=Path(ref['metrics_path']).parent if model is reference else model['folder']
        saved=pd.read_csv(prediction_folder/'official_test_predictions.csv')
        assert len(saved)==1512 and saved.image_id.is_unique
        truth=saved.true_label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
        metric=score_predictions(truth,saved[CLASS_NAMES].to_numpy())
        for key in ['accuracy','macro_f1','macro_f05','weighted_f1','balanced_accuracy']:
            np.testing.assert_allclose(metric[key],model['metrics']['official_test'][key],atol=1e-12)
        labels=dict(zip(saved.image_id,saved.true_label))
        if baseline_ids is None: baseline_ids=labels
        else: assert labels==baseline_ids
    frame=pd.read_csv(selected[0]['folder'].parent/'split_manifest.csv')
    probes=frame[frame.split=='official_test'].groupby('label',sort=True).head(1)
    for _,row in probes.iterrows():
        result=classify_image(decode_image(Path(row.image_path).read_bytes()),selected[0])
        checks.append({'image_id':row.image_id,'true_label':row.label,**result})
    assert sum(row['status']=='classified' for row in checks)>=6,checks
    positive=next(row for row in checks if row['status']=='classified')
    image_path=Path(frame[frame.image_id==positive['image_id']].iloc[0].image_path)
    shutil.copyfile(image_path,out/'lesion.jpg')
    for model in selected+[reference]:
        result=classify_image(decode_image(image_path.read_bytes()),model)
        assert result['status']=='classified',(model['name'],result)
        assert len(result['probabilities'])==7 and np.isclose(sum(result['probabilities'].values()),1)
    chart=Path('outputs/feature_models/panderm_large_capacity/capacity_search.png')
    chart_result=classify_image(decode_image(chart.read_bytes()),selected[0])
    assert chart_result['status']=='classified' and len(chart_result['probabilities'])==7,chart_result
    app=AppTest.from_file(str(ROOT/'scripts/app.py'),default_timeout=180).run()
    assert not app.exception,app.exception
    assert len(app.tabs)==5
    assert len(app.dataframe[0].value)==6
    app.sidebar.radio(key='checkpoint_stage').set_value('Original checkpoints').run()
    assert not app.exception
    assert not any('Fitting data' in str(value) for value in app.dataframe[0].value['Data role'])
    app.sidebar.radio(key='checkpoint_stage').set_value('Adapted models').run()
    assert not app.exception
    for value in app.sidebar.selectbox[0].options:
        app.sidebar.selectbox[0].set_value(value).run();assert not app.exception,app.exception
    app.sidebar.selectbox[0].set_value(app.sidebar.selectbox[0].options[0]).run()
    app.file_uploader[0].set_value(('lesion.jpg',image_path.read_bytes(),'image/jpeg')).run()
    assert not app.exception and not app.error,app.error
    app.button(key='analyze_image').click().run()
    assert not app.exception and not app.error,app.error
    assert any('Predicted category:' in value.value for value in app.success),[v.value for v in app.warning]
    positive_message=app.success[0].value
    blank=BytesIO();Image.new('RGB',(224,224),'white').save(blank,format='PNG')
    app.file_uploader[0].set_value(('blank.png',blank.getvalue(),'image/png')).run()
    app.button(key='analyze_image').click().run()
    assert any('Predicted category:' in value.value for value in app.success)
    assert not any('withheld' in value.value or 'uniform or blank' in value.value for value in app.warning)
    app.file_uploader[0].set_value(('bad.jpg',b'not an image','image/jpeg')).run()
    assert any('not a readable' in value.value for value in app.error)
    assert not app.exception
    app.file_uploader[0].set_value(('chart.png',chart.read_bytes(),'image/png')).run()
    app.button(key='analyze_image').click().run()
    assert any('Predicted category:' in value.value for value in app.success)
    assert not app.exception
    assert not any('withheld' in value.value for value in app.warning)
    receipt={'passed':True,'models':[model['name'] for model in selected],'neural_heads':sum(m['neural'] for m in selected),
        'same_official_test_image_ids_and_labels':True,'metrics_recomputed':7,'research_papers':len(research()),
        'local_research_reference_inference_passed':True,'original_adapted_toggle_passed':True,
        'inference_class_probes':checks,'all_six_classifiers_inference_passed':True,'non_lesion_chart_prediction':chart_result,
        'streamlit_upload_checks':['valid JPEG produces seven probabilities','blank PNG produces seven probabilities','corrupt JPEG rejected','chart produces seven probabilities'],
        'all_four_cohort_views_passed':True,'positive_upload_message':positive_message,
        'content_similarity_filter_enabled':False,'binary_skin_vs_non_skin_validation':False}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
