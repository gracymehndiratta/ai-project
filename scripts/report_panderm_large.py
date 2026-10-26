"""Independently verify and report the four PanDerm Large classifiers."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def table(frame):
    lines = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def main():
    os.chdir(ROOT)
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import ImageFeaturePredictor, load_classifier
    from src.models.panderm import sha256_file
    from src.utils.feature_training import score_predictions
    torch.set_num_threads(4)
    output = Path('outputs/feature_models/panderm_large_comparison')
    frame = pd.read_csv(output/'split_manifest.csv')
    cache = Path('data/features/panderm_large_72dfc47de5e585b2.npy')
    receipt = json.loads(cache.with_suffix('.json').read_text())
    assert receipt['complete'] and sha256_file(cache) == receipt['embeddings_sha256']
    features = np.load(cache, mmap_mode='r')
    labels = frame.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    train = np.flatnonzero(frame.split=='train')
    assert len(train)==8207
    paper = pd.read_csv('artifacts/panderm_paper/metadata/HAM10000_clean/ISIC2018_splits/HAM_clean.csv')
    for name, upstream in [('train','train'),('validation','val'),('test','test')]:
        assert set(frame[frame.split==name].image_id)==set(paper[paper.split==upstream].image_id)
    assert frame[frame.split.isin(['train','validation','test'])].groupby('lesion_id').split.nunique().eq(1).all()
    marker_path = output/'selection_complete.json'
    marker = json.loads(marker_path.read_text())
    previous = json.loads((output/'previous_artifact_hashes.json').read_text())
    assert all(sha256_file(path)==digest for path,digest in previous.items())
    records, models = [], {}
    metric_names = ['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']
    for kind in ['paper','linear','mlp','xgboost']:
        folder = output/kind
        model, _, info = load_classifier(folder)
        models[kind] = model
        assert info['backbone']=='panderm_large' and info['feature_dimension']==1024
        assert marker['models'][kind]==sha256_file(folder/'model.joblib')
        if kind!='paper':
            assert model.projection.n_samples_seen_==8207
            np.testing.assert_allclose(model.projection.mean_,features[train].mean(0,dtype=np.float64),atol=1e-10)
        else:
            original = joblib.load('outputs/diagnostics/panderm_paper_reference/model.joblib')
            for key, tensor in original.models[0].state_dict().items():
                torch.testing.assert_close(tensor,model.models[0].state_dict()[key],rtol=0,atol=0)
        selection = json.loads((folder/'selection.json').read_text())
        if kind=='mlp':
            assert all(trial['epochs_observed']>=40 for trial in selection['trials'])
            assert selection['selected']['architecture']==[1024,256,64,7]
        if kind=='xgboost':
            assert model.models[0].get_booster().num_boosted_rounds()==selection['selected']['best_round']
            for trial in selection['trials']:
                history = trial['history']['validation_0']
                assert set(history)=={'macro_f1_cost'}
                assert abs(trial['validation_macro_f1'] + history['macro_f1_cost'][trial['best_round']-1]) < 1e-6
        metrics = json.loads((folder/'metrics.json').read_text())
        for split in ['train','validation','test','official_validation','official_test']:
            rows = np.flatnonzero(frame.split==split)
            path = folder/f'{split}_predictions.csv'
            saved = pd.read_csv(path)
            assert saved.image_id.tolist()==frame.iloc[rows].image_id.tolist()
            assert saved.true_label.tolist()==frame.iloc[rows].label.tolist()
            assert set(saved.image_id).isdisjoint(set(frame.iloc[train].image_id)) if split!='train' else True
            if split in ['test','official_test']:
                assert marker_path.stat().st_mtime_ns<=path.stat().st_mtime_ns
            predicted = model.predict_proba(np.asarray(features[rows]))
            np.testing.assert_allclose(predicted,saved[CLASS_NAMES].to_numpy(),atol=1e-7)
            measured = score_predictions(labels[rows],saved[CLASS_NAMES].to_numpy())
            for name in metric_names:
                np.testing.assert_allclose(measured[name],metrics[split][name],atol=1e-12)
            records.append({'model':kind,'split':split,'images':len(rows),**{name:measured[name] for name in metric_names}})
    # Verify actual image -> released Large encoder -> all four serialized heads.
    predictor = ImageFeaturePredictor(output/'linear','cuda' if torch.cuda.is_available() else 'cpu')
    assert all(not parameter.requires_grad for parameter in predictor.extractor.parameters())
    image_checks = []
    probes = frame[frame.split=='test'].groupby('label',sort=True).head(1)
    for index,row in probes.iterrows():
        with Image.open(row.image_path) as image:
            batch = predictor.transform(image.convert('RGB')).unsqueeze(0).to(predictor.device)
        with torch.inference_mode():
            actual = predictor.extractor(batch).cpu().numpy()
        np.testing.assert_allclose(actual[0],features[index],rtol=1e-4,atol=1e-4)
        for kind,model in models.items():
            direct = model.predict_proba(actual)
            cached = model.predict_proba(np.asarray(features[index:index+1]))
            assert direct.argmax()==cached.argmax()
            np.testing.assert_allclose(direct,cached,rtol=1e-3,atol=1e-4)
            image_checks.append({'image_id':row.image_id,'model':kind,'max_probability_delta':float(abs(direct-cached).max())})
    results = pd.DataFrame(records)
    results.to_csv(output/'verified_statistics.csv',index=False)
    # Historical models share only the external official test population.
    history = []
    for path in Path('outputs/feature_models').glob('*/*/metrics.json'):
        if output in path.parents: continue
        old = json.loads(path.read_text())
        if 'official_test' in old:
            history.append({'experiment':path.parent.parent.name,'model':path.parent.name,
                **{k:old['official_test'].get(k, float('nan')) for k in metric_names}})
    historical = pd.concat([pd.DataFrame(history), results[results.split=='official_test'].assign(experiment=output.name).drop(columns=['split','images'])],ignore_index=True)
    historical.to_csv(output/'historical_official_test.csv',index=False)
    shown = ['model','accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05']
    def percentages(split):
        selected = results[results.split==split][shown].copy()
        selected[shown[1:]] *= 100
        return table(selected)
    chosen = json.loads((output/'best_model.json').read_text())['model']
    gaps = results[results.split.isin(['train','validation','test'])].pivot(index='model',columns='split',values='macro_f1').reset_index()
    gaps[['train','validation','test']] *= 100
    text = f'''# PanDerm Large: four classifiers

All four use the same frozen published ViT-Large backbone (303,326,208 parameters, 1024 features), checkpoint SHA256 {receipt['checkpoint_sha256']}. The encoder and authors' preprocessing are unchanged. No nearest-neighbor substitute.

## Shared data and evaluation

The authors' released HAM CSV defines 8,207 training, 575 validation and 1,232 test images. All recognized rows are used, with lesion-disjoint partitions; its one `trainisic` row is excluded exactly as upstream. The released CSV differs from the paper's stated 60/20/20 proportions. Official validation (193) and test (1,512) images remain separate. No dataset subsampling. No demographic inputs or SMOTE in this experiment.

All custom-head hyperparameters and checkpoints were selected using validation macro F1 only; the authors' probe has fixed parameters and was retained unchanged from its verified completed training. All heads were frozen before this run's test evaluation. These test datasets have been examined in earlier experiments, so they are not fresh blind benchmarks. Pretraining overlap with downstream datasets has not been independently excluded.

## Released HAM test: 1,232 images (percent)

{percentages('test')}

## Official challenge test: 1,512 images (percent)

{percentages('official_test')}

Macro scores give each lesion class equal weight. Weighted scores use the number of images in each class; the paper's reported 92.6 F1 is weighted F1. These averages must not be interchanged.

## Training and validation macro F1 (percent)

{table(gaps)}

Validation-selected model: **{chosen}**. High train F1 alone does not establish good generalization. Gaps are reported rather than hidden by training-score selection.

## Classifiers

- `paper`: unchanged authors' logistic-regression implementation on raw features, C=71.68, LBFGS max_iter=1000, seed=0. No fitted feature scaling or class weighting. This is the released-code reference, with current runtime/hardware, not a guarantee of exact manuscript replication.
- `linear`: train-only standardization; L2 logistic regression, C grid .0003/.001/.003/.01/.1, class weight powers .5/1, max_iter=2000.
- `mlp`: 1024 -> 256 -> 64 -> 7, LayerNorm/ReLU, dropout .4/.25, AdamW lr .0007, weight decay .02, smoothing .03; at least 40 observed epochs, up to 180, patience 30, best validation checkpoint restored. Class-weight power .5/1 selected on validation.
- `xgboost`: depth 2/3, class weight power .5/1, learning rate .04, modest L1/L2 penalties, up to 1200 rounds, at least 200 observations, patience 100; best validation checkpoint restored.

## Saved outputs

Each model folder has `model.joblib`, metadata, complete training/selection details, metrics for all five splits, and image-level probabilities. `per_class_statistics.csv` provides precision, recall, F1, F0.5 and support for each lesion. `historical_official_test.csv` preserves comparisons with earlier experiments on the same external test images; training sizes/protocols differ, so it is not an architecture-only comparison. Older internal-test numbers use another split and should not be compared directly to this released HAM test.

All {len(previous)} earlier artifacts were verified unchanged. Twenty saved prediction files and all requested metrics were independently recomputed. Actual image inference was checked for all seven classes through all four heads (28 checks).

Re-run: `.\\.venv\\Scripts\\python.exe scripts/train_panderm_large_heads.py`, then `.\\.venv\\Scripts\\python.exe scripts/report_panderm_large.py`.
'''
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    (output/'verification.json').write_text(json.dumps({'passed':True,'previous_artifacts_unchanged':len(previous),
        'prediction_files_recomputed':20,'image_checks':image_checks,'same_released_split_all_heads':True,
        'encoder_frozen':True,'authors_head_unchanged':True},indent=2))
    print(results.to_string(index=False),flush=True)
    print('Verified previous artifacts, split integrity, serialized models, 20 prediction files and 28 actual image/head checks.',flush=True)


if __name__=='__main__': main()
