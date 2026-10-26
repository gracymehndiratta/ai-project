"""Verify two additional models and compare without modifying earlier results."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def table(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    os.chdir(ROOT)
    import joblib
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    from sklearn.preprocessing import StandardScaler
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import ImageFeaturePredictor,load_classifier
    from src.models.panderm import sha256_file
    from src.utils.feature_training import score_predictions
    from src.utils.panderm_refinement import OUTPUT,CACHE
    torch.set_num_threads(4)
    frame=pd.read_csv(OUTPUT/'split_manifest.csv')
    original=Path('outputs/feature_models/panderm_large_comparison')
    pd.testing.assert_frame_equal(frame,pd.read_csv(original/'split_manifest.csv'))
    features=np.load(CACHE,mmap_mode='r')
    train=np.flatnonzero(frame.split=='train')
    truth=frame.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    fold_frame=pd.read_csv(OUTPUT/'training_folds.csv')
    assert fold_frame.image_id.tolist()==frame.iloc[train].image_id.tolist()
    assert fold_frame.groupby('lesion_id').cv_fold.nunique().eq(1).all()
    previous=json.loads((OUTPUT/'previous_artifact_hashes.json').read_text())
    assert all(sha256_file(path)==digest for path,digest in previous.items())
    trials=[json.loads(path.read_text()) for path in sorted((OUTPUT/'trials').glob('*.json'))]
    assert len(trials)==29
    for trial in trials:
        probabilities=np.load(OUTPUT/'trials'/f'{trial["number"]:02d}_oof.npy')
        measured=score_predictions(truth[train],probabilities)
        np.testing.assert_allclose(measured['macro_f1'],trial['oof_macro_f1'],atol=1e-12)
        for detail in trial['folds']:
            held=fold_frame.cv_fold.to_numpy()==detail['fold']
            assert held.sum()==detail['holdout_images']
            measured=score_predictions(truth[train][held],probabilities[held])
            np.testing.assert_allclose(measured['macro_f1'],detail['holdout_macro_f1'],atol=1e-12)
            assert detail['observations'] >= (400 if trial['config']['kind']=='xgboost' else 50)
    marker_path=OUTPUT/'selection_complete.json';marker=json.loads(marker_path.read_text())
    metrics_names=['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']
    records,models=[],{}
    for kind in ['mlp','xgboost']:
        folder=OUTPUT/kind
        model,_,info=load_classifier(folder);models[kind]=model
        assert info['backbone']=='panderm_large' and info['feature_dimension']==1024
        assert sha256_file(folder/'model.joblib')==marker['models'][kind]
        scaler=model.projection if isinstance(model.projection,StandardScaler) else model.projection.steps[0][1]
        assert scaler.n_samples_seen_==8207
        np.testing.assert_allclose(scaler.mean_,features[train].mean(0,dtype=np.float64),atol=1e-10)
        selection=json.loads((folder/'selection.json').read_text())
        if selection['config']['representation']=='lda6':
            lda=model.projection.steps[1][1]
            transformed=scaler.transform(np.array(features[train]))
            expected=np.array([transformed[truth[train]==label].mean(0) for label in range(7)])
            np.testing.assert_allclose(lda.means_,expected,atol=2e-6,rtol=1e-4)
            assert lda.transform(transformed[:2]).shape==(2,6)
        candidates=[r for r in trials if r['config']['kind']==kind]
        assert selection['oof_macro_f1']==max(t['oof_macro_f1'] for t in candidates)
        assert selection['selected_without_validation_or_test']
        if kind=='mlp':
            assert [(layer.in_features,layer.out_features) for layer in model.models[0] if isinstance(layer,torch.nn.Linear)]==[(1024,192),(192,64),(64,7)]
        else:
            assert model.models[0].get_booster().num_boosted_rounds()==selection['median_budget']
        metrics=json.loads((folder/'metrics.json').read_text())
        for split in ['train','validation','test','official_validation','official_test']:
            rows=np.flatnonzero(frame.split==split)
            saved_path=folder/f'{split}_predictions.csv';saved=pd.read_csv(saved_path)
            assert saved.image_id.tolist()==frame.iloc[rows].image_id.tolist()
            assert saved.true_label.tolist()==frame.iloc[rows].label.tolist()
            if split in ['test','official_test']: assert marker_path.stat().st_mtime_ns<=saved_path.stat().st_mtime_ns
            probability=model.predict_proba(np.array(features[rows]))
            np.testing.assert_allclose(probability,saved[CLASS_NAMES].to_numpy(),atol=1e-7)
            measured=score_predictions(truth[rows],saved[CLASS_NAMES].to_numpy())
            for name in metrics_names: np.testing.assert_allclose(measured[name],metrics[split][name],atol=1e-12)
            records.append({'model':'New smaller NN' if kind=='mlp' else 'New tuned XGBoost','split':split,**{k:measured[k] for k in metrics_names}})
    for kind in ['paper','linear','mlp','xgboost']:
        old=json.loads((original/kind/'metrics.json').read_text())
        for split in ['train','validation','test','official_validation','official_test']:
            records.append({'model':{'paper':'Authors original','linear':'Existing linear','mlp':'Existing NN','xgboost':'Existing XGBoost'}[kind],
                'split':split,**{k:old[split][k] for k in metrics_names}})
    combined=pd.DataFrame(records)
    combined.to_csv(OUTPUT/'comparison_all_six.csv',index=False)
    wide=combined.pivot(index='model',columns='split',values=metrics_names)
    wide.columns=[f'{split}_{metric}' for metric,split in wide.columns]
    wide.reset_index().to_csv(OUTPUT/'comparison.csv',index=False)
    predictor=ImageFeaturePredictor(OUTPUT/'mlp','cuda' if torch.cuda.is_available() else 'cpu')
    assert all(not p.requires_grad for p in predictor.extractor.parameters())
    checks=[]
    for index,row in frame[frame.split=='test'].groupby('label',sort=True).head(1).iterrows():
        with Image.open(row.image_path) as image: batch=predictor.transform(image.convert('RGB')).unsqueeze(0).to(predictor.device)
        with torch.inference_mode(): actual=predictor.extractor(batch).cpu().numpy()
        np.testing.assert_allclose(actual[0],features[index],rtol=1e-4,atol=1e-4)
        for kind,model in models.items():
            direct=model.predict_proba(actual);cached=model.predict_proba(np.array(features[index:index+1]))
            np.testing.assert_allclose(direct,cached,rtol=1e-3,atol=1e-4)
            assert direct.argmax()==cached.argmax()
            checks.append({'image_id':row.image_id,'model':kind,'max_probability_delta':float(abs(direct-cached).max())})
    def percentages(split):
        columns=['model','accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05']
        selected=combined[combined.split==split][columns].copy();selected[columns[1:]]*=100
        return table(selected)
    gaps=combined[combined.split.isin(['train','validation','test','official_test'])].pivot(index='model',columns='split',values='macro_f1').reset_index()
    gaps[['train','validation','test','official_test']]*=100
    training=combined[combined.split=='train'][['model','accuracy','macro_f1']].copy();training[['accuracy','macro_f1']]*=100
    selections={k:json.loads((OUTPUT/k/'selection.json').read_text()) for k in models}
    parameters=sum(p.numel() for p in models['mlp'].models[0].parameters())
    old_parameters=sum(p.numel() for p in joblib.load(original/'mlp'/'model.joblib').models[0].parameters())
    reference=json.loads((OUTPUT/'linear_reference_cv.json').read_text()) if (OUTPUT/'linear_reference_cv.json').exists() else None
    reference_note=(f"The existing linear configuration, refitted on the same training folds for diagnosis only, has pooled OOF macro F1 {reference['oof_macro_f1']*100:.2f}%. Its settings were previously chosen on the released validation set, so it is a reference rather than a newly nested model-selection estimate." if reference else '')
    text=f'''# Additional PanDerm Large models

All earlier models are retained unchanged. Both additions use exactly the same frozen 303,326,208-parameter published Large backbone, preprocessing, cached embeddings and released HAM split as the previous four-model comparison. No SMOTE, demographic input or extra dataset is used.

## Training accuracy and F1 (percent)

{table(training)}

Low test accuracy does not rule out overfitting. Compare training and held-out scores, noting that class proportions differ. The earlier neural network's accuracy advantage over the authors' probe was just one additional correct image out of 1,232; worse macro F1/F0.5 can reflect minority-class errors rather than overfitting alone. Its training-to-test F1 gap is stronger evidence. Shallow axis-aligned trees are not equivalent to an arbitrary linear decision surface on distributed neural embeddings; there is no guaranteed XGBoost advantage.

## Released HAM test: 1,232 images (percent)

{percentages('test')}

## Official challenge test: 1,512 images (percent)

{percentages('official_test')}

Macro scores weight lesion types equally; weighted scores weight image counts. No test score was used to select these new settings. Test datasets have been inspected previously and should be treated as reused comparison benchmarks, not fresh blind validation.

## Generalization gaps: macro F1 (percent)

{table(gaps)}

## Tuning method

The previous validation set has only five dermatofibroma and seven vascular images. To reduce selection sensitivity to those few cases, all new candidate selection takes place in three stratified, lesion-disjoint folds **inside the 8,207 training images**. Every fold trains its own scaler and PCA if applicable. No validation or test image fits preprocessing or the heads. Fold held-out scores select stopping budgets, so the cross-validation score is a tuning diagnostic, not an unbiased performance estimate.

Twenty initial XGBoost configurations compare raw 1,024 features with training-fold PCA to 128/256 features, depths 1/2/3/4/6, unweighted versus square-root class weighting, lower L2=5, no L1, minimum child weight 1/3, learning rate .04, and .9 row/column sampling. Before any new test evaluation, six further configurations were added: shrinkage linear discriminant analysis (LDA) transforms 1,024 features to six supervised discriminative directions, followed by depth 1/2/3 trees and the same two weight settings. LDA uses only fitting-fold labels and is refitted on the full training set for final inference. This is an explicitly supervised feature projection, not a raw-embedding-only tree head. The original pre-extension protocol is retained for audit.

Each fold allows up to 2,000 rounds, observes at least 400, and allows 200 rounds without macro-F1 improvement. The winning configuration is refitted on all 8,207 rows at its median best fold budget. No original model is overwritten.

Selected XGBoost: `{json.dumps(selections['xgboost']['config'],sort_keys=True)}`, {selections['xgboost']['median_budget']} rounds; tuning OOF macro F1 {selections['xgboost']['oof_macro_f1']*100:.2f}%.

{reference_note}

The smaller neural head is **1024 -> 192 -> 64 -> 7**, with {parameters:,} trainable parameters versus {old_parameters:,} before ({(1-parameters/old_parameters)*100:.1f}% fewer). LayerNorm/ReLU, dropout .3/.2 or .4/.2, modest AdamW weight decay .01/.02, smoothing .03 and weight power .25/.5 are compared in three predefined configurations. At least 50 of up to 180 epochs are observed, with patience 40; the selected head is refitted at the median best fold budget. Fixed lr=.0007 in both folds and final refit preserves the training schedule.

Selected smaller NN: `{json.dumps(selections['mlp']['config'],sort_keys=True)}`, {selections['mlp']['median_budget']} epochs; tuning OOF macro F1 {selections['mlp']['oof_macro_f1']*100:.2f}%.

Both final models were frozen before validation/test reporting. Full fold histories, image-level OOF probabilities, final split probabilities, per-lesion statistics and selected configurations are saved alongside this report. The original paper CSV includes one unrecognized `trainisic` row excluded exactly as upstream; its split proportions differ from the manuscript's stated 60/20/20.

## Verification

All {len(previous)} previous artifacts have matching SHA256 hashes. All 29 candidate OOF scores, ten final prediction files, training-only preprocessing and serialized-model inference were independently checked. Actual image inference was verified through all seven lesion classes and both new heads (14 checks). `train-panderm-refined.cmd` reproduces training and this report using resumable trial outputs.
'''
    (OUTPUT/'REPORT.md').write_text(text,encoding='utf-8')
    (OUTPUT/'verification.json').write_text(json.dumps({'passed':True,'previous_artifacts_unchanged':len(previous),
        'candidate_oof_scores_recomputed':len(trials),'final_prediction_files_recomputed':10,'image_checks':checks,
        'nn_parameters':parameters,'old_nn_parameters':old_parameters,'training_images':len(train),'no_validation_or_test_selection':True},indent=2))
    print(combined.to_string(index=False),flush=True)
    print('Verified all prior artifacts, 29 tuning scores, ten final prediction files and 14 actual image/head checks.',flush=True)


if __name__=='__main__': main()
