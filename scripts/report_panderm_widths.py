"""Verify three narrower heads and compare all nine PanDerm Large models."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{value:.2f}' if isinstance(value,float) else str(value) for value in row)+' |')
    return '\n'.join(lines)


def main():
    os.chdir(ROOT)
    import joblib
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import ImageFeaturePredictor,load_classifier
    from src.models.panderm import sha256_file
    from src.utils.feature_training import score_predictions
    from src.utils.panderm_widths import OUTPUT,CACHE,WIDTHS
    torch.set_num_threads(4)
    frame=pd.read_csv(OUTPUT/'split_manifest.csv')
    pd.testing.assert_frame_equal(frame,pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv'))
    folds=pd.read_csv(OUTPUT/'training_folds.csv')
    pd.testing.assert_frame_equal(folds,pd.read_csv('outputs/feature_models/panderm_large_refined/training_folds.csv'))
    train=np.flatnonzero(frame.split=='train')
    assert frame.iloc[train].image_id.tolist()==folds.image_id.tolist()
    assert folds.groupby('lesion_id').cv_fold.nunique().eq(1).all()
    features=np.load(CACHE,mmap_mode='r')
    truth=frame.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    previous=json.loads((OUTPUT/'previous_artifact_hashes.json').read_text())
    assert all(sha256_file(path)==digest for path,digest in previous.items())
    marker_path=OUTPUT/'selection_complete.json';marker=json.loads(marker_path.read_text())
    metric_names=['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']
    models,records,sizes={},{},[]
    rows=[]
    names={'half':'Half NN 96-32','quarter':'Quarter NN 48-16','tiny':'Tiny NN 8-4'}
    for name,widths in WIDTHS.items():
        candidates=[]
        for index in range(3):
            trial=json.loads((OUTPUT/'trials'/f'{name}_{index}.json').read_text())
            probability=np.load(OUTPUT/'trials'/f'{name}_{index}_oof.npy')
            metric=score_predictions(truth[train],probability)
            np.testing.assert_allclose(metric['macro_f1'],trial['oof_macro_f1'],atol=1e-12)
            for detail in trial['folds']:
                held=folds.cv_fold.to_numpy()==detail['fold']
                metric=score_predictions(truth[train][held],probability[held])
                np.testing.assert_allclose(metric['macro_f1'],detail['holdout_macro_f1'],atol=1e-12)
                assert detail['observations']>=50
            candidates.append(trial)
        folder=OUTPUT/name
        model,_,info=load_classifier(folder);models[name]=model
        assert sha256_file(folder/'model.joblib')==marker['models'][name]
        assert info['backbone']=='panderm_large' and info['feature_dimension']==1024
        assert [(layer.in_features,layer.out_features) for layer in model.models[0] if isinstance(layer,torch.nn.Linear)]==[(1024,widths[0]),(widths[0],widths[1]),(widths[1],7)]
        assert model.projection.n_samples_seen_==8207
        np.testing.assert_allclose(model.projection.mean_,features[train].mean(0,dtype=np.float64),atol=1e-10)
        selection=json.loads((folder/'selection.json').read_text())
        assert selection['selected_without_validation_or_test']
        assert selection['oof_macro_f1']==max(t['oof_macro_f1'] for t in candidates)
        assert selection['median_budget']==selection['refit']['observations']
        sizes.append({'model':names[name],'hidden_units':f'{widths[0]} / {widths[1]}','parameters':sum(p.numel() for p in model.models[0].parameters()),
            'epochs':selection['median_budget'],'tuning_macro_f1':selection['oof_macro_f1']*100})
        metrics=json.loads((folder/'metrics.json').read_text())
        for split in ['train','validation','test','official_validation','official_test']:
            indices=np.flatnonzero(frame.split==split)
            saved_path=folder/f'{split}_predictions.csv';saved=pd.read_csv(saved_path)
            assert saved.image_id.tolist()==frame.iloc[indices].image_id.tolist()
            assert saved.true_label.tolist()==frame.iloc[indices].label.tolist()
            if split in ['test','official_test']: assert marker_path.stat().st_mtime_ns<=saved_path.stat().st_mtime_ns
            probability=model.predict_proba(np.array(features[indices]))
            np.testing.assert_allclose(probability,saved[CLASS_NAMES].to_numpy(),atol=1e-7)
            measured=score_predictions(truth[indices],saved[CLASS_NAMES].to_numpy())
            for key in metric_names: np.testing.assert_allclose(measured[key],metrics[split][key],atol=1e-12)
            rows.append({'model':names[name],'split':split,**{key:measured[key] for key in metric_names}})
    old=pd.read_csv('outputs/feature_models/panderm_large_refined/comparison_all_six.csv')
    old['model']=old.model.replace({'Existing NN':'NN 256-64','New smaller NN':'NN 192-64','Existing linear':'Linear head',
        'Authors original':'Authors original','Existing XGBoost':'Original XGBoost','New tuned XGBoost':'Tuned XGBoost + LDA'})
    combined=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    assert combined.model.nunique()==9
    combined.to_csv(OUTPUT/'comparison_all_nine.csv',index=False)
    wide=combined.pivot(index='model',columns='split',values=metric_names)
    wide.columns=[f'{split}_{metric}' for metric,split in wide.columns]
    wide.reset_index().to_csv(OUTPUT/'comparison.csv',index=False)
    pd.DataFrame(sizes).to_csv(OUTPUT/'architectures.csv',index=False)
    # Load the backbone once, and check each new head on a real image per class.
    predictor=ImageFeaturePredictor(OUTPUT/'half','cuda' if torch.cuda.is_available() else 'cpu')
    assert all(not p.requires_grad for p in predictor.extractor.parameters())
    checks=[]
    for index,row in frame[frame.split=='test'].groupby('label',sort=True).head(1).iterrows():
        with Image.open(row.image_path) as image: batch=predictor.transform(image.convert('RGB')).unsqueeze(0).to(predictor.device)
        with torch.inference_mode(): actual=predictor.extractor(batch).cpu().numpy()
        np.testing.assert_allclose(actual[0],features[index],rtol=1e-4,atol=1e-4)
        for name,model in models.items():
            direct=model.predict_proba(actual);cached=model.predict_proba(np.array(features[index:index+1]))
            np.testing.assert_allclose(direct,cached,rtol=1e-3,atol=1e-4)
            assert direct.argmax()==cached.argmax()
            checks.append({'image_id':row.image_id,'model':name,'max_probability_delta':float(abs(direct-cached).max())})
    def percentages(split):
        columns=['model','accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05']
        data=combined[combined.split==split][columns].copy();data[columns[1:]]*=100
        return table(data)
    gaps=combined[combined.split.isin(['train','validation','test','official_test'])].pivot(index='model',columns='split',values='macro_f1').reset_index()
    gaps[['train','validation','test','official_test']]*=100
    parameters=pd.DataFrame(sizes)
    text=f'''# PanDerm Large: three additional narrower neural heads

The new heads have two hidden layers each: half the previous 192/64 widths (96/32), one quarter (48/16), and a tiny 8/4 head. All use the same frozen published PanDerm Large encoder and 1,024 cached image features. No dataset subsampling, new dataset, metadata or SMOTE. Every earlier model and result is retained unchanged.

## Architectures and training budgets

{table(parameters)}

The linear head has 7,175 trainable parameters. The tiny head's 8,295 parameters are about 15.6% more, but its nonlinear bottleneck does not contain every linear classifier as a special case. This tests an alternative compact nonlinear head; it does not isolate a strict capacity increase over logistic regression. LayerNorm and ReLU follow both hidden layers; softmax follows seven output scores at prediction time.

## Released HAM test: 1,232 images, percent

{percentages('test')}

## Official challenge test: 1,512 images, percent

{percentages('official_test')}

Macro metrics weight all seven lesion classes equally. Weighted metrics weight their image counts. All heads use the authors' released split: 8,207 training, 575 validation, 1,232 test, plus 193 official validation and 1,512 official test images. The one unrecognized `trainisic` row is excluded exactly as upstream. The released split differs from the manuscript's stated 60/20/20 proportions.

## Training and held-out macro F1, percent

{table(gaps)}

Training accuracy and all other metrics for each split are in `comparison_all_nine.csv`. Per-lesion precision, recall, F1, F0.5 and support for the three new heads are in `per_class_statistics.csv`.

## Selection and regularization

All three architectures use exactly the same three lesion-disjoint, stratified folds **within the 8,207 training images** as the previous 192/64 experiment. Each fold fits its own scaler. For half/quarter heads the same three settings as the 192/64 search are compared: weight powers .25/.5, weight decay .01/.02, dropout .3/.2 or .4/.2. Tiny heads use lighter dropout (0/0 or .1/.1) and weight decay (.005/.01) to avoid suppressing their few units. All use AdamW, fixed learning rate .0007, batch128 and label smoothing .03.

Up to 180 epochs are allowed; at least 50 are observed before patience40 may stop a fold. The best scoring fold checkpoint can be earlier. Pooled OOF macro F1 selects one configuration per architecture; its median best epoch determines final fitting on all 8,207 training rows. Final networks are frozen before validation/test evaluation. No validation or test score selects settings. These fold scores are tuning diagnostics because fold holdouts also choose stopping epochs, not unbiased cross-validation estimates. Test datasets have been inspected in prior experiments, so they are reused comparisons, not fresh blind evaluations.

Architecture size is not the only difference from all historical runs: some older heads used validation selection or another learning-rate schedule. This is a model comparison, not a controlled proof that width alone caused a metric change.

## Verification and reproducibility

All {len(previous)} earlier artifacts retain their SHA256 hashes. Nine saved candidate OOF scores and their 27 fold scores, all 15 final prediction files and requested metrics, train-only scalers and the serialized architectures were independently checked. Actual-image inference was verified on all seven classes through all three heads (21 checks). Every model has a checkpoint, metadata, full selection history, metrics and image-level probabilities.

Run `train-panderm-widths.cmd` to resume/reproduce training and regenerate this report. The app offers each new architecture separately alongside earlier models.
'''
    (OUTPUT/'REPORT.md').write_text(text,encoding='utf-8')
    (OUTPUT/'verification.json').write_text(json.dumps({'passed':True,'previous_artifacts_unchanged':len(previous),
        'candidate_oof_scores_recomputed':9,'fold_scores_recomputed':27,'final_prediction_files_recomputed':15,
        'image_checks':checks,'same_folds_and_split_as_192_64':True,'no_validation_or_test_selection':True},indent=2))
    print(combined.to_string(index=False),flush=True)
    print(pd.DataFrame(sizes).to_string(index=False),flush=True)
    print('Verified all earlier artifacts, nine candidate scores, 15 final prediction files and 21 image/head checks.',flush=True)


if __name__=='__main__': main()
