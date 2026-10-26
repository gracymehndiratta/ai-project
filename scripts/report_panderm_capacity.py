"""Audit the bounded search, historical leaders, conditional uncertainty and plot."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def interval(frame,truth,first,second):
    """Paired lesion bootstrap, fixed fitted models; not training uncertainty."""
    import numpy as np
    groups,ids=np.unique(frame.lesion_id.astype(str),return_inverse=True)
    confusion=[]
    for probability in [first,second]:
        matrices=np.zeros((len(groups),49),dtype=np.int64)
        np.add.at(matrices,(ids,truth*7+probability.argmax(1)),1)
        confusion.append(matrices)
    rng=np.random.default_rng(2026)
    counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
    scores=[]
    for matrices in confusion:
        values=(counts@matrices).reshape(-1,7,7)
        numerator=2*np.diagonal(values,axis1=1,axis2=2)
        denominator=values.sum(1)+values.sum(2)
        scores.append(np.divide(numerator,denominator,out=np.zeros(numerator.shape,dtype=float),where=denominator!=0).mean(1))
    delta=scores[0]-scores[1]
    return {'samples':2000,'groups':len(groups),'delta_macro_f1_ci95':np.quantile(delta,[.025,.975]).tolist(),
        'limits':'Conditional on chosen fitted models; excludes training/model-selection uncertainty. Reused test data, exploratory comparison.'}


def main():
    os.chdir(ROOT)
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from threadpoolctl import threadpool_limits
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import ImageFeaturePredictor,load_classifier
    from src.models.panderm import sha256_file
    from src.models.panderm_capacity import ResidualHead
    from src.utils.feature_training import score_predictions
    from src.utils.panderm_capacity import OUTPUT,CACHE
    torch.set_num_threads(4)
    frame=pd.read_csv(OUTPUT/'split_manifest.csv');folds=pd.read_csv(OUTPUT/'training_folds.csv')
    pd.testing.assert_frame_equal(frame,pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv'))
    pd.testing.assert_frame_equal(folds,pd.read_csv('outputs/feature_models/panderm_large_refined/training_folds.csv'))
    train=np.flatnonzero(frame.split=='train');features=np.load(CACHE,mmap_mode='r')
    truth=frame.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    assert folds.image_id.tolist()==frame.iloc[train].image_id.tolist()
    assert folds.groupby('lesion_id').cv_fold.nunique().eq(1).all()
    previous=json.loads((OUTPUT/'previous_artifact_hashes.json').read_text())
    assert all(sha256_file(path)==digest for path,digest in previous.items())
    control=np.load(OUTPUT/'linear_control_oof.npy');control_metric=score_predictions(truth[train],control)
    expected=json.loads((OUTPUT/'linear_control.json').read_text())
    np.testing.assert_allclose(control_metric['macro_f1'],expected['oof_macro_f1'],atol=1e-12)
    trials=[];search=[]
    for path in sorted((OUTPUT/'trials').glob('*.json')):
        trial=json.loads(path.read_text());trials.append(trial)
        probability=np.load(path.with_name(path.stem+'_oof.npy'))
        metric=score_predictions(truth[train],probability)
        np.testing.assert_allclose(metric['macro_f1'],trial['oof_macro_f1'],atol=1e-12)
        for detail in trial['folds']:
            held=folds.cv_fold.to_numpy()==detail['fold']
            measured=score_predictions(truth[train][held],probability[held])
            np.testing.assert_allclose(measured['macro_f1'],detail['holdout_macro_f1'],atol=1e-12)
            assert detail['observations']>=50
        config=trial['config'];widths=config['widths']
        params=sum((a+1)*b for a,b in zip([1024,*widths],[*widths,7]))+(7175 if config['family']=='residual' else 0)
        search.append({'family':config['family'],'widths':'-'.join(map(str,widths)),'scale':config['scale'],
            'parameters':params,'tuning_macro_f1':trial['oof_macro_f1']*100,'median_epoch':trial['median_budget']})
    assert len(trials)==10
    folder=OUTPUT/'winner';selection=json.loads((folder/'selection.json').read_text())
    assert abs(selection['oof_macro_f1']-max(control_metric['macro_f1'],max(t['oof_macro_f1'] for t in trials)))<1e-12
    model,_,info=load_classifier(folder)
    marker_path=OUTPUT/'selection_complete.json';marker=json.loads(marker_path.read_text())
    assert marker['model_sha256']==sha256_file(folder/'model.joblib')
    assert info['backbone']=='panderm_large' and info['feature_dimension']==1024
    assert model.projection.n_samples_seen_==8207
    np.testing.assert_allclose(model.projection.mean_,features[train].mean(0,dtype=np.float64),atol=1e-10)
    old_linear=joblib.load('outputs/feature_models/panderm_large_comparison/linear/model.joblib')
    if isinstance(model.models[0],ResidualHead):
        net=model.models[0]
        assert not any(p.requires_grad for p in net.linear.parameters())
        np.testing.assert_allclose(net.linear.weight.detach().numpy(),old_linear.models[0].coef_.astype(np.float32),rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(net.linear.bias.detach().numpy(),old_linear.models[0].intercept_.astype(np.float32),rtol=1e-5,atol=1e-5)
    metric_names=['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']
    rows=[];uncertainty={};metrics=json.loads((folder/'metrics.json').read_text())
    for split in ['train','validation','test','official_validation','official_test']:
        indices=np.flatnonzero(frame.split==split);path=folder/f'{split}_predictions.csv';saved=pd.read_csv(path)
        assert saved.image_id.tolist()==frame.iloc[indices].image_id.tolist()
        assert saved.true_label.tolist()==frame.iloc[indices].label.tolist()
        if split in ['test','official_test']: assert marker_path.stat().st_mtime_ns<=path.stat().st_mtime_ns
        probability=model.predict_proba(np.array(features[indices]))
        np.testing.assert_allclose(probability,saved[CLASS_NAMES].to_numpy(),atol=1e-7)
        metric=score_predictions(truth[indices],saved[CLASS_NAMES].to_numpy())
        for key in metric_names: np.testing.assert_allclose(metric[key],metrics[split][key],atol=1e-12)
        rows.append({'model':'Best new capacity candidate','split':split,**{key:metric[key] for key in metric_names}})
        if split in ['test','official_test']:
            with threadpool_limits(limits=4): uncertainty[split]=interval(frame.iloc[indices],truth[indices],probability,old_linear.predict_proba(np.array(features[indices])))
    (OUTPUT/'paired_bootstrap.json').write_text(json.dumps(uncertainty,indent=2))
    historical=pd.read_csv('outputs/feature_models/panderm_large_widths/comparison_all_nine.csv')
    combined=pd.concat([historical,pd.DataFrame(rows)],ignore_index=True)
    combined.to_csv(OUTPUT/'comparison_all_ten.csv',index=False)
    wide=combined.pivot(index='model',columns='split',values=metric_names);wide.columns=[f'{split}_{metric}' for metric,split in wide.columns]
    wide.reset_index().to_csv(OUTPUT/'comparison.csv',index=False)
    # Matched training folds for existing neural architectures; no old validation scores mixed into this ranking.
    matched=[{'family':'linear control','widths':'none','parameters':7175,'tuning_macro_f1':control_metric['macro_f1']*100}]
    matched.extend(search)
    old_192=json.loads(Path('outputs/feature_models/panderm_large_refined/mlp/selection.json').read_text())
    matched.append({'family':'old two-hidden MLP','widths':'192-64','parameters':210119,'tuning_macro_f1':old_192['oof_macro_f1']*100})
    for name,widths,params in [('half','96-32',101991),('quarter','48-16',50231),('tiny','8-4',8295)]:
        selected=json.loads(Path(f'outputs/feature_models/panderm_large_widths/{name}/selection.json').read_text())
        matched.append({'family':'old two-hidden MLP','widths':widths,'parameters':params,'tuning_macro_f1':selected['oof_macro_f1']*100})
    ranked=pd.DataFrame(matched).sort_values('tuning_macro_f1',ascending=False)
    ranked.to_csv(OUTPUT/'matched_fold_ranking.csv',index=False)
    best_known=ranked.iloc[0]
    best_known_details={key:(None if pd.isna(value) else value) for key,value in best_known.to_dict().items()}
    leaders=[]
    for split in ['test','official_test']:
        for metric in ['accuracy','macro_f1','macro_f05','weighted_f1']:
            row=historical[historical.split==split].sort_values(metric,ascending=False).iloc[0]
            leaders.append({'split':split,'metric':metric,'previous_leader':row.model,'score_percent':row[metric]*100})
    pd.DataFrame(leaders).to_csv(OUTPUT/'previous_leaders.csv',index=False)
    predictor=ImageFeaturePredictor(folder,'cuda' if torch.cuda.is_available() else 'cpu')
    assert all(not p.requires_grad for p in predictor.extractor.parameters())
    checks=[]
    for index,row in frame[frame.split=='test'].groupby('label',sort=True).head(1).iterrows():
        with Image.open(row.image_path) as image: batch=predictor.transform(image.convert('RGB')).unsqueeze(0).to(predictor.device)
        with torch.inference_mode(): actual=predictor.extractor(batch).cpu().numpy()
        np.testing.assert_allclose(actual[0],features[index],rtol=1e-4,atol=1e-4)
        direct=model.predict_proba(actual);cached=model.predict_proba(np.array(features[index:index+1]))
        np.testing.assert_allclose(direct,cached,atol=1e-4,rtol=1e-3);assert direct.argmax()==cached.argmax()
        checks.append({'image_id':row.image_id,'max_probability_delta':float(abs(direct-cached).max())})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.5))
    for family,color in [('residual','#2474b5'),('plain','#d66b22'),('old two-hidden MLP','#38844e')]:
        points=ranked[ranked.family==family].sort_values('parameters')
        if family=='residual': points=points.sort_values('tuning_macro_f1',ascending=False).drop_duplicates('parameters').sort_values('parameters')
        ax.plot(points.parameters,points.tuning_macro_f1,'o-',label=family,color=color)
        for _,point in points.iterrows(): ax.annotate(point.widths,(point.parameters,point.tuning_macro_f1),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.axhline(control_metric['macro_f1']*100,color='#555555',linestyle='--',label='Linear control')
    ax.set_xscale('log');ax.set_xlabel('Total classifier parameters (log scale)');ax.set_ylabel('Training-fold tuning macro F1 (%)')
    ax.set_title('Observed architecture search — no single universal size optimum established')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUTPUT/'capacity_search.png',dpi=160);plt.close(fig)
    def percentages(split):
        columns=['model','accuracy','macro_f1','macro_f05','weighted_f1']
        data=combined[combined.split==split][columns].copy();data[columns[1:]]*=100
        return table(data)
    text=f'''# Searching for a useful amount of nonlinear capacity

A linear softmax head is a neural classifier with no hidden layers. Generalization need not have one smooth maximum as a function of parameter count: architecture, regularization, optimization, distribution and metric matter. The previous 8/4 bottleneck had more parameters but did not preserve the linear function. These residual candidates explicitly retain the fitted linear logits and start from zero nonlinear correction.

## Previous leaders, checked from saved results

{table(pd.DataFrame(leaders))}

## Bounded training-fold search

{table(pd.DataFrame(search))}

Eight residual settings cover hidden widths 2/2, 8/4, 32/8 and 64/16 with correction scales .25/1. Two single-hidden GELU heads cover widths 32/128. All use fixed AdamW lr .0007, weight decay .02, dropout .2, class-weight power .5, smoothing .03 and batch128. Three lesion-separated folds inside the same 8,207 training images fit their own scalers and linear controls (C=.01, power .5). Up to 160 epochs, minimum50 observed, patience40; zero correction at epoch0 is a valid residual checkpoint. Fold held-out scores choose checkpoints and configurations, so OOF is tuning evidence, not an unbiased estimate.

The best **new** candidate is `{json.dumps(selection['config'],sort_keys=True)}`, with {selection['oof_macro_f1']*100:.2f}% tuning macro F1 versus {control_metric['macro_f1']*100:.2f}% for linear control. Total head parameters: {info['parameters']:,}; parameters optimized in the nonlinear stage: {info['nonlinear_training_parameters']:,}. The selected budget is {selection['median_budget']} epochs, followed by fitting all 8,207 rows. The residual base is fitted first and frozen, rather than being updated with the nonlinear optimizer.

Across the matched-fold candidates including previously saved neural heads, the highest observed tuning score belongs to **{best_known['family']} {best_known['widths']}**, {best_known['tuning_macro_f1']:.2f}%. The 256/64 historical model is omitted from this OOF ranking because its earlier selection used a different validation procedure; its test results remain in the comparisons below. Different candidate counts, regularization and checkpoint selection introduce selection uncertainty. This is not proof of a unique or global optimum.

## HAM test, 1,232 images, percent

{percentages('test')}

## Official test, 1,512 images, percent

{percentages('official_test')}

The same released splits, images, encoder and preprocessing are retained. No demographics or SMOTE are introduced. All new selection is completed before validation/test reporting. Tests were previously inspected and motivated this new search, so these are exploratory reused comparisons, not independent confirmation. Do not pick the architecture by repeatedly maximizing these test results.

## Conditional uncertainty, new candidate minus linear macro F1

HAM 95% paired lesion-bootstrap interval: {uncertainty['test']['delta_macro_f1_ci95']}; official test: {uncertainty['official_test']['delta_macro_f1_ci95']}. Values are fractions (multiply by100 for percentage points). These intervals condition on fitted selected models, exclude training/search uncertainty, and do not establish a global optimum.

## Verification and saved artifacts

All {len(previous)} prior files retain their hashes. Ten candidate OOF scores and 30 fold scores, five final prediction files, training-only scaling, frozen baseline weights (for residual winner) and seven real image inference checks passed. A regression test confirms zero correction matches linear logits and nonlinear training leaves baseline coefficients untouched. `matched_fold_ranking.csv`, `previous_leaders.csv`, `comparison_all_ten.csv`, `paired_bootstrap.json`, `capacity_search.png`, per-lesion statistics and the winning checkpoint are saved alongside this report. `train-panderm-capacity.cmd` reproduces/resumes the bounded search.
'''
    (OUTPUT/'REPORT.md').write_text(text,encoding='utf-8')
    (OUTPUT/'verification.json').write_text(json.dumps({'passed':True,'previous_artifacts_unchanged':len(previous),'candidate_oof_scores_recomputed':10,
        'fold_scores_recomputed':30,'prediction_files_recomputed':5,'image_checks':checks,'best_known_matched_fold':best_known_details},indent=2,allow_nan=False))
    print('New winner:',selection['config'],flush=True);print('Best known matched-fold candidate:',best_known.to_dict(),flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print(json.dumps(uncertainty,indent=2),flush=True)


if __name__=='__main__': main()
