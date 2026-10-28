"""User-requested one-update adaptation on former test images; not test evaluation."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def main(step=1, allow_test_fitting=False):
    from src.utils.research_integrity import require_exploratory_opt_in
    require_exploratory_opt_in(allow_test_fitting)
    os.chdir(ROOT)
    from datetime import datetime,timezone
    import copy
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from src.data.full_isic import CLASS_NAMES
    from src.models.panderm import sha256_file
    from src.models.feature_models import load_classifier
    from src.utils.tuned_training import class_weights
    from src.utils.feature_training import score_predictions
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert 1<=step<=5
    learning_rate=.00007 if step==1 else .00028
    seed=2025+step
    def run_folder(number):
        return Path('outputs/diagnostics/panderm_test_adaptation_once') if number==1 else Path(f'outputs/diagnostics/panderm_test_adaptation_step{number}')
    previous_run=run_folder(max(1,step-1))
    out=run_folder(step)
    out.mkdir(parents=True,exist_ok=True)
    source=Path('outputs/feature_models/panderm_large_capacity/winner') if step==1 else previous_run/'adapted'
    original,identity,info=load_classifier(source)
    assert info['details']=={'family':'plain','widths':[128],'scale':None}
    hashes=out/'previous_artifact_hashes.json'
    if not hashes.exists():
        paths=[p for p in Path('outputs/feature_models').rglob('*') if p.is_file()]
        for earlier in range(1,step): paths.extend(p for p in run_folder(earlier).rglob('*') if p.is_file())
        hashes.write_text(json.dumps({str(p):sha256_file(p) for p in paths},indent=2))
    frame=pd.read_csv('outputs/feature_models/panderm_large_capacity/split_manifest.csv')
    cache=Path('data/features/panderm_large_72dfc47de5e585b2.npy')
    cache_info=json.loads(cache.with_suffix('.json').read_text())
    assert cache_info['complete'] and sha256_file(cache)==cache_info['embeddings_sha256']
    features=np.load(cache,mmap_mode='r');truth=frame.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    rows=np.flatnonzero(frame.split.isin(['test','official_test']))
    assert len(rows)==2744 and frame.iloc[rows].image_id.is_unique
    train=np.flatnonzero(frame.split=='train')
    assert len(train)==8207 and set(frame.iloc[train].image_id).isdisjoint(set(frame.iloc[rows].image_id))
    frame.iloc[rows].assign(data_role_after_update='additional_training_data').to_csv(out/'adaptation_manifest.csv',index=False)
    folder=out/'adapted';folder.mkdir(exist_ok=True)
    receipt_path=out/'update_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());assert receipt['source_model_sha256']==sha256_file(source/'model.joblib')
        adapted=joblib.load(folder/'model.joblib')
        assert sha256_file(folder/'model.joblib')==receipt['adapted_model_sha256']
    else:
        torch.manual_seed(seed)
        adapted=copy.deepcopy(original)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=adapted.models[0].to(device).train()
        assert sum(p.numel() for p in model.parameters())==132103
        xx=torch.tensor(adapted.projection.transform(np.array(features[rows])).astype(np.float32),device=device)
        yy=torch.tensor(truth[rows],device=device)
        # Preserve the source training objective's weights, rather than fitting new scaling/weighting on tests.
        weights=torch.tensor(class_weights(truth[train],.5),dtype=torch.float32,device=device)
        criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=.03,reduction='sum')
        denominator=weights[yy].sum()
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.02)
        if step>1:
            prior_state=torch.load(previous_run/'optimizer_state.pt',map_location='cpu',weights_only=True)
            assert {int(state['step']) for state in prior_state['state'].values()}=={step-1}
            optimizer.load_state_dict(prior_state)
            for group in optimizer.param_groups: group['lr']=learning_rate
        optimizer.zero_grad(set_to_none=True)
        order=torch.randperm(len(yy),device=device,generator=torch.Generator(device=device).manual_seed(seed))
        microbatches=0;total_loss=0.
        for batch in order.split(128):
            loss=criterion(model(xx[batch]),yy[batch])/denominator
            loss.backward();microbatches+=1;total_loss+=float(loss.detach())
        gradient_norm=float(nn.utils.clip_grad_norm_(model.parameters(),5))
        optimizer.step()  # Exactly one parameter update after the full combined set.
        steps=[int(state['step']) for state in optimizer.state.values()]
        assert steps and set(steps)=={step}
        optimizer_state=optimizer.state_dict()
        for state in optimizer_state['state'].values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor): state[key]=value.cpu()
        torch.save(optimizer_state,out/'optimizer_state.pt')
        model.cpu().eval()
        assert any(not torch.equal(value,original.models[0].state_dict()[key]) for key,value in model.state_dict().items())
        joblib.dump(adapted,folder/'model.joblib');joblib.dump(identity,out/'scaler.joblib')
        metadata={**info,'training_variant':'test_data_adaptation_diagnostic','data_contaminated_for_test_evaluation':True,
            'excluded_from_clean_model_ranking':True,'adaptation_images':2744,'optimizer_updates':step,
            'optimizer_updates_this_run':1,'learning_rate':learning_rate}
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
        receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'source_model_sha256':sha256_file(source/'model.joblib'),
            'adapted_model_sha256':sha256_file(folder/'model.joblib'),'optimizer_updates':1,'optimizer_state_steps':steps,
            'cumulative_optimizer_updates':step,'learning_rate':learning_rate,'previous_learning_rate':.0007 if step==1 else .00007 if step==2 else .00028,
            'images_seen':len(rows),'passes':1,'microbatches':microbatches,
            'gradient_accumulation':'All 2744 examples contribute to one global weighted-mean gradient; no optimizer step per microbatch.',
            'optimizer':('New AdamW optimizer; original training moments were not retained.' if step==1 else
                'AdamW moments and step counters restored from the immediately preceding adaptation checkpoint.'),
            'weight_decay':.02,'clip_norm':5,'seed':seed,
            'objective':'Original training-class square-root weights retained, label smoothing .03; train-mode dropout .2.',
            'stochastic_training_loss':total_loss,'gradient_norm_before_clip':gradient_norm,
            'backbone':'Frozen original PanDerm Large cache; no encoder training or re-extraction.',
            'scaling':'Original 8207-image fitted StandardScaler retained without refitting.',
            'evaluation_limit':'Former HAM/official test images are now training data for this copy. Scores on them are fitting diagnostics, not held-out performance.'}
        receipt_path.write_text(json.dumps(receipt,indent=2))
    # Independent loaded-checkpoint predictions and both original/relabelled evaluation roles.
    adapted=joblib.load(folder/'model.joblib');records=[];all_metrics={}
    subsets={split:np.flatnonzero(frame.split==split) for split in ['train','validation','test','official_validation','official_test']}
    subsets['combined_adaptation']=rows
    for name,indices in subsets.items():
        role=('additional_fit' if name in ['test','official_test','combined_adaptation'] else
              'original_fit' if name=='train' else 'validation_excluded_from_this_update')
        all_metrics[name]={}
        for stage,model in [('before',original),('after',adapted)]:
            probability=model.predict_proba(np.array(features[indices]))
            metric=score_predictions(truth[indices],probability);all_metrics[name][stage]=metric
            saved=pd.DataFrame(probability,columns=CLASS_NAMES)
            saved.insert(0,'true_label',frame.iloc[indices].label.to_numpy());saved.insert(0,'image_id',frame.iloc[indices].image_id.to_numpy())
            path=out/f'{name}_{stage}_predictions.csv';saved.to_csv(path,index=False)
            reread=pd.read_csv(path);verified=score_predictions(truth[indices],reread[CLASS_NAMES].to_numpy())
            for key in ['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05']:
                np.testing.assert_allclose(metric[key],verified[key],atol=1e-12)
            records.append({'partition':name,'role_after_update':role,'stage':stage,'images':len(indices),
                **{key:metric[key] for key in ['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05']}})
    result=pd.DataFrame(records);result.to_csv(out/'comparison.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(all_metrics,indent=2))
    np.testing.assert_allclose(adapted.projection.mean_,original.projection.mean_,atol=0,rtol=0)
    assert adapted.projection.n_samples_seen_==8207
    previous=json.loads(hashes.read_text());assert all(sha256_file(path)==digest for path,digest in previous.items())
    optimizer_state=torch.load(out/'optimizer_state.pt',map_location='cpu',weights_only=True)
    assert {int(state['step']) for state in optimizer_state['state'].values()}=={step}
    assert all(group['lr']==learning_rate for group in optimizer_state['param_groups'])
    labels={'combined_adaptation':'Combined former tests: FITTING SCORES','test':'Former HAM test: FITTING SCORES',
        'official_test':'Former official test: FITTING SCORES','validation':'HAM validation: excluded from this update',
        'official_validation':'Official validation: excluded from this update'}
    sections=[]
    for name,title in labels.items():
        data=result[result.partition==name][['stage','accuracy','macro_f1','macro_f05']].copy();data[['accuracy','macro_f1','macro_f05']]*=100
        lines=['| Stage | Accuracy | Macro F1 | Macro F0.5 |','| --- | --- | --- | --- |']
        lines.extend(f'| {row.stage} | {row.accuracy:.2f} | {row.macro_f1:.2f} | {row.macro_f05:.2f} |' for row in data.itertuples())
        sections.append('## '+title+'\n\n'+'\n'.join(lines))
    text=f'''# One-update adaptation on former test data: cumulative step {step}

User-requested diagnostic on a separate copy of the 1024 -> 128 GELU -> 7 head. PanDerm Large and the original train-only scaler remain frozen. Learning rate is {learning_rate:.8f}. Exactly one additional AdamW optimizer step follows gradient accumulation across all 2,744 combined former test images (22 microbatches), giving {step} cumulative adaptation updates. This is one update per run, not 22 minibatch updates or a multi-epoch run. There is no selection or rollback based on resulting scores. {'AdamW moments/counters are continued from the preceding adaptation checkpoint.' if step>1 else 'A new AdamW optimizer was used for the first adaptation.'}

The original model and every clean comparison artifact are preserved. This adapted copy is stored under diagnostics, excluded from the clean model selector/ranking. Former test images now count as additional training data for this copy. Before/after scores on them cannot estimate held-out generalization. Validation images were excluded from this update but were already inspected in previous work, so those comparisons are not fresh independent validation either.

All values below are percentages.\n\n'''+ '\n\n'.join(sections)+f'\n\nVerified all {len(previous)} prior model artifacts unchanged, all 12 saved prediction files, unchanged scaling and AdamW state steps exactly equal to1. Full objective, seed and gradient-accumulation details are in update_receipt.json.\n'
    text=text.replace('exactly equal to1',f'exactly equal to {step}')
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    (out/'verification.json').write_text(json.dumps({'passed':True,'optimizer_updates_this_run':1,'cumulative_optimizer_updates':step,'previous_artifacts_unchanged':len(previous),
        'prediction_files_recomputed':12,'scaler_unchanged':True,'clean_ranking_excluded':True},indent=2))
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--step',type=int,choices=range(1,6),default=1)
    parser.add_argument('--allow-test-fitting', action='store_true', help='Historical fitting only; invalidates independent test claims')
    args=parser.parse_args()
    main(step=args.step, allow_test_fitting=args.allow_test_fitting)
