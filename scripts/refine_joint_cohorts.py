"""Small, explicitly training-exposed joint-cohort refinement; reference stays read-only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

KEYS=['accuracy','macro_f1','macro_f05','weighted_f1','balanced_accuracy']

def mean_five(metrics): return sum(metrics[key] for key in KEYS)/len(KEYS)

def main():
    import os,json,copy
    import joblib,numpy as np,pandas as pd,torch,xgboost as xgb
    from torch import nn
    from src.models.feature_models import load_classifier
    from src.models.panderm import sha256_file
    from src.utils.tuned_training import class_weights
    from src.utils.feature_training import score_predictions
    from src.data.full_isic import CLASS_NAMES
    os.chdir(ROOT);torch.set_num_threads(4);torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path('outputs/diagnostics/joint_refinement')
    if (root/'verification.json').exists():
        print('Completed refinement exists; no additional training performed.');return
    root.mkdir(exist_ok=True)
    old_root=Path('outputs/diagnostics/six_model_adaptation')
    sources=json.loads((old_root/'active_models.json').read_text(encoding='utf-8'))
    reference=json.loads((old_root/'research_reference.json').read_text(encoding='utf-8'))
    ref_metrics=json.loads(Path(reference['metrics_path']).read_text())
    reference_path=Path(reference['source_folder'])/'model.joblib'
    reference_hash=sha256_file(reference_path)
    assert reference_hash==reference['model_sha256']
    prior={str(path):sha256_file(path) for path in Path('outputs').rglob('*') if path.is_file() and path.suffix in ['.joblib','.pt','.ubj','.pth'] and root not in path.parents}
    (root/'previous_checkpoint_hashes.json').write_text(json.dumps(prior,indent=2))
    protocol={'max_additional_updates':8,'neural_linear_lr':.00014,'original_neural_lr':.0007,
        'xgb_lr':.04/9,'cohort_sample_weights':'Each cohort contributes 50% of total loss; within-cohort weights use original training class frequencies to power 0.25, normalized per cohort.',
        'loss':'Cross entropy, label smoothing .01; AdamW decay .01, clip norm5, retain original dropout and frozen projection.',
        'iteration':'One accumulated full-data gradient update, or one appended XGBoost round. These are different units.',
        'fit_counts':{'HAM':1232,'ISIC':1512},'composite_metrics':KEYS,'composite':'Unweighted arithmetic mean of the five displayed metrics, separately for each cohort.',
        'selection':'Maximize minimum composite improvement over the unchanged reference across HAM and ISIC. Step zero is eligible. Ties use mean improvement then earlier step. These are fitting-set selections.',
        'reference_training':'None; unchanged author-recipe reproduction.',
        'claim':'Results describe additional fitting on both former tests; they cannot establish independent superiority.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    frame=pd.read_csv(old_root/'split_manifest.csv')
    cache=Path('data/features/panderm_large_72dfc47de5e585b2.npy');features=np.load(cache,mmap_mode='r')
    pd.testing.assert_series_equal(frame.image_id,pd.read_csv(cache.with_name(cache.stem+'_manifest.csv')).image_id)
    labels=frame.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    indices={split:np.flatnonzero(frame.split==split) for split in ['train','validation','test','official_validation','official_test']}
    fit=np.flatnonzero(frame.split.isin(['test','official_test']));indices['combined_adaptation']=fit
    assert len(indices['test'])==1232 and len(indices['official_test'])==1512 and len(fit)==2744
    assert set(frame.iloc[fit].image_id).isdisjoint(set(frame[frame.split.isin(['validation','official_validation'])].image_id))
    frame.iloc[fit].assign(role='additional_fit').to_csv(root/'fit_manifest.csv',index=False)
    history=[];selected=[]
    for item in sources:
        if item['id'] not in ['large_linear','large_nn128','large_nn192','large_xgb']:continue
        model_id=item['id'];source=Path(item['folder']);base,scaler,info=load_classifier(source)
        wrapped=copy.deepcopy(base);folder=root/model_id;folder.mkdir(exist_ok=True)
        joblib.dump(scaler,folder/'scaler.joblib');frame.to_csv(folder/'split_manifest.csv',index=False)
        x=wrapped.projection.transform(np.array(features[fit])).astype(np.float32);y=labels[fit]
        orig=pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv')
        train_y=orig[orig.split=='train'].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
        cw=class_weights(train_y,.25);sample=cw[y].astype(np.float32)
        fit_splits=frame.iloc[fit].split.to_numpy()
        for split in ['test','official_test']:
            mask=fit_splits==split;sample[mask]/=sample[mask].sum();sample[mask]*=len(fit)/2
        assert np.allclose([sample[fit_splits==split].sum() for split in ['test','official_test']],len(fit)/2,rtol=1e-5)
        def evaluate(destination,full=False):
            destination.mkdir(exist_ok=True);metrics={}
            for split in (indices if full else ['test','official_test']):
                rows=indices[split];probability=wrapped.predict_proba(np.array(features[rows]))
                metrics[split]=score_predictions(labels[rows],probability)
                table=pd.DataFrame(probability,columns=CLASS_NAMES)
                table.insert(0,'true_label',frame.iloc[rows].label.to_numpy());table.insert(0,'image_id',frame.iloc[rows].image_id.to_numpy())
                table.to_csv(destination/f'{split}_predictions.csv',index=False)
            (destination/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
        def record(step,metrics):
            gains={s:mean_five(metrics[s])-mean_five(ref_metrics[s]) for s in ['test','official_test']}
            history.append({'model_id':model_id,'additional_update':step,'HAM_mean_five':mean_five(metrics['test']),
                'ISIC_mean_five':mean_five(metrics['official_test']),'HAM_gain':gains['test'],'ISIC_gain':gains['official_test'],
                **{s+'_'+k:metrics[s][k] for s in gains for k in KEYS}})
            return (min(gains.values()),sum(gains.values())/2,-step)
        baseline=evaluate(folder/'step0');best=(record(0,baseline),0)
        joblib.dump(wrapped,folder/'step0/model.joblib')
        kind=info['kind'];device='cuda' if torch.cuda.is_available() else 'cpu'
        if kind=='xgboost':
            clf=wrapped.models[0];booster=clf.get_booster().copy();initial_rounds=booster.num_boosted_rounds()
            params={k:v for k,v in clf.get_xgb_params().items() if v is not None and not callable(v)}
            params.update(learning_rate=.04/9,device=device,disable_default_eval_metric=True)
            data=xgb.DMatrix(x,label=y,weight=sample)
        else:
            model=wrapped.models[0];assert isinstance(model,nn.Module)
            model.to(device);tx=torch.tensor(x,device=device);ty=torch.tensor(y,device=device);tw=torch.tensor(sample,device=device)
            loss_fn=nn.CrossEntropyLoss(label_smoothing=.01,reduction='none')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.00014,weight_decay=.01)
        for step in range(1,9):
            if kind=='xgboost':
                booster=xgb.train(params,data,num_boost_round=1,xgb_model=booster,verbose_eval=False)
                booster.set_attr(best_iteration=None,best_score=None)
                assert booster.num_boosted_rounds()==initial_rounds+step
                clf._Booster=booster;clf.set_params(n_estimators=booster.num_boosted_rounds(),device='cpu',callbacks=None)
            else:
                torch.manual_seed(2026+step);model.to(device).train();optimizer.zero_grad(set_to_none=True)
                order=torch.randperm(len(y),device=device)
                for batch in order.split(128): (loss_fn(model(tx[batch]),ty[batch])*tw[batch]).sum().div(tw.sum()).backward()
                nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                assert {int(v['step']) for v in optimizer.state.values()}=={step}
                model.cpu().eval()
            step_folder=folder/f'step{step}';metrics=evaluate(step_folder)
            joblib.dump(wrapped,step_folder/'model.joblib')
            if kind!='xgboost':torch.save(optimizer.state_dict(),step_folder/'optimizer_state.pt')
            rank=record(step,metrics)
            if rank>best[0]:best=(rank,step)
            meta={**info,'comparison_family_id':model_id,'checkpoint_label':f'joint_refinement/{model_id}/step{step}',
                'training_variant':'joint_refinement','benchmark_eligible':False,'preferred_for_benchmark':False,
                'additional_updates':step,'learning_rate':.04/9 if kind=='xgboost' else .00014}
            (step_folder/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
            print(f'{model_id} update {step}: HAM {mean_five(metrics["test"])*100:.3f}; ISIC {mean_five(metrics["official_test"])*100:.3f}; minimum gain {rank[0]*100:+.3f} pp',flush=True)
            if kind!='xgboost':model.to(device)
        wrapped=joblib.load(folder/f'step{best[1]}/model.joblib')
        target=folder/'selected';target.mkdir(exist_ok=True);joblib.dump(wrapped,target/'model.joblib')
        metrics=evaluate(target,full=True)
        receipt={'source_checkpoint':str(source/'model.joblib'),'source_sha256':sha256_file(source/'model.joblib'),
            'source_prior_updates':info.get('adaptation_details',{}).get('new_iterations',0)+info.get('adaptation_details',{}).get('previous_iterations',0),
            'additional_updates_attempted':8,'additional_updates_selected':best[1],'selection_rank':best[0],
            'learning_rate':.04/9 if kind=='xgboost' else .00014,'fit_images':2744,'fit_counts':{'HAM':1232,'ISIC':1512},
            'model_sha256':sha256_file(target/'model.joblib'),'source_projection_frozen':True,'reference_unchanged':True}
        metadata={**info,'training_variant':'joint_refinement','comparison_family_id':model_id,
            'checkpoint_label':f'joint_refinement/{model_id}/selected','benchmark_eligible':True,'preferred_for_benchmark':True,
            'adaptation_details':receipt,'data_contaminated_for_test_evaluation':True,'excluded_from_clean_model_ranking':True}
        (target/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        (target/'receipt.json').write_text(json.dumps(receipt,indent=2))
        selected.append({'model_id':model_id,'folder':str(target),'additional_update':best[1],
            'HAM_mean_five':mean_five(metrics['test']),'ISIC_mean_five':mean_five(metrics['official_test']),
            'beats_reference_on_both':all(mean_five(metrics[s])>mean_five(ref_metrics[s]) for s in ['test','official_test'])})
    pd.DataFrame(history).to_csv(root/'history.csv',index=False)
    (root/'selected_models.json').write_text(json.dumps(selected,indent=2))
    assert sha256_file(reference_path)==reference_hash
    assert all(sha256_file(path)==digest for path,digest in prior.items())
    verification={'passed':True,'fit_images':2744,'HAM_images':1232,'ISIC_images':1512,'reference_unchanged':True,
        'preserved_checkpoints':len(prior),'selected_models':selected,'models_beating_reference_on_both':sum(m['beats_reference_on_both'] for m in selected)}
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2),flush=True)

if __name__=='__main__': main()
