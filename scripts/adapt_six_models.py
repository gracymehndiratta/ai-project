"""Requested local-model adaptation, with an untouched locally evaluated paper reference."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def main(allow_test_fitting=False):
    from src.utils.research_integrity import require_exploratory_opt_in
    require_exploratory_opt_in(allow_test_fitting)
    os.chdir(ROOT)
    import copy
    import json
    from datetime import datetime,timezone
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    import xgboost as xgb
    from src.data.full_isic import CLASS_NAMES
    from src.models.feature_models import load_classifier
    from src.models.panderm import sha256_file
    from src.utils.feature_training import score_predictions
    from src.utils.tuned_training import class_weights
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path('outputs/diagnostics/six_model_adaptation');out.mkdir(parents=True,exist_ok=True)
    paths={
        'large_linear':('outputs/feature_models/panderm_large_comparison/linear','PanDerm Large / Linear head'),
        'large_nn128':('outputs/feature_models/panderm_large_capacity/winner','PanDerm Large / Neural 128 · existing 5 updates'),
        'large_nn192':('outputs/feature_models/panderm_large_refined/mlp','PanDerm Large / Neural 192–64'),
        'large_xgb':('outputs/feature_models/panderm_large_refined/xgboost','PanDerm Large / XGBoost + LDA'),
        'base_linear':('outputs/feature_models/panderm_base_comparison/linear','PanDerm Base / Linear head'),
        'base_xgb':('outputs/feature_models/panderm_base_comparison/xgboost','PanDerm Base / XGBoost')}
    paper_source=Path('outputs/feature_models/panderm_large_comparison/paper')
    hashes_path=out/'previous_artifact_hashes.json'
    if not hashes_path.exists():
        old=[p for p in Path('outputs/feature_models').rglob('*') if p.is_file()]
        old.extend(p for p in Path('outputs/diagnostics').glob('panderm_test_adaptation*/*') if p.is_file())
        old.extend(p for p in Path('outputs/diagnostics').glob('panderm_test_adaptation*/adapted/*') if p.is_file())
        hashes_path.write_text(json.dumps({str(p):sha256_file(p) for p in old},indent=2))
    frame=pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv')
    labels=frame.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    adaptation=np.flatnonzero(frame.split.isin(['test','official_test']))
    assert len(adaptation)==2744 and frame.iloc[adaptation].image_id.is_unique
    split_rows={split:np.flatnonzero(frame.split==split) for split in ['train','validation','test','official_validation','official_test']}
    split_rows['combined_adaptation']=adaptation
    caches={};cache_paths={'panderm_large':'data/features/panderm_large_72dfc47de5e585b2.npy',
        'panderm_base':'data/features/panderm_base_9e9b688f78eb850d.npy'}
    for backbone,path in cache_paths.items():
        cache=Path(path);receipt=json.loads(cache.with_suffix('.json').read_text())
        expected=receipt.get('embeddings_sha256')
        if expected: assert sha256_file(cache)==expected
        values=np.load(cache,mmap_mode='r');assert values.shape==(11720,1024 if backbone=='panderm_large' else 768)
        pd.testing.assert_series_equal(pd.read_csv(cache.with_name(cache.stem+'_manifest.csv')).image_id,frame.image_id)
        caches[backbone]=values
    plan={'models':paths,'iterations':4,'neural_original_lr':.0007,'linear_assumed_lr':.0007,'lr_divisor':3,
        'adaptation_images':2744,'adaptation_splits':['released HAM test1232','official test1512'],
        'reuse':'128-unit head already has five test-data updates, latest lr=.00028 (2.5x lower than original .0007); reuse without additional training.',
        'neural_iteration':'One full-data gradient accumulation, one AdamW step. Batch128; class weights from original training split; smoothing .03; decay .02; clip5.',
        'linear_iteration':'Copy fitted logistic coefficients into an equivalent Torch linear layer; same gradient-update method. Estimated starting lr .0007 because original solver has no SGD learning rate.',
        'xgb_iteration':'Append exactly one boosting round at existing learning_rate/3; retain feature projection and old trees. This is not numerically equivalent to a neural optimizer update.',
        'paper_reference':'Read-only PanDerm Large + authors exact probe recipe, locally evaluated; do not adapt.',
        'selection':'Four updates are fixed before evaluation; no checkpoint selected or rolled back using the resulting scores.',
        'evaluation':'Former test sets are now fitting data for our six copies. Reference has no test-data adaptation. Display exposure differences instead of claiming a fair held-out win.'}
    protocol=out/'protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text(encoding='utf-8'))==json.loads(json.dumps(plan))
    else: protocol.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
    frame.to_csv(out/'split_manifest.csv',index=False)
    frame.iloc[adaptation].assign(role='additional_fit').to_csv(out/'adaptation_manifest.csv',index=False)
    records=[];registry=[]
    history_path=out/'adaptation_history.csv'
    history=pd.read_csv(history_path).to_dict('records') if history_path.exists() else []

    def evaluate(wrapped,features,destination):
        destination.mkdir(parents=True,exist_ok=True);metrics={}
        for split,indices in split_rows.items():
            probability=wrapped.predict_proba(np.array(features[indices]))
            metric=score_predictions(labels[indices],probability);metrics[split]=metric
            saved=pd.DataFrame(probability,columns=CLASS_NAMES)
            saved.insert(0,'true_label',frame.iloc[indices].label.to_numpy());saved.insert(0,'image_id',frame.iloc[indices].image_id.to_numpy())
            saved.to_csv(destination/f'{split}_predictions.csv',index=False)
        (destination/'metrics.json').write_text(json.dumps(metrics,indent=2))
        return metrics

    for model_id,(path,name) in paths.items():
        source=Path(path);base,parent_scaler,info=load_classifier(source)
        folder=out/model_id;folder.mkdir(exist_ok=True)
        frame.to_csv(folder/'split_manifest.csv',index=False);joblib.dump(parent_scaler,folder/'scaler.joblib')
        features=caches[info['backbone']]
        source_frame=pd.read_csv(source.parent/'split_manifest.csv')
        original_train=source_frame[source_frame.split=='train']
        training_ids=set(original_train.image_id)
        original_y=original_train.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
        exposure={split:{'images':len(indices),'prior_training_overlap':len(set(frame.iloc[indices].image_id)&training_ids),
            'adaptation_overlap':len(set(frame.iloc[indices].image_id)&set(frame.iloc[adaptation].image_id))} for split,indices in split_rows.items()}
        before=evaluate(base,features,folder/'before')
        final_folder=folder/('step5_reused' if model_id=='large_nn128' else 'step4')
        if not (final_folder/'receipt.json').exists():
            history=[row for row in history if row['model_id']!=model_id]
            adapted=copy.deepcopy(base)
            xx=adapted.projection.transform(np.array(features[adaptation])).astype(np.float32)
            y=labels[adaptation]
            power=info.get('details',{}).get('weight_power',info.get('details',{}).get('power',.5))
            weights=class_weights(original_y,power)
            if model_id=='large_nn128':
                reused=Path('outputs/diagnostics/panderm_test_adaptation_step5')
                adapted,_,adapted_info=load_classifier(reused/'adapted')
                final_folder.mkdir(exist_ok=True)
                joblib.dump(adapted,final_folder/'model.joblib')
                receipt={'new_iterations':0,'previous_iterations':5,'learning_rate':.00028,
                    'reused_checkpoint':str(reused/'adapted/model.joblib'),'reused_sha256':sha256_file(reused/'adapted/model.joblib')}
                metadata={**adapted_info,'display_name':name,'comparison_origin':'our adapted model'}
            elif info['kind']=='xgboost':
                clf=adapted.models[0];booster=clf.get_booster().copy();initial_rounds=booster.num_boosted_rounds()
                rate=float(clf.get_params().get('learning_rate') or .04)/3
                params={k:v for k,v in clf.get_xgb_params().items() if v is not None and not callable(v)}
                params.update(learning_rate=rate,device='cuda' if torch.cuda.is_available() else 'cpu',disable_default_eval_metric=True)
                dtrain=xgb.DMatrix(xx,label=y,weight=weights[y])
                for iteration in range(1,5):
                    booster=xgb.train(params,dtrain,num_boost_round=1,xgb_model=booster,verbose_eval=False)
                    booster.set_attr(best_iteration=None,best_score=None)
                    assert booster.num_boosted_rounds()==initial_rounds+iteration
                    clf._Booster=booster;clf.set_params(n_estimators=booster.num_boosted_rounds(),callbacks=None,device='cpu')
                    step_folder=folder/f'step{iteration}';step_folder.mkdir(exist_ok=True)
                    joblib.dump(adapted,step_folder/'model.joblib')
                    booster.save_model(step_folder/'booster.ubj')
                    m=score_predictions(y,adapted.predict_proba(np.array(features[adaptation])))
                    history.append({'model_id':model_id,'iteration':iteration,'learning_rate':rate,'boosting_rounds':booster.num_boosted_rounds(),
                        'combined_fitting_accuracy':m['accuracy'],'combined_fitting_macro_f1':m['macro_f1']})
                receipt={'new_iterations':4,'previous_iterations':0,'original_learning_rate':rate*3,'learning_rate':rate,
                    'initial_rounds':initial_rounds,'final_rounds':booster.num_boosted_rounds(),'iteration_type':'additional boosting round',
                    'prediction_iteration_limit_removed':True}
                metadata={**info,'display_name':name,'comparison_origin':'our adapted model'}
            else:
                torch.manual_seed(2026)
                model=adapted.models[0]
                if info['kind']=='linear':
                    converted=nn.Linear(model.coef_.shape[1],7)
                    with torch.no_grad():
                        converted.weight.copy_(torch.tensor(model.coef_,dtype=torch.float32))
                        converted.bias.copy_(torch.tensor(model.intercept_,dtype=torch.float32))
                    with torch.inference_mode():
                        np.testing.assert_allclose(converted(torch.tensor(xx[:128])).softmax(1).numpy(),model.predict_proba(xx[:128]),atol=2e-6,rtol=2e-5)
                    adapted.models[0]=converted;model=converted
                device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
                tx,ty=torch.tensor(xx,device=device),torch.tensor(y,device=device)
                tw=torch.tensor(weights,dtype=torch.float32,device=device)
                loss_fn=nn.CrossEntropyLoss(weight=tw,label_smoothing=.03,reduction='sum');denominator=tw[ty].sum()
                rate=.0007/3;optimizer=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=.02)
                for iteration in range(1,5):
                    model.to(device).train();optimizer.zero_grad(set_to_none=True)
                    order=torch.randperm(len(y),device=device,generator=torch.Generator(device=device).manual_seed(2026+iteration))
                    for batch in order.split(128):
                        (loss_fn(model(tx[batch]),ty[batch])/denominator).backward()
                    gradient_norm=float(nn.utils.clip_grad_norm_(model.parameters(),5));optimizer.step()
                    assert {int(state['step']) for state in optimizer.state.values()}=={iteration}
                    model.cpu().eval()
                    step_folder=folder/f'step{iteration}';step_folder.mkdir(exist_ok=True)
                    joblib.dump(adapted,step_folder/'model.joblib')
                    torch.save(optimizer.state_dict(),step_folder/'optimizer_state.pt')
                    m=score_predictions(y,adapted.predict_proba(np.array(features[adaptation])))
                    history.append({'model_id':model_id,'iteration':iteration,'learning_rate':rate,'gradient_norm':gradient_norm,
                        'combined_fitting_accuracy':m['accuracy'],'combined_fitting_macro_f1':m['macro_f1']})
                    # CPU inference does not migrate optimizer moments. Restore model to optimizer device before next update.
                    model.to(device)
                model.cpu().eval();joblib.dump(adapted,final_folder/'model.joblib')
                receipt={'new_iterations':4,'previous_iterations':0,'original_learning_rate':.0007,'learning_rate':rate,
                    'original_rate_estimated':info['kind']=='linear','iteration_type':'one accumulated full-data AdamW update',
                    'optimizer':'Fresh optimizer for this adaptation, moments continued through four updates','microbatches_per_update':22}
                metadata={**info,'display_name':name,'comparison_origin':'our adapted model'}
            metadata.update(format_version=1,artifact_type='tuned_ensemble',training_variant='six_model_test_adaptation',
                data_contaminated_for_test_evaluation=True,excluded_from_clean_model_ranking=True,
                adaptation_images=2744,adaptation_details=receipt,exposure=exposure)
            (final_folder/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
            receipt.update(source_model_sha256=sha256_file(source/'model.joblib'),model_sha256=sha256_file(final_folder/'model.joblib'),
                created_utc=datetime.now(timezone.utc).isoformat())
            (final_folder/'receipt.json').write_text(json.dumps(receipt,indent=2))
        adapted=joblib.load(final_folder/'model.joblib')
        after=evaluate(adapted,features,final_folder)
        for stage,metrics in [('before',before),('adapted',after)]:
            for split,m in metrics.items():
                records.append({'model_id':model_id,'model':name,'stage':stage,'split':split,'images':m['images'],
                    **{key:m[key] for key in ['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']}})
        registry.append({'id':model_id,'name':name,'folder':str(final_folder),'source_folder':str(source),
            'before_metrics':str(folder/'before/metrics.json'),'after_metrics':str(final_folder/'metrics.json'),
            'exposure':exposure,'neural':info['kind']=='mlp'})
        print(f'COMPLETED {name}: combined fitting F1={after["combined_adaptation"]["macro_f1"]:.4f}; official images F1={after["official_test"]["macro_f1"]:.4f}',flush=True)
    assert len(registry)==6 and sum(row['neural'] for row in registry)==2
    # Unchanged research reference: no optimizer, no .fit, just fresh inference of its verified existing author-recipe checkpoint.
    reference,_,info=load_classifier(paper_source)
    reference_hash=sha256_file(paper_source/'model.joblib')
    reference_metrics=evaluate(reference,caches['panderm_large'],out/'research_panderm')
    assert reference_hash==sha256_file(paper_source/'model.joblib')
    paper={'paper_id':'panderm2025','name':'PanDerm Large / Authors’ recipe — local reproduction',
        'source_folder':str(paper_source),'metrics_path':str(out/'research_panderm/metrics.json'),
        'model_sha256':reference_hash,'adapted_to_test_data':False,'status':'Evaluated locally without further training',
        'provenance':'Official pretrained Large encoder + unchanged authors’ trained linear-probe recipe; local reproduction, not a claim of exact manuscript-checkpoint equivalence.'}
    (out/'research_reference.json').write_text(json.dumps(paper,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(records).to_csv(out/'comparison.csv',index=False);pd.DataFrame(history).to_csv(out/'adaptation_history.csv',index=False)
    old=json.loads(hashes_path.read_text());assert all(sha256_file(path)==digest for path,digest in old.items())
    (out/'active_models.json').write_text(json.dumps(registry,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'verification.json').write_text(json.dumps({'passed':True,'selected_models':6,'neural_heads':2,'previous_files_unchanged':len(old),
        'research_reference_unchanged':True,'adaptation_images':2744,'new_iterations_per_unadapted_model':4,
        'existing_128_checkpoint_reused':True},indent=2))
    print(pd.DataFrame(records).query("stage=='adapted' and split=='official_test'").to_string(index=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-test-fitting', action='store_true', help='Historical fitting only; invalidates independent test claims')
    main(allow_test_fitting=parser.parse_args().allow_test_fitting)
