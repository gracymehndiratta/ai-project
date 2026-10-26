"""Bounded training-fold search around a linear classifier, not test-set search."""
from datetime import datetime,timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from threadpoolctl import threadpool_limits

from src.data.full_isic import CLASS_NAMES
from src.models.panderm import sha256_file
from src.models.panderm_capacity import ResidualHead,plain_head
from src.utils.feature_training import score_predictions
from src.utils.tuned_training import TunedClassifier,class_weights,macro_f1_cost

OUTPUT=Path('outputs/feature_models/panderm_large_capacity')
CACHE=Path('data/features/panderm_large_72dfc47de5e585b2.npy')


def linear(x,y):
    result=LogisticRegression(C=.01,class_weight=dict(enumerate(class_weights(y,.5))),max_iter=2000,random_state=42)
    with threadpool_limits(limits=4): result.fit(x,y)
    return result


def fit(x,y,base,config,seed,stop=None,epochs=None):
    torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=(ResidualHead(base.coef_,base.intercept_,config['widths'],config['scale'])
        if config['family']=='residual' else plain_head(config['widths'][0]))
    model.to(device)
    xx,yy=torch.tensor(x,device=device),torch.tensor(y,device=device)
    if stop is not None: vx,vy=torch.tensor(stop[0],device=device),stop[1]
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y,.5),dtype=torch.float32,device=device),label_smoothing=.03)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0007,weight_decay=.02)
    generator=torch.Generator(device=device).manual_seed(seed)
    history=[];best=-1.;best_epoch=0;state=None
    if stop is not None and config['family']=='residual':
        model.eval()
        with torch.inference_mode(): probability=model(vx).softmax(1).cpu().numpy()
        np.testing.assert_allclose(probability,base.predict_proba(stop[0]),atol=2e-6,rtol=2e-5)
        best=-macro_f1_cost(vy,probability)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({'epoch':0,'macro_f1':best})
    maximum=160 if epochs is None else epochs
    observed=0
    for epoch in range(1,maximum+1):
        observed=epoch;model.train()
        for batch in torch.randperm(len(y),device=device,generator=generator).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(xx[batch]),yy[batch]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if stop is not None:
            model.eval()
            with torch.inference_mode(): probability=model(vx).softmax(1).cpu().numpy()
            score=-macro_f1_cost(vy,probability)
            history.append({'epoch':epoch,'macro_f1':score})
            if score>best+1e-6:
                best,best_epoch=score,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch>=50 and epoch-best_epoch>=40: break
    model.cpu()
    if stop is not None: model.load_state_dict(state)
    model.eval()
    return model,{'best_budget':best_epoch if stop is not None else epochs,'observations':observed,'history':history}


def train():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    OUTPUT.mkdir(parents=True,exist_ok=True)
    hashes=OUTPUT/'previous_artifact_hashes.json'
    if not hashes.exists(): hashes.write_text(json.dumps({str(p):sha256_file(p) for p in OUTPUT.parent.rglob('*') if p.is_file() and OUTPUT not in p.parents},indent=2))
    frame=pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv')
    folds=pd.read_csv('outputs/feature_models/panderm_large_refined/training_folds.csv')
    receipt=json.loads(CACHE.with_suffix('.json').read_text())
    assert receipt['complete'] and sha256_file(CACHE)==receipt['embeddings_sha256']
    assert sha256_file('artifacts/panderm_paper/panderm_large.pth')==receipt['checkpoint_sha256']
    features=np.load(CACHE,mmap_mode='r');rows=np.flatnonzero(frame.split=='train')
    assert folds.image_id.tolist()==frame.iloc[rows].image_id.tolist()
    assert folds.groupby('lesion_id').cv_fold.nunique().eq(1).all()
    x=np.array(features[rows]);y=frame.iloc[rows].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    configs=[{'family':'residual','widths':widths,'scale':scale} for widths in [[2,2],[8,4],[32,8],[64,16]] for scale in [.25,1.]]
    configs += [{'family':'plain','widths':[width],'scale':None} for width in [32,128]]
    plan={'backbone':'panderm_large','checkpoint_sha256':receipt['checkpoint_sha256'],'candidates':configs,
        'selection':'Maximum pooled three-fold training OOF macro F1, including an explicitly refitted linear control. Fold scores also select stopping epochs: tuning evidence, not unbiased validation.',
        'linear_control':'Train-fold StandardScaler + L2 logistic C=.01, class power .5, max_iter2000, seed42. Same as current linear head.',
        'residual':'Frozen fitted linear logits plus scale * nonlinear correction. GELU/dropout.2; final correction layer initialized zero. Epoch0 is a valid checkpoint, so a fold can retain the linear control.',
        'optimizer':{'name':'AdamW','lr':.0007,'weight_decay':.02,'smoothing':.03,'class_power':.5,'batch_size':128},
        'stopping':{'maximum':160,'minimum_observed':50,'patience':40},
        'final_fit':'All8207 training images. Median best fold epoch. Freeze winner before reporting all validation/test scores.',
        'test_limit':'Architecture search motivated by previously inspected results; reused tests are exploratory comparisons, not independent confirmation. No new test result selects the candidate.',
        'maximum_limit':'Best in this bounded search; neither a unique peak nor a global neural-network optimum is established.'}
    path=OUTPUT/'experiment.json'
    if path.exists(): assert json.loads(path.read_text())==plan
    else: path.write_text(json.dumps(plan,indent=2))
    frame.to_csv(OUTPUT/'split_manifest.csv',index=False);folds.to_csv(OUTPUT/'training_folds.csv',index=False)
    partitions=[];control=np.zeros((len(y),7))
    for fold in range(3):
        held=np.flatnonzero(folds.cv_fold.to_numpy()==fold);fitting=np.flatnonzero(folds.cv_fold.to_numpy()!=fold)
        scaler=StandardScaler().fit(x[fitting]);xx=scaler.transform(x[fitting]).astype(np.float32);vx=scaler.transform(x[held]).astype(np.float32)
        base=linear(xx,y[fitting]);control[held]=base.predict_proba(vx)
        partitions.append((fitting,held,xx,vx,base))
    control_metric=score_predictions(y,control)
    np.save(OUTPUT/'linear_control_oof.npy',control)
    (OUTPUT/'linear_control.json').write_text(json.dumps({'oof_macro_f1':control_metric['macro_f1'],'oof_accuracy':control_metric['accuracy']},indent=2))
    trials=OUTPUT/'trials';trials.mkdir(exist_ok=True);results=[]
    for index,config in enumerate(configs):
        path=trials/f'{index:02d}.json'
        if path.exists():
            result=json.loads(path.read_text());assert result['config']==config;results.append(result);continue
        probability=np.zeros((len(y),7));details=[]
        for fold,(fitting,held,xx,vx,base) in enumerate(partitions):
            model,detail=fit(xx,y[fitting],base,config,42+fold,stop=(vx,y[held]))
            wrapped=TunedClassifier([model],FunctionTransformer(validate=False))
            probability[held]=wrapped.predict_proba(vx)
            fit_metric=score_predictions(y[fitting],wrapped.predict_proba(xx));held_metric=score_predictions(y[held],probability[held])
            detail.update(fold=fold,train_macro_f1=fit_metric['macro_f1'],holdout_macro_f1=held_metric['macro_f1'])
            details.append(detail)
            print(f'Candidate{index+1}/10 {config} fold{fold+1}: trainF1={fit_metric["macro_f1"]:.4f}, heldF1={held_metric["macro_f1"]:.4f}, best epoch={detail["best_budget"]}',flush=True)
        metric=score_predictions(y,probability)
        result={'index':index,'config':config,'folds':details,'oof_macro_f1':metric['macro_f1'],'oof_accuracy':metric['accuracy'],
            'median_budget':int(np.median([detail['best_budget'] for detail in details]))}
        np.save(trials/f'{index:02d}_oof.npy',probability);path.write_text(json.dumps(result,indent=2));results.append(result)
        print(f'COMPLETE candidate{index+1}: pooled OOF F1={metric["macro_f1"]:.4f}',flush=True)
    best=max(results,key=lambda result:result['oof_macro_f1'])
    winner=best if best['oof_macro_f1']>control_metric['macro_f1'] else {'config':{'family':'linear','widths':[],'scale':None},'median_budget':0,'oof_macro_f1':control_metric['macro_f1']}
    pd.DataFrame([{'candidate':r['index'],**r['config'],'oof_macro_f1':r['oof_macro_f1'],'median_budget':r['median_budget']} for r in results]).to_csv(OUTPUT/'search_summary.csv',index=False)
    scaler=StandardScaler().fit(x);xx=scaler.transform(x).astype(np.float32);base=linear(xx,y)
    if winner['config']['family']=='linear': model=base;details={'best_budget':0,'observations':0}
    else: model,details=fit(xx,y,base,winner['config'],42,epochs=winner['median_budget'])
    folder=OUTPUT/'winner';folder.mkdir(exist_ok=True)
    joblib.dump(TunedClassifier([model],scaler),folder/'model.joblib');joblib.dump(FunctionTransformer(validate=False),OUTPUT/'scaler.joblib')
    (folder/'selection.json').write_text(json.dumps({**winner,'refit':details,'linear_oof_macro_f1':control_metric['macro_f1'],'selected_without_new_validation_or_test':True},indent=2))
    total=sum(p.numel() for p in model.parameters()) if isinstance(model,nn.Module) else 7175
    correction=sum(p.numel() for p in model.parameters() if p.requires_grad) if isinstance(model,nn.Module) else 7175
    (folder/'metadata.json').write_text(json.dumps({'format_version':1,'artifact_type':'tuned_ensemble','kind':'mlp' if isinstance(model,nn.Module) else 'linear',
        'backbone':'panderm_large','weights':'PanDerm_Large:'+receipt['checkpoint_sha256'],'feature_dimension':1024,'feature_views':'canonical',
        'class_names':CLASS_NAMES,'training_variant':'panderm_capacity_search','details':winner['config'],'parameters':total,'nonlinear_training_parameters':correction,
        'fingerprint':receipt['fingerprint']},indent=2))
    (OUTPUT/'selection_complete.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256_file(folder/'model.joblib'),
        'selected_before_test_evaluation':True,'selection_source':'Training-fold OOF macro F1 only'},indent=2))
    metrics={};records=[];classes=[];loaded=joblib.load(folder/'model.joblib')
    for split in ['train','validation','test','official_validation','official_test']:
        indices=np.flatnonzero(frame.split==split);truth=frame.iloc[indices].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
        probability=loaded.predict_proba(np.array(features[indices]));metric=score_predictions(truth,probability);metrics[split]=metric
        saved=pd.DataFrame(probability,columns=CLASS_NAMES);saved.insert(0,'true_label',frame.iloc[indices].label.to_numpy());saved.insert(0,'image_id',frame.iloc[indices].image_id.to_numpy())
        saved.to_csv(folder/f'{split}_predictions.csv',index=False)
        records.append({'model':'Capacity search winner','split':split,**{key:metric[key] for key in ['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']}})
        classes.extend({'split':split,'lesion':label,**metric['per_class'][label]} for label in CLASS_NAMES)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2));pd.DataFrame(records).to_csv(OUTPUT/'statistics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUTPUT/'per_class_statistics.csv',index=False)
    previous=json.loads(hashes.read_text());assert all(sha256_file(path)==digest for path,digest in previous.items())
    print('SELECTED:',json.dumps(winner['config']),flush=True);print(pd.DataFrame(records).to_string(index=False),flush=True)
