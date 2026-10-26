"""Three additional two-hidden-layer heads; reuse the frozen Large protocol."""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from src.data.full_isic import CLASS_NAMES
from src.models.panderm import sha256_file
from src.utils.feature_training import score_predictions
from src.utils.tuned_training import TunedClassifier, class_weights, macro_f1_cost

OUTPUT=Path('outputs/feature_models/panderm_large_widths')
CACHE=Path('data/features/panderm_large_72dfc47de5e585b2.npy')
WIDTHS={'half':[96,32],'quarter':[48,16],'tiny':[8,4]}


def network(widths,dropout):
    first,second=widths
    return nn.Sequential(nn.Linear(1024,first),nn.LayerNorm(first),nn.ReLU(),nn.Dropout(dropout[0]),
        nn.Linear(first,second),nn.LayerNorm(second),nn.ReLU(),nn.Dropout(dropout[1]),nn.Linear(second,7))


def fit(x,y,widths,config,seed,stop=None,epochs=None):
    torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=network(widths,config['dropout']).to(device)
    xx,yy=torch.tensor(x,device=device),torch.tensor(y,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y,config['power']),dtype=torch.float32,device=device),label_smoothing=.03)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=config['decay'])
    generator=torch.Generator(device=device).manual_seed(seed)
    if stop is not None: vx,vy=torch.tensor(stop[0],device=device),stop[1]
    history,best,state,best_epoch=[],-1.,None,0
    for epoch in range(1,(epochs or 180)+1):
        model.train()
        for rows in torch.randperm(len(y),device=device,generator=generator).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(xx[rows]),yy[rows]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if stop is not None:
            model.eval()
            with torch.inference_mode(): probability=model(vx).softmax(1).cpu().numpy()
            f1=-macro_f1_cost(vy,probability)
            history.append({'epoch':epoch,'macro_f1':f1})
            if f1>best:
                best,best_epoch=f1,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch>=50 and epoch-best_epoch>=40: break
    model.cpu()
    if stop is not None: model.load_state_dict(state)
    model.eval()
    return model,{'best_budget':best_epoch if stop is not None else epochs,'observations':epoch,'history':history}


def train():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    OUTPUT.mkdir(parents=True,exist_ok=True)
    hashes=OUTPUT/'previous_artifact_hashes.json'
    if not hashes.exists():
        hashes.write_text(json.dumps({str(p):sha256_file(p) for p in OUTPUT.parent.rglob('*') if p.is_file() and OUTPUT not in p.parents},indent=2))
    frame=pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv')
    fold_frame=pd.read_csv('outputs/feature_models/panderm_large_refined/training_folds.csv')
    receipt=json.loads(CACHE.with_suffix('.json').read_text())
    assert receipt['complete'] and sha256_file(CACHE)==receipt['embeddings_sha256']
    assert sha256_file('artifacts/panderm_paper/panderm_large.pth')==receipt['checkpoint_sha256']
    features=np.load(CACHE,mmap_mode='r')
    rows=np.flatnonzero(frame.split=='train')
    assert frame.iloc[rows].image_id.tolist()==fold_frame.image_id.tolist()
    assert fold_frame.groupby('lesion_id').cv_fold.nunique().eq(1).all()
    x=np.array(features[rows]);y=frame.iloc[rows].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    assert len(y)==8207 and features.shape==(11720,1024)
    candidates={}
    for name in WIDTHS:
        candidates[name]=([{'power':.25,'decay':.01,'dropout':[.3,.2]},
                           {'power':.5,'decay':.01,'dropout':[.3,.2]},
                           {'power':.5,'decay':.02,'dropout':[.4,.2]}] if name!='tiny' else
                          [{'power':.25,'decay':.005,'dropout':[0.,0.]},
                           {'power':.5,'decay':.005,'dropout':[0.,0.]},
                           {'power':.5,'decay':.01,'dropout':[.1,.1]}])
    plan={'backbone':'panderm_large','checkpoint_sha256':receipt['checkpoint_sha256'],'widths':WIDTHS,'candidates':candidates,
        'fold_source':'outputs/feature_models/panderm_large_refined/training_folds.csv',
        'folds_sha256':sha256_file('outputs/feature_models/panderm_large_refined/training_folds.csv'),
        'selection':'Three lesion-separated training folds; pooled OOF macro F1 per architecture; same folds as the 192-64 experiment. OOF used for stopping/selection is a tuning diagnostic, not unbiased evaluation.',
        'stopping':{'maximum':180,'minimum_observations':50,'patience':40},
        'optimizer':{'name':'AdamW','lr':.0007,'batch_size':128,'label_smoothing':.03,'schedule':'fixed'},
        'final_fit':'All 8207 training rows at median best fold epoch, seed42. No validation/test selection.',
        'tiny_caveat':'8-4 hidden units slightly increases parameter count over a linear head but creates a nonlinear bottleneck; not a strict superset of the linear classifier.',
        'test_caveat':'Previously inspected tests are reused comparisons; no new test scores guide this search.'}
    protocol=OUTPUT/'experiment.json'
    if protocol.exists(): assert json.loads(protocol.read_text())==plan
    else: protocol.write_text(json.dumps(plan,indent=2))
    fold_frame.to_csv(OUTPUT/'training_folds.csv',index=False)
    frame.to_csv(OUTPUT/'split_manifest.csv',index=False)
    folds=[]
    for number in range(3):
        hold=np.flatnonzero(fold_frame.cv_fold.to_numpy()==number)
        fitting=np.flatnonzero(fold_frame.cv_fold.to_numpy()!=number)
        scaler=StandardScaler().fit(x[fitting])
        folds.append((fitting,hold,scaler.transform(x[fitting]).astype(np.float32),scaler.transform(x[hold]).astype(np.float32)))
    trials=OUTPUT/'trials';trials.mkdir(exist_ok=True)
    selected={}
    for name,widths in WIDTHS.items():
        results=[]
        for index,config in enumerate(candidates[name]):
            path=trials/f'{name}_{index}.json'
            if path.exists():
                result=json.loads(path.read_text());assert result['config']==config
                results.append(result);continue
            probability=np.zeros((len(y),7));details=[]
            for number,(fitting,hold,xx,vx) in enumerate(folds):
                model,detail=fit(xx,y[fitting],widths,config,42+number,stop=(vx,y[hold]))
                wrapped=TunedClassifier([model],FunctionTransformer(validate=False))
                probability[hold]=wrapped.predict_proba(vx)
                fit_metric=score_predictions(y[fitting],wrapped.predict_proba(xx))
                hold_metric=score_predictions(y[hold],probability[hold])
                detail.update(fold=number,train_macro_f1=fit_metric['macro_f1'],train_accuracy=fit_metric['accuracy'],holdout_macro_f1=hold_metric['macro_f1'],holdout_accuracy=hold_metric['accuracy'])
                details.append(detail)
                print(f'{name} config{index+1}/3 fold{number+1}/3: train F1={fit_metric["macro_f1"]:.4f}, held F1={hold_metric["macro_f1"]:.4f}, best epoch={detail["best_budget"]}',flush=True)
            metric=score_predictions(y,probability)
            result={'architecture':[1024,*widths,7],'config':config,'folds':details,'oof_macro_f1':metric['macro_f1'],'oof_accuracy':metric['accuracy'],
                'median_budget':max(1,int(np.median([d['best_budget'] for d in details])))}
            np.save(trials/f'{name}_{index}_oof.npy',probability)
            path.write_text(json.dumps(result,indent=2));results.append(result)
            print(f'COMPLETE {name} config{index+1}: OOF F1={metric["macro_f1"]:.4f}',flush=True)
        selected[name]=max(results,key=lambda r:r['oof_macro_f1'])
    scaler=StandardScaler().fit(x);scaled=scaler.transform(x).astype(np.float32)
    joblib.dump(FunctionTransformer(validate=False),OUTPUT/'scaler.joblib')
    for name,selection in selected.items():
        folder=OUTPUT/name;folder.mkdir(exist_ok=True)
        if (folder/'selection.json').exists(): continue
        model,refit=fit(scaled,y,WIDTHS[name],selection['config'],42,epochs=selection['median_budget'])
        joblib.dump(TunedClassifier([model],scaler),folder/'model.joblib')
        (folder/'selection.json').write_text(json.dumps({**selection,'refit':refit,'selected_without_validation_or_test':True},indent=2))
        (folder/'metadata.json').write_text(json.dumps({'format_version':1,'artifact_type':'tuned_ensemble','kind':'mlp',
            'backbone':'panderm_large','weights':'PanDerm_Large:'+receipt['checkpoint_sha256'],'feature_dimension':1024,
            'feature_views':'canonical','class_names':CLASS_NAMES,'fingerprint':receipt['fingerprint'],
            'training_variant':'panderm_width_comparison','architecture':[1024,*WIDTHS[name],7],
            'width_name':name,'parameters':sum(p.numel() for p in model.parameters()),'details':selection['config']},indent=2))
    marker=OUTPUT/'selection_complete.json'
    if not marker.exists(): marker.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'models':{name:sha256_file(OUTPUT/name/'model.joblib') for name in selected},'selected_before_test_evaluation':True},indent=2))
    records,classes=[],[]
    for name in selected:
        folder=OUTPUT/name;model=joblib.load(folder/'model.joblib');metrics={}
        for split in ['train','validation','test','official_validation','official_test']:
            indices=np.flatnonzero(frame.split==split)
            labels=frame.iloc[indices].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
            probability=model.predict_proba(np.array(features[indices]));metric=score_predictions(labels,probability);metrics[split]=metric
            saved=pd.DataFrame(probability,columns=CLASS_NAMES)
            saved.insert(0,'true_label',frame.iloc[indices].label.to_numpy());saved.insert(0,'image_id',frame.iloc[indices].image_id.to_numpy())
            saved.to_csv(folder/f'{split}_predictions.csv',index=False)
            records.append({'model':name,'split':split,**{key:metric[key] for key in ['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']}})
            classes.extend({'model':name,'split':split,'lesion':label,**metric['per_class'][label]} for label in CLASS_NAMES)
        (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(records).to_csv(OUTPUT/'statistics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUTPUT/'per_class_statistics.csv',index=False)
    previous=json.loads(hashes.read_text());assert all(sha256_file(path)==digest for path,digest in previous.items())
    print(pd.DataFrame(records).to_string(index=False),flush=True)
