"""Training-fold model selection for two additional frozen Large classifiers."""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb

from src.data.full_isic import CLASS_NAMES
from src.models.panderm import sha256_file
from src.utils.feature_training import score_predictions
from src.utils.tuned_training import TunedClassifier, class_weights, macro_f1_cost

OUTPUT = Path('outputs/feature_models/panderm_large_refined')
CACHE = Path('data/features/panderm_large_72dfc47de5e585b2.npy')


class F1Stopping(xgb.callback.TrainingCallback):
    def __init__(self):
        self.best, self.round = float('inf'), 0

    def after_iteration(self, model, epoch, evals_log):
        value = evals_log['validation_0']['macro_f1_cost'][-1]
        if value < self.best - 1e-6:
            self.best, self.round = value, epoch
        model.set_attr(best_iteration=str(self.round), best_score=str(self.best))
        return epoch+1 >= 400 and epoch-self.round >= 200


def network(dropout):
    return nn.Sequential(nn.Linear(1024,192),nn.LayerNorm(192),nn.ReLU(),nn.Dropout(dropout),
        nn.Linear(192,64),nn.LayerNorm(64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,7))


def fit_nn(x,y,config,seed,stop=None,epochs=None):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = network(config['dropout']).to(device)
    xx,yy = torch.tensor(x,device=device),torch.tensor(y,device=device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y,config['power']),dtype=torch.float32,device=device),label_smoothing=.03)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=config['decay'])
    generator = torch.Generator(device=device).manual_seed(seed)
    if stop is not None:
        vx,vy = torch.tensor(stop[0],device=device),stop[1]
    history,best,state,best_epoch = [],-1.,None,0
    for epoch in range(1,(epochs or 180)+1):
        model.train()
        for rows in torch.randperm(len(y),device=device,generator=generator).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(xx[rows]),yy[rows]);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if stop is not None:
            model.eval()
            with torch.inference_mode():
                probability=model(vx).softmax(1).cpu().numpy()
            f1=-macro_f1_cost(vy,probability)
            history.append({'epoch':epoch,'macro_f1':f1})
            if f1>best:
                best,best_epoch=f1,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch>=50 and epoch-best_epoch>=40: break
    model.cpu()
    if stop is not None: model.load_state_dict(state)
    model.eval()
    return model, {'best_budget':best_epoch if stop is not None else epochs,'observations':epoch,'history':history}


def projection(name,x,y=None):
    if name=='raw': model=StandardScaler()
    elif name=='lda6':
        model=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(n_components=6,solver='eigen',shrinkage='auto'))
    else:
        model=make_pipeline(StandardScaler(),PCA(n_components=int(name[3:]),svd_solver='randomized',random_state=42))
    with threadpool_limits(limits=4): model.fit(x,y)
    return model


def fit_xgb(x,y,config,seed,stop=None,rounds=None):
    model=xgb.XGBClassifier(n_estimators=rounds or 2000,max_depth=config['depth'],learning_rate=.04,
        min_child_weight=config['child'],reg_lambda=config['l2'],reg_alpha=0.,subsample=.9,colsample_bytree=.9,
        max_bin=256,tree_method='hist',device='cuda' if torch.cuda.is_available() else 'cpu',
        objective='multi:softprob',num_class=7,disable_default_eval_metric=True,eval_metric=macro_f1_cost,
        callbacks=[F1Stopping()] if stop is not None else None,n_jobs=4,random_state=seed)
    model.fit(x,y,sample_weight=class_weights(y,config['power'])[y],eval_set=[stop] if stop is not None else None,verbose=False)
    if stop is not None:
        history=model.evals_result()['validation_0']['macro_f1_cost']
        budget=int(model.best_iteration+1)
        model._Booster=model.get_booster()[:budget]
    else: budget,history=rounds,[]
    model.set_params(device='cpu',callbacks=None)
    return model,{'best_budget':budget,'observations':len(history) if stop is not None else rounds,'history':history}


def train(worker_kind=None):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    OUTPUT.mkdir(parents=True,exist_ok=True)
    hashes=OUTPUT/'previous_artifact_hashes.json'
    if not hashes.exists():
        hashes.write_text(json.dumps({str(p):sha256_file(p) for p in OUTPUT.parent.rglob('*') if p.is_file() and OUTPUT not in p.parents},indent=2))
    frame=pd.read_csv('outputs/feature_models/panderm_large_comparison/split_manifest.csv')
    receipt=json.loads(CACHE.with_suffix('.json').read_text())
    assert receipt['complete'] and sha256_file(CACHE)==receipt['embeddings_sha256']
    assert sha256_file('artifacts/panderm_paper/panderm_large.pth')==receipt['checkpoint_sha256']
    features=np.load(CACHE,mmap_mode='r')
    train_rows=np.flatnonzero(frame.split=='train')
    x=np.array(features[train_rows]);y=frame.iloc[train_rows].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
    groups=frame.iloc[train_rows].lesion_id.to_numpy()
    assert len(y)==8207 and features.shape==(11720,1024)
    splits=list(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(x,y,groups))
    assignments=np.zeros(len(y),dtype=int)
    for fold,(fit,hold) in enumerate(splits):
        assert set(groups[fit]).isdisjoint(set(groups[hold]));assignments[hold]=fold
    frame.iloc[train_rows].assign(cv_fold=assignments).to_csv(OUTPUT/'training_folds.csv',index=False)
    candidates=[]
    for representation,depth,l2,child in [('raw',1,5.,1),('raw',2,5.,1),('raw',3,5.,1),('raw',4,5.,1),
            ('raw',6,5.,3),('pca128',2,5.,1),('pca128',3,5.,1),('pca128',4,5.,1),('pca256',3,5.,1),('pca256',4,5.,1)]:
        for power in [0.,.5]:
            candidates.append({'kind':'xgboost','representation':representation,'depth':depth,'l2':l2,'child':child,'power':power})
    for power,decay,dropout in [(.25,.01,.3),(.5,.01,.3),(.5,.02,.4)]:
        candidates.append({'kind':'mlp','representation':'raw','power':power,'decay':decay,'dropout':dropout})
    for depth in [1,2,3]:
        for power in [0.,.5]:
            candidates.append({'kind':'xgboost','representation':'lda6','depth':depth,'l2':5.,'child':1,'power':power})
    plan={'backbone':'panderm_large','checkpoint_sha256':receipt['checkpoint_sha256'],'candidates':candidates,
        'selection':'Pooled three-fold training OOF macro F1. Same folds for all candidates. Fold scoring also selects stopping budgets; OOF is tuning evidence, not an unbiased test estimate.',
        'final_fit':'All 8207 training images, median best fold budget. Validation575 reserved for diagnostic reporting; neither it nor tests select settings.',
        'xgb_stopping':{'max':2000,'minimum_observed':400,'patience':200,'metric':'macro_f1_cost'},
        'nn_stopping':{'max':180,'minimum_observed':50,'patience':40},'nn_architecture':[1024,192,64,7],
        'regularization':'No SMOTE; class weighting and feature projection fitted inside each training fold only.',
        'test_policy':'Previously inspected test sets used for comparison only after both final heads are frozen.'}
    plan_path=OUTPUT/'experiment.json'
    if plan_path.exists(): assert json.loads(plan_path.read_text())==plan
    else: plan_path.write_text(json.dumps(plan,indent=2))
    trials_dir=OUTPUT/'trials';trials_dir.mkdir(exist_ok=True)
    projection_cache={}
    results=[]
    for number,config in enumerate(candidates):
        if worker_kind is not None:
            matches=(config['representation']=={'lda':'lda6','pca256':'pca256'}[worker_kind]
                if worker_kind in {'lda','pca256'} else config['kind']==worker_kind)
            if not matches: continue
        path=trials_dir/f'{number:02d}.json'
        if path.exists():
            existing=json.loads(path.read_text())
            assert existing['config']==config
            results.append(existing);continue
        probabilities=np.zeros((len(y),7));details=[]
        for fold,(fit,hold) in enumerate(splits):
            cache_key=(config['representation'],fold)
            if cache_key not in projection_cache:
                proj=projection(config['representation'],x[fit],y[fit])
                projection_cache[cache_key]=(proj.transform(x[fit]).astype(np.float32),proj.transform(x[hold]).astype(np.float32))
            xx,vx=projection_cache[cache_key]
            fitter=fit_nn if config['kind']=='mlp' else fit_xgb
            model,detail=fitter(xx,y[fit],config,42+fold,stop=(vx,y[hold]))
            wrapped=TunedClassifier([model],FunctionTransformer(validate=False))
            probabilities[hold]=wrapped.predict_proba(vx)
            fitted=score_predictions(y[fit],wrapped.predict_proba(xx))
            held=score_predictions(y[hold],probabilities[hold])
            detail.update(fold=fold,train_images=len(fit),holdout_images=len(hold),
                train_accuracy=fitted['accuracy'],train_macro_f1=fitted['macro_f1'],
                holdout_accuracy=held['accuracy'],holdout_macro_f1=held['macro_f1'])
            details.append(detail)
            print(f'Trial {number+1}/{len(candidates)} {config} fold{fold+1}: train F1={fitted["macro_f1"]:.4f}, holdout F1={held["macro_f1"]:.4f}, budget={detail["best_budget"]}',flush=True)
        metric=score_predictions(y,probabilities)
        result={'number':number,'config':config,'folds':details,'oof_macro_f1':metric['macro_f1'],
            'oof_accuracy':metric['accuracy'],'oof_macro_f05':metric['macro_f05'],
            'median_budget':max(1,int(np.median([d['best_budget'] for d in details])))}
        np.save(trials_dir/f'{number:02d}_oof.npy',probabilities)
        path.write_text(json.dumps(result,indent=2));results.append(result)
        print(f'COMPLETE trial{number+1}: pooled OOF F1={metric["macro_f1"]:.4f}',flush=True)
    if worker_kind is not None:
        print(f'Completed independent {worker_kind} tuning trials; final fitting remains with the main process.',flush=True)
        return
    selected={kind:max((r for r in results if r['config']['kind']==kind),key=lambda r:r['oof_macro_f1']) for kind in ['xgboost','mlp']}
    pd.DataFrame([{'trial':r['number'],**r['config'],'oof_macro_f1':r['oof_macro_f1'],'median_budget':r['median_budget']} for r in results]).to_csv(OUTPUT/'search_summary.csv',index=False)
    joblib.dump(FunctionTransformer(validate=False),OUTPUT/'scaler.joblib')
    for kind,selection in selected.items():
        folder=OUTPUT/kind;folder.mkdir(exist_ok=True)
        if (folder/'selection.json').exists(): continue
        config=selection['config'];proj=projection(config['representation'],x,y)
        xx=proj.transform(x).astype(np.float32)
        if kind=='mlp': model,details=fit_nn(xx,y,config,42,epochs=selection['median_budget'])
        else: model,details=fit_xgb(xx,y,config,42,rounds=selection['median_budget'])
        wrapped=TunedClassifier([model],proj);joblib.dump(wrapped,folder/'model.joblib')
        (folder/'selection.json').write_text(json.dumps({**selection,'refit':details,'selected_without_validation_or_test':True},indent=2))
        (folder/'metadata.json').write_text(json.dumps({'format_version':1,'artifact_type':'tuned_ensemble','kind':kind,
            'backbone':'panderm_large','weights':'PanDerm_Large:'+receipt['checkpoint_sha256'],
            'feature_dimension':1024,'feature_views':'canonical','class_names':CLASS_NAMES,'fingerprint':receipt['fingerprint'],
            'training_variant':'panderm_cv_refined','details':config,'architecture':[1024,192,64,7] if kind=='mlp' else None},indent=2))
    marker=OUTPUT/'selection_complete.json'
    if not marker.exists(): marker.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'selected_before_test_evaluation':True,'models':{k:sha256_file(OUTPUT/k/'model.joblib') for k in selected}},indent=2))
    frame.to_csv(OUTPUT/'split_manifest.csv',index=False)
    records,classes=[],[]
    for kind in selected:
        folder=OUTPUT/kind;model=joblib.load(folder/'model.joblib');metrics={}
        for split in ['train','validation','test','official_validation','official_test']:
            rows=np.flatnonzero(frame.split==split)
            truth=frame.iloc[rows].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
            probability=model.predict_proba(np.array(features[rows]))
            metric=score_predictions(truth,probability);metrics[split]=metric
            saved=pd.DataFrame(probability,columns=CLASS_NAMES)
            saved.insert(0,'true_label',frame.iloc[rows].label.to_numpy());saved.insert(0,'image_id',frame.iloc[rows].image_id.to_numpy())
            saved.to_csv(folder/f'{split}_predictions.csv',index=False)
            records.append({'model':kind,'split':split,**{name:metric[name] for name in ['accuracy','macro_f1','macro_f05','weighted_f1','weighted_f05','balanced_accuracy']}})
            classes.extend({'model':kind,'split':split,'lesion':name,**metric['per_class'][name]} for name in CLASS_NAMES)
        (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(records).to_csv(OUTPUT/'statistics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUTPUT/'per_class_statistics.csv',index=False)
    previous=json.loads(hashes.read_text());assert all(sha256_file(path)==digest for path,digest in previous.items())
    print(pd.DataFrame(records).to_string(index=False),flush=True)
