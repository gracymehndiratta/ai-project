"""Index existing checkpoints and evaluate missing cached-feature scores; no training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def main():
    import os,json,copy
    import joblib,numpy as np,pandas as pd,torch
    from src.ui.catalog import saved_models,read
    from src.models.feature_models import load_classifier,classifier_probabilities
    from src.utils.feature_training import score_predictions
    from src.data.full_isic import CLASS_NAMES
    os.chdir(ROOT);torch.set_num_threads(4)
    out=Path('outputs/ui_catalog');out.mkdir(exist_ok=True)
    entries=[]
    def evaluate(entry,cache,manifest):
        head,scaler,info=load_classifier(entry['folder'],metadata=entry['info'],checkpoint_path=entry['artifact_path'],scaler_path=entry['scaler_path'])
        frame=pd.read_csv(manifest);features=np.load(cache,mmap_mode='r');metrics={}
        for split in ['validation','test','official_validation','official_test']:
            rows=np.flatnonzero(frame.split==split)
            x=scaler.transform(np.array(features[rows])).astype(np.float32)
            y=frame.iloc[rows].label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
            metrics[split]=score_predictions(y,classifier_probabilities(head,x,info['kind']))
        entry['metrics']=metrics
    # Every saved intermediate adaptation checkpoint is a usable seven-class predictor.
    active=read('outputs/diagnostics/six_model_adaptation/active_models.json')
    cache_paths={'panderm_large':'data/features/panderm_large_72dfc47de5e585b2.npy','panderm_base':'data/features/panderm_base_9e9b688f78eb850d.npy'}
    for item in active:
        if item['id']=='large_nn128': continue
        final=Path(item['folder']);info=read(final/'metadata.json')
        for step in range(1,4):
            folder=final.parent/f'step{step}';artifact=folder/'model.joblib'
            if not artifact.exists(): continue
            entry={'id':artifact.as_posix(),'folder':str(final),'artifact_path':str(artifact),'scaler_path':str(final.parent/'scaler.joblib'),
                'manifest_path':str(final.parent/'split_manifest.csv'),'info':info,'name':item['name'],'neural':item['neural'],
                'stage':'adapted','exposure':item['exposure'],'family_id':item['id'],'checkpoint_label':item['id']+f'/step{step}',
                'reference':False,'benchmark_eligible':False}
            evaluate(entry,cache_paths[info['backbone']],entry['manifest_path']);entries.append(entry)
    # Saved class-balancing ablations use the original EfficientNet preprocessing/scaler.
    source=Path('outputs/feature_models/efficientnet_b0')
    for folder in sorted(Path('outputs/diagnostics/smote_ablation').glob('*/*')):
        if not folder.is_dir() or folder.name not in ['mlp','xgboost']: continue
        info=copy.deepcopy(read(source/folder.name/'metadata.json'))
        artifact=folder/('model.pt' if folder.name=='mlp' else 'model.ubj')
        if not artifact.exists(): continue
        entry={'id':artifact.as_posix(),'folder':str(folder),'artifact_path':str(artifact),'scaler_path':str(source/'scaler.joblib'),
            'manifest_path':str(source/'split_manifest.csv'),'info':info,'name':'EfficientNet-B0 / '+('Neural 256→64' if folder.name=='mlp' else 'XGBoost')+' / ablation',
            'neural':folder.name=='mlp','stage':'original','exposure':{},'family_id':folder.as_posix(),
            'checkpoint_label':folder.parent.name+'/'+folder.name,'reference':False}
        evaluate(entry,'data/features/efficientnet_b0_2af1eff877bf4a7d.npy',entry['manifest_path']);entries.append(entry)
    # The standalone fine-tuned image CNN is saved outside the feature-head folders.
    folder=Path('outputs/cnn_baseline/efficientnet_b0_finetuned')
    if (folder/'best.pt').exists():
        metrics={}
        for path in folder.glob('*_predictions.csv'):
            table=pd.read_csv(path);y=table.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
            metrics[path.name.removesuffix('_predictions.csv')]=score_predictions(y,table[['score_'+name for name in CLASS_NAMES]].to_numpy())
        for filename in ['best.pt','last.pt']:
            artifact=folder/filename
            if not artifact.exists():continue
            entry={'id':artifact.as_posix(),'folder':str(folder),'artifact_path':str(artifact),'manifest_path':str(folder/'split_manifest.csv'),
                'info':{'backbone':'efficientnet_b0','kind':'cnn','class_names':CLASS_NAMES,'artifact_type':'image_cnn'},
                'name':'EfficientNet-B0 / Fine-tuned image CNN','neural':True,'stage':'original','exposure':{},
                'family_id':'efficientnet_b0_image_cnn','checkpoint_label':'efficientnet_b0_finetuned/'+filename,
                'reference':False,'metrics':metrics if filename=='best.pt' else {}}
            if filename=='last.pt':
                receipt=out/'cnn_last_metrics.json'
                if receipt.exists(): entry['metrics']=read(receipt)
                else:
                    from src.ui.dashboard import image_cnn
                    from PIL import Image
                    model,transform,device=image_cnn(str(artifact),artifact.stat().st_mtime_ns)
                    frame=pd.read_csv(folder/'split_manifest.csv');subset=frame[frame.split=='official_test']
                    predictions=[]
                    for begin in range(0,len(subset),32):
                        batch=[]
                        for image_path in subset.iloc[begin:begin+32].image_path:
                            with Image.open(image_path) as image: batch.append(transform(image.convert('RGB')))
                        with torch.inference_mode(): predictions.append(model(torch.stack(batch).to(device)).softmax(1).cpu().numpy())
                        if begin%320==0: print(f'CNN last checkpoint evaluation: {min(begin+32,len(subset))}/{len(subset)}',flush=True)
                    probs=np.concatenate(predictions)
                    y=subset.label.map(dict(zip(CLASS_NAMES,range(7)))).to_numpy()
                    entry['metrics']={'official_test':score_predictions(y,probs)}
                    receipt.write_text(json.dumps(entry['metrics'],indent=2))
                    pd.DataFrame(probs,columns=CLASS_NAMES).assign(image_id=subset.image_id.to_numpy(),true_label=subset.label.to_numpy()).to_csv(out/'cnn_last_official_test_predictions.csv',index=False)
            entries.append(entry)
    # Identical diagnostic copy of the paper probe remains explicitly a reference.
    folder=Path('outputs/diagnostics/panderm_paper_reference');source=Path('outputs/feature_models/panderm_large_comparison/paper')
    if (folder/'model.joblib').exists():
        entries.append({'id':str(folder/'model.joblib'),'folder':str(source),'artifact_path':str(folder/'model.joblib'),
            'scaler_path':str(source.parent/'scaler.joblib'),'manifest_path':str(source.parent/'split_manifest.csv'),
            'info':read(source/'metadata.json'),'name':'PanDerm Large / Authors’ linear recipe','neural':False,
            'stage':'original','exposure':{},'family_id':'paper_reference','checkpoint_label':'panderm_paper_reference',
            'reference':True,'metrics':read(source/'metrics.json')})
    (out/'additional_models.json').write_text(json.dumps(entries,indent=2,ensure_ascii=False),encoding='utf-8')
    models=saved_models()
    (out/'inventory.csv').write_text(pd.DataFrame([{'model':m['selection_label'],'artifact':str(m['artifact_path']),'selection_basis':m['selection_basis']} for m in models]).to_csv(index=False),encoding='utf-8')
    print(f'Indexed {len(models)} saved seven-class predictors; no weights modified.',flush=True)

if __name__=='__main__':main()
