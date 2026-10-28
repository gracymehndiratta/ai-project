"""Read-only inference audit of every selectable saved checkpoint."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def main():
    import os,json
    import numpy as np,pandas as pd,torch
    from PIL import Image
    from src.ui.catalog import saved_models,benchmark_models
    from src.models.feature_models import load_classifier,classifier_probabilities
    from src.ui.dashboard import classify_image
    os.chdir(ROOT);torch.set_num_threads(4)
    models=saved_models();results=[]
    caches={'panderm_large':'data/features/panderm_large_72dfc47de5e585b2.npy',
        'panderm_base':'data/features/panderm_base_9e9b688f78eb850d.npy',
        'efficientnet_b0':'data/features/efficientnet_b0_2af1eff877bf4a7d.npy',
        'multiview':'data/features/efficientnet_b0_ccb90bc07d54e7c0.npy'}
    features={k:np.load(v,mmap_mode='r') for k,v in caches.items()}
    image=Image.open('artifacts/ui_qa/lesion.jpg').convert('RGB')
    for m in models:
        info=m['info']
        if info.get('artifact_type')=='image_cnn':
            result=classify_image(image,m);assert result['status']=='classified',result
            probability=np.array(list(result['probabilities'].values()))[None,:]
        else:
            key='multiview' if info.get('feature_views')=='full_d4_center' else info['backbone']
            head,scaler,metadata=load_classifier(m['folder'],metadata=info,checkpoint_path=m['artifact_path'],scaler_path=m['scaler_path'])
            xx=scaler.transform(np.array(features[key][:2])).astype(np.float32)
            probability=classifier_probabilities(head,xx,info['kind'])
        assert probability.shape[1]==7 and np.isfinite(probability).all(),m['id']
        assert np.allclose(probability.sum(1),1,atol=1e-5),m['id']
        results.append({'id':m['id'],'passed':True,'macro_f1':m['ranking_score']})
    # Exercise preprocessing paths which were previously unavailable in the UI.
    for multiview in [False,True]:
        choice=next(m for m in models if m['info']['backbone']=='efficientnet_b0' and m['info']['kind']=='linear' and (m['info'].get('feature_views')=='full_d4_center')==multiview)
        result=classify_image(image,choice)
        assert result['status']=='classified' and len(result['probabilities'])==7,result
    report={'passed':True,'saved_checkpoints':len(models),'all_checkpoint_heads_passed':True,'efficientnet_canonical_and_multiview_image_inference_passed':True,
        'benchmark_models':[m['id'] for m in benchmark_models(models)],'models':results}
    Path('artifacts/ui_qa/catalog_verification.json').write_text(json.dumps(report,indent=2))
    print(f'PASS: {len(models)} saved predictors, canonical and multi-view image inference, ranked six-model benchmark',flush=True)

if __name__=='__main__':main()
