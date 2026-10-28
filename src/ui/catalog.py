"""Discover saved seven-class predictors independently of benchmark eligibility."""
import json
import math
from pathlib import Path
from src.ui.results import model_name


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rank_models(models):
    return sorted(models,key=lambda m:(-m.get('ranking_score',-math.inf),m['id']))


def saved_models(root=Path('outputs')):
    result=[]
    active=read(root/'diagnostics/six_model_adaptation/active_models.json')
    source_families={Path(m['source_folder']).as_posix():m['id'] for m in active}
    for path in sorted(root.rglob('metadata.json')):
        info=read(path);folder=path.parent
        if info.get('format_version')!=1 or len(info.get('class_names',[]))!=7: continue
        artifact=next((folder/name for name in ['model.joblib','model.pt','model.ubj'] if (folder/name).is_file()),None)
        scaler=folder.parent/'scaler.joblib'
        if artifact is None or not scaler.exists(): continue
        metric_path=folder/'metrics.json'
        if metric_path.exists(): metrics=read(metric_path)
        elif (folder.parent/'metrics.json').exists():
            metrics={key:value['after'] for key,value in read(folder.parent/'metrics.json').items() if isinstance(value,dict) and 'after' in value}
        else: metrics={}
        adapted=bool(info.get('data_contaminated_for_test_evaluation'))
        family=source_families.get(folder.as_posix(),folder.as_posix())
        if 'six_model_adaptation' in folder.parts: family=folder.parent.name
        elif folder.name=='adapted' and 'panderm_test_adaptation' in folder.parent.name: family='large_nn128'
        family=info.get('comparison_family_id',family)
        m={'id':artifact.as_posix(),'folder':folder,'artifact_path':artifact,'scaler_path':scaler,'info':info,
            'metrics':metrics,'name':model_name(folder,info),'neural':info['kind'] in ['mlp','specialists'],
            'stage':'adapted' if adapted else 'original','exposure':info.get('exposure',{}),'family_id':family,
            'manifest_path':folder.parent/'split_manifest.csv','checkpoint_label':info.get('checkpoint_label',folder.parent.name+'/'+folder.name),
            'reference':info['kind'] in ['paper','nearest'],
            'preferred_for_benchmark':info.get('preferred_for_benchmark',False),
            'benchmark_eligible':info.get('benchmark_eligible',not (folder.name=='adapted' and 'panderm_test_adaptation' in folder.parent.name))}
        # Earlier adaptation copies have their manifest in the diagnostic parent.
        if adapted and not m['exposure']:
            m['exposure']={s:{'images':v.get('images',0),'adaptation_overlap':v.get('images',0) if s in ['test','official_test','combined_adaptation'] else 0} for s,v in metrics.items()}
        if not m['manifest_path'].exists() and info['backbone']=='panderm_large':
            m['manifest_path']=Path('outputs/feature_models/panderm_large_comparison/split_manifest.csv')
        result.append(m)
    extra=root/'ui_catalog/additional_models.json'
    if extra.exists():
        for m in read(extra):
            for key in ['folder','artifact_path','scaler_path','manifest_path']:
                if m.get(key): m[key]=Path(m[key])
            if m['artifact_path'].is_file(): result.append(m)
    unique={m['artifact_path'].resolve().as_posix():m for m in result}
    for m in unique.values():
        score=m.get('metrics',{}).get('official_test',{}).get('macro_f1')
        m['ranking_score']=float(score) if score is not None and math.isfinite(float(score)) else -math.inf
        m['selection_label']=(f"F1 {score:.2%}" if score is not None else 'F1 unavailable')+f" · {m['name']} · {m['checkpoint_label']}"
    return rank_models(list(unique.values()))


def benchmark_models(models,stage='adapted'):
    """Best checkpoint per experiment family, with at most two neural systems."""
    families=set();chosen=[];neural=0
    preferred={m['family_id'] for m in models if m.get('preferred_for_benchmark') and m.get('stage')==stage}
    for m in rank_models(models):
        if m['family_id'] in preferred and not m.get('preferred_for_benchmark'): continue
        if m.get('reference') or not m.get('benchmark_eligible',True) or m['family_id'] in families: continue
        if stage=='adapted' and m.get('stage')!='adapted': continue
        if stage=='original' and m.get('stage')=='adapted': continue
        if m.get('metrics',{}).get('official_test',{}).get('images')!=1512: continue
        if m['neural'] and neural>=2: continue
        chosen.append(m);families.add(m['family_id']);neural+=int(m['neural'])
        if len(chosen)==6: break
    return chosen
