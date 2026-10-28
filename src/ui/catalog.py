"""Discover saved seven-class predictors independently of benchmark eligibility."""
import json
from pathlib import Path
from src.ui.results import model_name
from src.utils.research_integrity import contaminated, presentation_order


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rank_models(models):
    return presentation_order(models)


def saved_models(root=Path('outputs')):
    result=[]
    registry=root/'diagnostics/six_model_adaptation/active_models.json'
    active=read(registry) if registry.exists() else []
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
        adapted=contaminated({'info':info,'folder':folder})
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
        m['selection_label']=f"{m['name']} · {m['checkpoint_label']}" + (' · historical fitting' if contaminated(m) else '')
        m['selection_basis']='alphabetical; evaluation scores never used'
    return rank_models(list(unique.values()))


def benchmark_models(models,stage='original'):
    """Stable presentation subset; adapted stage is a historical fitting archive."""
    families=set();chosen=[];neural=0
    for m in rank_models(models):
        if stage!='adapted' and contaminated(m): continue
        if m.get('reference') or not m.get('benchmark_eligible',True) or m['family_id'] in families: continue
        if stage=='adapted' and m.get('stage')!='adapted': continue
        if stage=='original' and m.get('stage')=='adapted': continue
        if m['neural'] and neural>=2: continue
        chosen.append(m);families.add(m['family_id']);neural+=int(m['neural'])
        if len(chosen)==6: break
    return chosen
