"""Read recorded results without modifying model artifacts or fitting scores."""
import json
from pathlib import Path
import pandas as pd
from src.utils.research_integrity import contaminated, presentation_order

CLASS_LABELS={'akiec':'Actinic keratosis / Bowen disease','bcc':'Basal cell carcinoma','bkl':'Benign keratosis-like lesion',
    'df':'Dermatofibroma','mel':'Melanoma','nv':'Melanocytic nevus','vasc':'Vascular lesion'}
SPLITS={'Official ISIC 2018 test · 1,512 images':'official_test','Released HAM test · 1,232 images':'test',
    'HAM validation · 575 images':'validation','Official validation · 193 images':'official_validation'}

COMPOSITE_METRICS=('accuracy','macro_f1','macro_f05','weighted_f1','balanced_accuracy')


def mean_five(metrics):
    return sum(metrics[key] for key in COMPOSITE_METRICS)/len(COMPOSITE_METRICS)


def adapted_models(stage='adapted'):
    registry=Path('outputs/diagnostics/six_model_adaptation/active_models.json')
    if not registry.exists(): return []
    selected=[]
    for item in json.loads(registry.read_text(encoding='utf-8')):
        folder=Path(item['folder'] if stage=='adapted' else item['source_folder'])
        info=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        metrics=json.loads(Path(item['after_metrics'] if stage=='adapted' else item['before_metrics']).read_text())
        selected.append({'id':item['id'],'folder':folder,'name':item['name'].replace(' · existing 5 updates',''),
            'info':info,'metrics':metrics,'neural':item['neural'],'exposure':item['exposure'],'stage':stage})
    if len(selected)!=6 or sum(model['neural'] for model in selected)>2:
        raise ValueError('The active comparison requires six models with at most two neural heads.')
    return selected


def local_research():
    path=Path('outputs/diagnostics/six_model_adaptation/research_reference.json')
    if not path.exists(): return []
    entry=json.loads(path.read_text(encoding='utf-8'))
    entry['metrics']=json.loads(Path(entry['metrics_path']).read_text())
    return [entry]


def exposure_label(model,split):
    exposure=model.get('exposure',{}).get(split,{})
    if model.get('stage')=='adapted' and exposure.get('adaptation_overlap',0): return 'Fitting data: used for adaptation'
    if exposure.get('prior_training_overlap',0): return f"Original training overlap: {exposure['prior_training_overlap']}/{exposure['images']}"
    return 'Fitting / reused evaluation; independence not established' if contaminated(model) else 'Historical evaluation; independence across experiments not established'


def model_name(folder,info):
    backbone={'panderm_large':'PanDerm Large','panderm_base':'PanDerm Base','efficientnet_b0':'EfficientNet-B0'}.get(info['backbone'],info['backbone'])
    kind={'linear':'Linear head','paper':'Authors’ linear recipe','mlp':'Neural head','xgboost':'XGBoost',
        'nearest':'Nearest neighbor','specialists':'Seven specialists','cnn':'Fine-tuned image CNN'}[info['kind']]
    if info['kind']=='mlp':
        architecture=info.get('architecture') or info.get('details',{}).get('architecture')
        if not architecture and info.get('details',{}).get('widths'):
            architecture=[1024,*info['details']['widths'],7]
        if not architecture:
            architecture=[1024 if info['backbone']=='panderm_large' else 768,256,64,7]
        kind+=' '+ '→'.join(map(str,architecture[1:-1]))
    if info.get('details',{}).get('representation')=='lda6': kind+=' + LDA'
    suffix={'tuned_multiview':'tuned multi-view','smote':'SMOTE','specialists_regularized':'regularized',
        'specialists_smote':'regularized + SMOTE'}.get(info.get('training_variant'))
    return f'{backbone} / {kind}'+(f' / {suffix}' if suffix else '')


def all_models(root=Path('outputs/feature_models')):
    models=[]
    for path in sorted(Path(root).glob('*/*/metadata.json')):
        info=json.loads(path.read_text());metric_path=path.parent/'metrics.json'
        if info.get('format_version')!=1 or not metric_path.exists(): continue
        if contaminated({'info':info,'folder':path.parent}): continue
        metrics=json.loads(metric_path.read_text())
        if 'official_test' not in metrics: continue
        models.append({'id':path.parent.as_posix(),'folder':path.parent,'name':model_name(path.parent,info),
            'info':info,'metrics':metrics,'neural':info['kind']=='mlp'})
    return presentation_order(models)


def top_six(models):
    selected=[];neural_count=0
    for model in presentation_order(models):
        if contaminated(model): continue
        # Binary specialist systems can contain MLPs; conservatively count mixed systems too.
        is_neural=model['neural'] or model['info']['kind']=='specialists'
        if is_neural and neural_count>=2: continue
        if model['metrics']['official_test'].get('images')!=1512: continue
        selected.append(model);neural_count+=int(is_neural)
        if len(selected)==6: break
    return selected


def summary(models,split):
    rows=[]
    for model in models:
        m=model['metrics'].get(split)
        if m is None: continue
        macro=m['per_class']['macro avg']
        rows.append({'Model':model['name'],'Images':m['images'],'Accuracy (%)':m['accuracy']*100,
            'Balanced accuracy (%)':m['balanced_accuracy']*100,'Macro precision (%)':macro['precision']*100,
            'Macro recall (%)':macro['recall']*100,'Macro F1 (%)':m['macro_f1']*100,'Macro F0.5 (%)':m.get('macro_f05',float('nan'))*100,
            'Weighted F1 (%)':m['weighted_f1']*100,'Weighted F0.5 (%)':m.get('weighted_f05',float('nan'))*100,
            'Mean five metrics (%)':mean_five(m)*100,
            'Backbone':model['info']['backbone'],'Head':model['info']['kind'],'Data role':exposure_label(model,split)})
    return pd.DataFrame(rows)


def per_class(models,split,metric):
    result=[]
    for model in models:
        row={'Model':model['name'],'Cohort':split}
        for label in CLASS_LABELS:
            item=model['metrics'][split]['per_class'][label]
            row[label.upper()]=item.get(metric,float('nan'))*100
        result.append(row)
    return pd.DataFrame(result)


def research(path=Path('data/reference/research_papers.json')):
    papers=json.loads(Path(path).read_text(encoding='utf-8'))
    if len(papers)!=5: raise ValueError('Exactly five sourced paper references are required')
    return papers


def paper_table(papers):
    rows=[]
    for paper in papers:
        variants=[{'label':'','cohort':paper['cohort'],'metrics':paper['metrics'],'f1_definition':paper['f1_definition']},*paper.get('additional_results',[])]
        for variant in variants:
            m=variant['metrics']
            rows.append({'Paper':paper['short_name']+(' / '+variant['label'] if variant['label'] else ''),'Year':paper['year'],'Dataset / evaluation':variant['cohort'],
            'Accuracy (%)':m.get('accuracy'),'Balanced accuracy (%)':m.get('balanced_accuracy'),
            'Macro F1 (%)':m.get('macro_f1'),'Weighted F1 (%)':m.get('weighted_f1'),
            'Other F1 (%)':m.get('other_f1'),'F1 definition':variant['f1_definition'],
            'F0.5 (%)':m.get('f05'),'Source':paper['url']})
    return pd.DataFrame(rows)
