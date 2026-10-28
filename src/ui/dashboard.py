"""Six-model research dashboard; expensive inference is loaded only on demand."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
from src.ui.results import CLASS_LABELS,SPLITS,all_models,top_six,summary,per_class,research,paper_table,adapted_models,local_research,exposure_label,mean_five
from src.ui.federated import federated_panel
from src.ui.catalog import saved_models,benchmark_models
from src.ui.uploads import decode_image
from src.models.feature_models import build_extractor,load_classifier,classifier_probabilities,MultiViewTransform,extract_multiview


@st.cache_resource(max_entries=2)
def extractor(backbone):
    model,transform,dimension,weights=build_extractor(backbone)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(),transform,device,dimension,weights


@st.cache_resource(max_entries=12)
def classifier(path,version,metadata_json=None,checkpoint_path=None,scaler_path=None):
    return load_classifier(path,metadata=json.loads(metadata_json) if metadata_json else None,
        checkpoint_path=checkpoint_path,scaler_path=scaler_path)


@st.cache_resource(max_entries=2)
def image_cnn(path,version):
    from torchvision import models,transforms
    from torchvision.transforms import InterpolationMode
    model=models.efficientnet_b0(weights=None)
    model.classifier=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(1280,7))
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    transform=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return model.to(device).eval(),transform,device



def formatted(data):
    return data.map(lambda v:f'{v:.2f}' if isinstance(v,(float,np.floating)) and pd.notna(v)
        else 'Not reported' if v is None or (isinstance(v,float) and pd.isna(v)) else v)


def show_table(data,key):
    columns={name:st.column_config.NumberColumn(name,format='%.2f') for name in data.columns if name.endswith('(%)')}
    st.dataframe(data,hide_index=True,width="stretch",column_config=columns,key=key)


def classify_image(image,chosen):
    info=chosen['info'];folder=chosen['folder']
    artifact=chosen.get('artifact_path',folder/'model.joblib')
    if info.get('artifact_type')=='image_cnn':
        model,transform,device=image_cnn(str(artifact),artifact.stat().st_mtime_ns)
        with torch.inference_mode(): probabilities=model(transform(image).unsqueeze(0).to(device)).softmax(1).cpu().numpy()[0]
        metadata=info
    else:
        head,parent_scaler,metadata=classifier(str(folder),artifact.stat().st_mtime_ns,json.dumps(info),str(artifact),str(chosen.get('scaler_path',folder.parent/'scaler.joblib')))
        encoder,transform,device,dimension,weights=extractor(info['backbone'])
        multiview=metadata.get('feature_views')=='full_d4_center'
        if multiview: transform=MultiViewTransform(transform);dimension*=2
        if metadata['weights']!=weights or metadata['feature_dimension']!=dimension:
            raise ValueError('The saved classifier does not match its encoder.')
        with torch.inference_mode():
            batch=transform(image).unsqueeze(0).to(device)
            features=(extract_multiview(encoder,batch) if multiview else encoder(batch)).cpu().numpy()
        probabilities=classifier_probabilities(head,parent_scaler.transform(features).astype(np.float32),metadata['kind'])[0]
    if probabilities.shape!=(7,) or not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(),1):
        raise ValueError('The classifier returned invalid probabilities.')
    labels=metadata['class_names'];index=int(probabilities.argmax())
    return {'status':'classified','label':labels[index],'probability':float(probabilities[index]),
        'probabilities':dict(zip(labels,map(float,probabilities)))}


def image_checker(models,selected):
    st.subheader('Check a dermoscopic image')
    st.info('Any readable JPG or PNG can be scored. The model always chooses among seven lesion categories; it does not verify that the image contains a skin lesion.')
    choices=models
    if not choices: st.warning('No saved classifiers were found.');return
    by_id={m['id']:m for m in choices}
    selected_id=st.selectbox('Prediction model',list(by_id),format_func=lambda key:by_id[key]['selection_label'],key='prediction_model',filter_mode='contains')
    chosen=by_id[selected_id]
    st.caption(f"{len(choices)} saved seven-class predictors. Ordered by official-cohort macro F1, highest first. Unscored checkpoints appear last. The benchmark filter does not restrict this dropdown.")
    st.caption('Checkpoint: '+str(chosen['artifact_path']))
    st.caption('Class probabilities compare the seven trained categories; they do not establish lesion presence.')
    mode=st.radio('Image source',['Upload image','Try a dataset example'],horizontal=True,key='image_source')
    payload=None
    if mode=='Upload image':
        upload=st.file_uploader('Upload JPG or PNG',type=['jpg','jpeg','png'],max_upload_size=10,key='lesion_upload')
        if upload is not None: payload=upload.getvalue()
    else:
        code=st.selectbox('Example lesion type',list(CLASS_LABELS),format_func=lambda code:f'{code.upper()} · {CLASS_LABELS[code]}',key='example_class')
        manifest=pd.read_csv(chosen.get('manifest_path',chosen['folder'].parent/'split_manifest.csv'))
        example=manifest[(manifest.split=='official_test')&(manifest.label==code)].iloc[0]
        payload=Path(example.image_path).read_bytes()
        st.caption(f'Public dataset example: {example.image_id}. True label: {code.upper()}. This is an inference demonstration.')
    if payload is None: return
    try: image=decode_image(payload)
    except ValueError as error: st.error(str(error));return
    st.image(image,caption='Input image',width=380)
    if st.button('Analyze image',type='primary',key='analyze_image'):
        try:
            with st.spinner('Analyzing image…'): result=classify_image(image,chosen)
            if result['status']!='classified':
                st.warning(result['reason']);st.caption('This does not establish that the image is healthy or lesion-free.')
            else:
                label=result['label'];st.success(f'Predicted category: {CLASS_LABELS[label]} ({label.upper()})')
                st.metric('Predicted-class probability',f"{result['probability']:.2%}")
                values=pd.DataFrame([{'Lesion':f'{code.upper()} · {CLASS_LABELS[code]}','Probability (%)':value*100} for code,value in result['probabilities'].items()])
                st.bar_chart(values,x='Lesion',y='Probability (%)',horizontal=True);show_table(values,'prediction_probabilities')
                st.caption('This prediction does not establish that the image contains a skin lesion.')
        except Exception as error:
            st.error('Image analysis could not complete. Comparison results remain available.')
            with st.expander('Technical error'): st.code(str(error))
    st.caption('Research demonstration only. Not for clinical diagnosis.')


def paper_view(papers,selected,split,cohort):
    st.subheader('Research models evaluated here')
    references=local_research()
    rows=summary(selected,split).drop(columns=['Backbone','Head'])
    for ref in references:
        m=ref['metrics'][split]
        row={'Model':ref['name'],'Images':m['images'],'Mean five metrics (%)':mean_five(m)*100,'Data role':'Unchanged reference; no test adaptation'}
        for title,key in [('Accuracy','accuracy'),('Macro F1','macro_f1'),('Macro F0.5','macro_f05'),('Weighted F1','weighted_f1'),('Balanced accuracy','balanced_accuracy')]: row[title+' (%)']=m[key]*100
        rows=pd.concat([rows,pd.DataFrame([row])],ignore_index=True)
    show_table(rows[['Model','Images','Accuracy (%)','Macro F1 (%)','Macro F0.5 (%)','Mean five metrics (%)','Data role']],'locally_run_research')
    for ref in references:
        st.caption(ref['provenance'])
        st.caption('Reference weights were checked unchanged before and after evaluation. All displayed scores above were computed locally from saved predictions.')
    st.subheader('Both cohorts — five-metric average')
    st.caption('Equal-weight mean of accuracy, macro F1, macro F0.5, weighted F1 and balanced accuracy. This is a declared comparison criterion, not a universal ranking. Adapted checkpoints were selected using fitting scores on both cohorts.')
    comparisons=[]
    reference=references[0]['metrics']
    for model in selected:
        ham=mean_five(model['metrics']['test']);isic=mean_five(model['metrics']['official_test'])
        comparisons.append({'Model':model['name'],'HAM mean (%)':ham*100,'ISIC mean (%)':isic*100,
            'HAM difference (pp)':(ham-mean_five(reference['test']))*100,
            'ISIC difference (pp)':(isic-mean_five(reference['official_test']))*100,
            'Above reference on both':ham>mean_five(reference['test']) and isic>mean_five(reference['official_test'])})
    comparisons.append({'Model':'Authors’ recipe — unchanged','HAM mean (%)':mean_five(reference['test'])*100,
        'ISIC mean (%)':mean_five(reference['official_test'])*100,'HAM difference (pp)':0.,'ISIC difference (pp)':0.,'Above reference on both':False})
    show_table(pd.DataFrame(comparisons),'joint_cohort_comparison')
    st.subheader('Five-paper execution status')
    statuses=[{'Paper':p['short_name'],'Local evaluation':'Completed: official encoder + authors’ linear-probe recipe' if p['id']=='panderm2025' else 'Not run: matching trained checkpoint not verified','Source':p['url']} for p in papers]
    st.dataframe(pd.DataFrame(statuses),hide_index=True,width='stretch',column_config={'Source':st.column_config.LinkColumn('Source',display_text='Open paper')})
    st.caption('The other four papers have no local scores here. Rebuilding a similarly named architecture would not reproduce their trained model.')
    with st.expander('Published findings — background only, not local benchmark results'):
        st.dataframe(formatted(paper_table(papers)),hide_index=True,width='stretch',column_config={'Source':st.column_config.LinkColumn('Source',display_text='Open paper')})
        for paper in papers:
            st.markdown(f"**[{paper['title']}]({paper['url']})**")
            st.write(paper['finding']);st.caption(paper['caveat'])


def main():
    st.set_page_config(page_title='Skin Lesion Research Comparison',page_icon='🔬',layout='wide')
    st.markdown('<style>.block-container{max-width:1450px;padding-top:2rem} div[data-testid="stMetric"]{background:rgba(125,150,180,.12);padding:14px;border-radius:12px;border:1px solid rgba(125,150,180,.2)}</style>',unsafe_allow_html=True)
    st.title('Skin Lesion Research Comparison')
    st.write('Six local models, five research papers, and seven lesion categories — with explicit metrics and evaluation protocols.')
    models=saved_models();papers=research()
    stage=st.sidebar.radio('Model checkpoints',['Adapted models','Original checkpoints'],key='checkpoint_stage')
    selected=benchmark_models(models,'adapted' if stage=='Adapted models' else 'original')
    if len(selected)!=6: st.error('Six complete model artifacts are required in outputs/feature_models.');return
    st.sidebar.header('Evaluation cohort')
    cohort=st.sidebar.selectbox('Show local results on',list(SPLITS),key='evaluation_cohort');split=SPLITS[cohort]
    st.sidebar.caption('Top six eligible model families by official-cohort macro F1, with at most two nonlinear neural systems. The four refined Large families use checkpoints selected by the weaker of their HAM/ISIC five-metric gains; other families use their best eligible saved checkpoint. The adapted view uses adapted copies; the authors’ recipe is a separate unchanged reference.')
    st.sidebar.caption('Tests were inspected repeatedly: exploratory comparisons, not a fresh blind benchmark.')
    for column,label,value in zip(st.columns(4),['Compared models','Neural heads','Research papers','Lesion categories'],[6,sum(m['neural'] for m in selected),len(papers),len(CLASS_LABELS)]): column.metric(label,value)
    tabs=st.tabs(['Our six models','Research papers','Seven lesion types','Image checker','Adaptation history','Federated learning'])
    with tabs[0]:
        st.subheader('Six selected local models')
        st.caption('All six use the same evaluation image IDs. Toggle original checkpoints to compare before and after adaptation. Both former test sets are fitting data for adapted models.')
        data=summary(selected,split)
        if data.Images.nunique()>1: st.warning('Internal cohorts differ across model families. Use the official test view for the same 1,512-image population.')
        show_table(data[['Model','Images','Accuracy (%)','Macro F1 (%)','Macro F0.5 (%)','Weighted F1 (%)','Balanced accuracy (%)','Mean five metrics (%)','Data role']],'six_model_results')
        st.bar_chart(data,x='Model',y='Macro F1 (%)',horizontal=True)
        st.download_button('Download local comparison CSV',data.to_csv(index=False).encode(),'local_model_comparison.csv','text/csv',key='download_local')
        with st.expander('Full precision, recall and weighted statistics'): show_table(data,'full_local_results')
        with st.expander('Model provenance and training exposure'):
            for model in selected: st.write(f"**{model['name']}** — `{model['folder'].as_posix()}`")
            st.caption('Base uses the earlier 7,014-image training split; Large uses the authors’ released 8,207-image split. Training exposure differs. Official-test image IDs are checked by the verification script.')
    with tabs[1]: paper_view(papers,selected,split,cohort)
    with tabs[2]:
        st.subheader('Compare all seven lesion types')
        for code,label in CLASS_LABELS.items(): st.caption(f'{code.upper()} — {label}')
        label=st.selectbox('Class metric',['F1','F0.5','Precision','Recall'],key='lesion_metric')
        metric={'F1':'f1-score','F0.5':'f0.5-score','Precision':'precision','Recall':'recall'}[label]
        show_table(per_class(selected,split,metric),'seven_class_local')
        st.caption('Per-class metrics are one-versus-rest. Per-class “accuracy” is not substituted for multiclass accuracy.')
        st.subheader('Unchanged research reference — locally computed class scores')
        refrows=[]
        for ref in local_research():
            refrows.append({'Model':ref['name'],**{code.upper():ref['metrics'][split]['per_class'][code].get(metric,float('nan'))*100 for code in CLASS_LABELS}})
        show_table(pd.DataFrame(refrows),'seven_class_research')
    with tabs[3]: image_checker(models,selected)
    with tabs[4]:
        st.subheader('Adaptation receipts and preserved originals')
        joint=Path('outputs/diagnostics/joint_refinement/history.csv')
        if joint.exists():
            st.write('Latest refinement: both former tests (HAM 1,232 + ISIC 1,512), equal cohort weight, at most eight further updates per Large head. Neural/linear LR 0.00014; XGBoost eta 0.004444; backbone and feature projections frozen. Original dropout retained, class-weight power 0.25, smoothing 0.01, AdamW decay 0.01.')
            st.caption('Checkpoint selection maximizes the weaker cohort’s five-metric gain over the unchanged reference, with step zero eligible. These remain fitting comparisons. All earlier checkpoints are preserved.')
            st.dataframe(pd.read_csv(joint),hide_index=True,width='stretch')

        st.write('Five previously unadapted models received four updates on 2,744 combined former-test images. The neural 128 model reuses its existing five-update checkpoint. Neural/linear updates use accumulated full-data gradients at LR 0.00023333; XGBoost appends four rounds at eta 0.01333333 with its original projection frozen.')
        st.caption('The author-reference model received no adaptation. XGBoost rounds and neural gradient steps are different units of work.')
        history=Path('outputs/diagnostics/six_model_adaptation/adaptation_history.csv')
        if history.exists(): st.dataframe(pd.read_csv(history),hide_index=True,width='stretch')
        st.warning('The table below preserves the earlier 128-unit experiment. These are fitting metrics, not held-out test results.')
        rows=[]
        for number in range(1,6):
            path=Path('outputs/diagnostics/panderm_test_adaptation_once' if number==1 else f'outputs/diagnostics/panderm_test_adaptation_step{number}')/'metrics.json'
            if not path.exists(): continue
            values=json.loads(path.read_text());fit=values['combined_adaptation']['after'];validation=values['validation']['after']
            rows.append({'Updates':number,'Combined fitting accuracy (%)':fit['accuracy']*100,'Combined fitting macro F1 (%)':fit['macro_f1']*100,
                'Combined fitting macro F0.5 (%)':fit['macro_f05']*100,'HAM validation macro F1 (%)':validation['macro_f1']*100})
        show_table(pd.DataFrame(rows),'adaptation_history')
        st.caption('Validation was excluded from adaptation, but inspected repeatedly. Official validation contains only one dermatofibroma case.')

    with tabs[5]: federated_panel()
