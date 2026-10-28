"""Federated-learning controls alongside the saved-model dashboard."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import streamlit as st
import yaml
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[2]


def demo_config(config,directory):
    directory=Path(directory).resolve();demo=copy.deepcopy(config)
    demo['dataset'].update(processed_dir=str(directory/'data'),raw_dir=str(directory/'raw'),synthetic_samples=140,image_size=32)
    demo['model'].update(backbone='custom_cnn',freeze_backbone=False)
    demo['federated'].update(num_clients=2,num_rounds=2,local_epochs=1,batch_size=8,fraction_fit=1.,fraction_evaluate=1.,aggregation='fedavg',iid=True)
    demo['differential_privacy']['enabled']=False
    demo['outputs']={key:str(directory/key) for key in demo['outputs']}
    return demo


def run_demo(config,directory,timeout=180):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    path=directory/'config.yaml';path.write_text(yaml.safe_dump(demo_config(config,directory)),encoding='utf-8')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1')
    return subprocess.run([sys.executable,str(ROOT/'scripts/federated_demo.py'),'--config',str(path)],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=timeout,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


def federated_panel():
    st.subheader('Federated learning: setup and test')
    st.warning('High PanDerm classification scores come from separate non-federated experiments. Real-image federated utility, a matched centralized baseline and achieved privacy parameters must be demonstrated separately. Configured DP settings are not evidence of a completed privacy guarantee.')
    config=load_config(str(ROOT/'config.yaml'));fed=config['federated']
    ham_manifest=ROOT/'data/processed/HAM10000_metadata.csv'
    if ham_manifest.exists():
        ham=pd.read_csv(ham_manifest)
        ham['image_path']=ham['image_path'].map(lambda value: Path(value) if isinstance(value,str) else None)
        ham=ham[ham.image_path.map(lambda path: path is not None and path.is_file())]
        examples=[]
        for label in ('akiec','bcc','bkl','df','mel','nv','vasc'):
            matching=ham[ham.label.eq(label)]
            if not matching.empty: examples.append(matching.iloc[0])
        if examples:
            st.markdown('**Real HAM10000 client data**')
            st.caption(f'{len(ham):,} original HAM10000 dermoscopic images are available locally. These examples are read directly from the dataset and grouped by diagnosis; no synthetic images are used in this gallery.')
            columns=st.columns(len(examples))
            for column,row in zip(columns,examples):
                column.image(str(row.image_path),use_container_width=True)
                column.caption(f"{row.label.upper()} · {row.image_id}")
    else:
        st.warning('HAM10000 images are not available at data/processed/HAM10000_metadata.csv. The dataset gallery is unavailable.')
    st.write('Each client trains on its own partition. The server averages model weights in proportion to client sample counts, then evaluates the global model.')
    st.code('Global model → clients train locally → weighted aggregation → updated global model',language=None)
    st.write(f"Configured training: {fed['num_clients']} clients, {fed['num_rounds']} rounds, {fed['local_epochs']} local epochs, {fed['aggregation'].upper()}, {'IID' if fed['iid'] else 'non-IID'} partitions.")
    st.caption(f"Differential privacy in the configured training: {'enabled' if config['differential_privacy']['enabled'] else 'disabled'}. This project simulates clients on one computer.")
    st.info('The quick demonstration uses random synthetic images, two CPU clients and two FedAvg rounds. Differential privacy is disabled for this demonstration. Its accuracy does not measure skin-lesion performance.')
    directory=ROOT/'artifacts/federated-demo'
    if st.button('Run federated demonstration',key='run_federated_demo',type='primary'):
        try:
            with st.spinner('Training two clients and aggregating their models…'): result=run_demo(config,directory)
            if result.returncode:
                st.error('The demonstration did not complete.')
                with st.expander('Demonstration diagnostics'):st.code((result.stderr+'\n'+result.stdout)[-8000:])
            else: st.success('Federated demonstration completed.')
        except subprocess.TimeoutExpired:
            st.error('The demonstration exceeded its three-minute time limit. Previously completed results remain available.')
        except OSError as error: st.error(f'Could not start the demonstration: {error}')
    report_path=directory/'models_dir/federated_metrics.json'
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'))
        st.caption('Last completed synthetic demonstration; its files are separate from all saved lesion classifiers.')
        st.json(report['setup']);st.dataframe(report['round_history'],hide_index=True,width='stretch')
        st.metric('Synthetic test accuracy',f"{report['final']['test_accuracy']:.1%}")
    with st.expander('Run the configured full Flower simulation'):
        st.code('.venv\\Scripts\\python.exe scripts/train_federated.py --config config.yaml',language='powershell')
        st.caption('Uses the prepared dataset and the configured FedAvg/FedProx and privacy settings. The quick demo above is a separate small software check.')
