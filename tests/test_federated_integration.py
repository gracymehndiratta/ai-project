"""Regression checks for the merged federated controls and non-private metrics."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import TensorDataset
from src.ui.federated import demo_config
from src.utils.config import load_config
from src.client.client import SkinLesionClient
from scripts.federated_demo import run


def test_demo_config_is_isolated_and_does_not_mutate_real_training(tmp_path):
    original=load_config('config.yaml');before=deepcopy(original)
    demo=demo_config(original,tmp_path)
    assert original==before
    assert demo['dataset']['synthetic_samples']==140 and demo['dataset']['image_size']==32
    assert demo['federated']['num_clients']==2 and demo['federated']['num_rounds']==2
    assert demo['model']['backbone']=='custom_cnn' and not demo['differential_privacy']['enabled']
    assert all(Path(value).is_relative_to(tmp_path) for value in demo['outputs'].values())


def test_non_private_client_metrics_are_finite_and_json_serializable(tmp_path):
    torch.set_num_threads(2)
    config=demo_config(load_config('config.yaml'),tmp_path)
    data=TensorDataset(torch.randn(14,3,32,32),torch.arange(7).repeat(2))
    client=SkinLesionClient(0,data,data,config,torch.device('cpu'))
    params,count,metrics=client.fit(client.get_parameters({}),{})
    assert count==14 and params
    assert 'epsilon' not in metrics and 'client_id' not in metrics
    assert 0<=metrics['train_accuracy']<=1
    json.dumps(metrics,allow_nan=False)


@pytest.mark.parametrize('setting',['privacy','fedprox'])
def test_quick_demo_rejects_unsupported_training_modes(tmp_path,setting):
    config=demo_config(load_config('config.yaml'),tmp_path)
    if setting=='privacy':config['differential_privacy']['enabled']=True
    else:config['federated']['aggregation']='fedprox'
    with pytest.raises(ValueError,match='non-private FedAvg'):run(config)
