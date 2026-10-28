"""Small, sequential Flower-client demonstration using synthetic data."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_federated import load_data_for_simulation
from src.client.client import SkinLesionClient
from src.models.model import build_model
from src.utils.config import ensure_dirs, load_config
from src.utils.training import evaluate
from torch.utils.data import DataLoader


def run(config):
    if config['differential_privacy']['enabled'] or config['federated']['aggregation'] != 'fedavg':
        raise ValueError('This demonstration supports non-private FedAvg only')
    torch.set_num_threads(2)
    ensure_dirs(config)
    trains, vals, test, _ = load_data_for_simulation(config, use_synthetic=True)
    model = build_model(config['model']['backbone'], config['dataset']['num_classes'],
                        dropout=config['model']['dropout'], freeze_backbone=False)
    parameters = [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]
    clients = [SkinLesionClient(i, train, val, config, torch.device('cpu'))
               for i, (train, val) in enumerate(zip(trains, vals))]
    history = []
    for round_number in range(1, config['federated']['num_rounds'] + 1):
        results = [client.fit(parameters, {}) for client in clients]
        total = sum(count for _, count, _ in results)
        parameters = [sum(weights[index] * (count / total) for weights, count, _ in results)
                      for index in range(len(parameters))]
        history.append({'round': round_number, 'participating_clients': len(results),
                        'training_examples': total,
                        'train_accuracy': sum(metrics['train_accuracy'] * count for _, count, metrics in results) / total})
    state = {key: torch.from_numpy(value) for key, value in zip(model.state_dict(), parameters)}
    model.load_state_dict(state)
    loss, accuracy = evaluate(model, DataLoader(test, batch_size=8), torch.nn.CrossEntropyLoss(), torch.device('cpu'))
    report = {'setup': {'synthetic': True, 'clients': len(clients), 'rounds': len(history),
                        'local_epochs': config['federated']['local_epochs'], 'aggregation': 'fedavg',
                        'iid': config['federated']['iid'], 'differential_privacy': False,
                        'client_train_sizes': [len(ds) for ds in trains]},
              'round_history': history, 'final': {'test_accuracy': accuracy, 'test_loss': loss}}
    output = Path(config['outputs']['models_dir']) / 'federated_metrics.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    torch.save(model.state_dict(), output.parent / 'synthetic_federated_demo.pt')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(load_config(args.config))
