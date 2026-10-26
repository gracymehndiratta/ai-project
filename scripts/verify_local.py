"""Verify local dependencies and train on generated images without dataset downloads."""
from pathlib import Path
import sys
import copy
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    import os
    os.chdir(ROOT)
    import torch
    from src.utils.config import load_config
    from scripts.train_centralized import train_centralized
    from scripts.train_federated import train_federated
    from streamlit.testing.v1 import AppTest
    from PIL import Image
    from scripts.app import load_model, predict
    from unittest.mock import patch

    torch.set_num_threads(4)
    config = load_config()
    config['dataset']['synthetic_samples'] = 140
    config['dataset']['processed_dir'] = './artifacts/local-check/data'
    for key in config['outputs']:
        config['outputs'][key] = './artifacts/local-check/' + key
    config['centralized'].update(epochs=1, batch_size=8)
    config['federated'].update(num_clients=2, num_rounds=1, local_epochs=1, batch_size=8, iid=True)
    print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), flush=True)
    centralized = train_centralized(copy.deepcopy(config), use_synthetic=True)
    federated = train_federated(copy.deepcopy(config), use_synthetic=True)
    model, device = load_model(config, str(Path(config['outputs']['models_dir']) / 'centralized_best.pt'))
    _, confidence, probabilities = predict(Image.new('RGB', (224, 224)), model, device, config)
    assert len(probabilities) == 7 and abs(sum(probabilities.values()) - 1.0) < 1e-5
    assert 0 <= confidence <= 1
    app = AppTest.from_file(str(ROOT / 'scripts/app.py')).run(timeout=60)
    assert not app.exception, app.exception
    with patch('src.utils.config.load_config', return_value=config):
        demo = AppTest.from_file(str(ROOT / 'scripts/app.py')).run(timeout=60)
        assert not demo.exception, demo.exception
        assert len(demo.get('file_uploader')) == 1
    report = {'centralized': centralized, 'federated': federated,
              'cuda': torch.cuda.is_available(), 'app_without_model': 'passed',
              'checkpoint_inference': 'passed', 'app_with_test_model': 'passed'}
    Path('artifacts/local-check/results.json').write_text(json.dumps(report, indent=2))
    print('Local verification passed.', flush=True)
