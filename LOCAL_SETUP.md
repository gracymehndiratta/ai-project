# Local Windows setup

The repository is cloned in `C:\Users\Manan Kulshrestha\Downloads\AI_project`.
Use the isolated Python environment in `.venv`; activation is optional.

## Open the demo

Double-click `run-local.cmd`, or run:

```powershell
.\run-local.cmd
```

The launcher uses http://127.0.0.1:8501. To choose a different port:

```powershell
.\run-local.cmd -Port 8502
```

The complete ISIC 2018 Task 3 release is downloaded and prepared. Linear, neural
network, and XGBoost classifiers have been trained on cached EfficientNet-B0
features. The demo defaults to the neural network, selected by validation macro-F1.
See [TRAINING.md](TRAINING.md) for commands and saved experiment artifacts.

## Commands (from the project folder)

```powershell
# Validate with generated random images; no dataset download:
.\.venv\Scripts\python.exe scripts/verify_local.py

# Recommended: train all three classifiers using the existing feature cache:
.\.venv\Scripts\python.exe scripts/train_feature_models.py

# Test the new feature pipeline and all three artifact formats:
.\.venv\Scripts\python.exe -m pytest tests/test_feature_pipeline.py -q

# Open notebooks (starts a separate local service):
.\.venv\Scripts\python.exe -m jupyterlab
```

Verification output, generated images, and test checkpoints stay under
`artifacts/local-check/`, separate from the real data and models. Synthetic
checkpoints are only for validating software and cannot classify real lesions.

Real data is in `data/raw/`, metadata is in `data/processed/`, and the reusable
feature cache is in `data/features/`. Training results and checkpoints are under
`outputs/feature_models/efficientnet_b0/`.

## Verified on this laptop

- Python 3.11 in `.venv`, with GPU-enabled PyTorch 2.11.0 and CUDA 12.8.
- NVIDIA GeForce RTX 4060 Laptop GPU detected and used for training.
- Centralized training, federated training with differential privacy, checkpoint
  reload, prediction, and Streamlit tests pass using generated random images.
- Two-round non-IID federated training with differential privacy also passes.
- Notebook exploration cells run successfully without the real dataset.
- EfficientNet-B0, ResNet18, and MobileNet-V2 pretrained weights are cached.
- Installed dependencies pass `pip check`.

Exact package versions are recorded in `requirements-local.lock`.
To restore packages from the retained local wheel cache:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-local.ps1
```

The downloaded package files are under `artifacts/` and `artifacts/wheels/`.
The demo server may already be running; use its URL directly in that case.
