# Federated Privacy-Preserving Skin Lesion Classifier

A multi-institutional deep-learning prototype for dermoscopic image classification using **Federated Learning (FL)**, **Differential Privacy (DP)** via Opacus, and **FedProx** regularization to handle non-IID client data.

> ⚠️ **Academic prototype only** — not intended for clinical diagnosis.

---

## 📋 Overview

Skin-lesion classifiers often need large, diverse datasets, but patient data cannot be freely pooled across hospitals. This project trains a shared global model without moving raw dermoscopic images between clients.

**Key features**
- Centralized CNN baseline for comparison
- Federated simulation with multiple healthcare clients
- Non-IID data partitioning via Dirichlet distribution
- FedProx regularization for heterogeneous clients
- Differential-privacy training via Opacus (optional)
- Streamlit demo interface for inference

**Dataset**: [HAM10000](https://doi.org/10.7910/DVN/DBW86T) (10,015 images, 7 classes)

---

## 🗂️ Project Structure

```
.
├── config.yaml                  # Central configuration
├── requirements.txt             # Python dependencies
├── scripts/
│   ├── download_data.py         # HAM10000 preparation helper
│   ├── train_centralized.py     # Centralized baseline trainer
│   ├── train_federated.py       # Federated simulation
│   ├── evaluate.py              # Evaluate a saved model
│   └── app.py                   # Streamlit prediction demo
├── src/
│   ├── data/                    # Dataset, transforms, partitioning
│   ├── models/                  # CNN architectures
│   ├── client/                  # Flower client (DP + FedProx)
│   ├── server/                  # Flower strategy
│   └── utils/                   # Training loops, metrics, config
├── outputs/                     # Saved models, logs, plots
└── data/                        # Raw and processed data
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare data

Option A — use the real HAM10000 dataset:

```bash
python scripts/download_data.py
```

Follow the printed instructions to download `HAM10000_metadata.csv` and the image archives into `data/raw/`.

Option B — run on synthetic data for quick testing:

No manual download needed; use the `--synthetic` flag with training scripts.

### 3. Train centralized baseline

```bash
python scripts/train_centralized.py
```

For a quick synthetic test:

```bash
python scripts/train_centralized.py --synthetic
```

### 4. Train federated model

```bash
python scripts/train_federated.py
```

Quick synthetic test:

```bash
python scripts/train_federated.py --synthetic
```

### 5. Evaluate a saved model

```bash
python scripts/evaluate.py --model outputs/models/federated_global_best.pt
```

### 6. Compare centralized vs federated results

```bash
python scripts/compare.py
```

### 7. Run a privacy-utility trade-off experiment

```bash
python scripts/experiment_privacy_utility.py --synthetic --epsilons 1.0 2.0 4.0 8.0 16.0
```

### 8. Launch demo interface

```bash
streamlit run scripts/app.py
```

---

## ⚙️ Configuration

All hyperparameters live in `config.yaml`. Key sections:

- `dataset` — image size, class names, paths
- `model` — backbone (`custom_cnn` or `resnet18`), dropout
- `centralized` — epochs, batch size, learning rate
- `federated` — number of clients/rounds, IID vs Dirichlet, FedProx mu
- `differential_privacy` — enable DP, epsilon/delta targets, max grad norm

---

## 🔒 Privacy Notes

- Raw images remain on each simulated client; only model parameters are shared.
- Differential Privacy is optional and adds noise to local gradients. Expect an accuracy/privacy trade-off.
- This is a research prototype; real-world medical deployment requires additional legal, security and clinical safeguards.

---

## 📊 Expected Outputs

Saved to `outputs/`:

- `models/centralized_best.pt` — best centralized model
- `models/federated_global_best.pt` — final global model
- `plots/*_confusion_matrix.png` — confusion matrices
- `plots/*_history.png` — accuracy/loss curves
- `logs/` — TensorBoard logs

---

## 🧪 Testing with Synthetic Data

Synthetic data is useful for verifying the pipeline without downloading HAM10000. Because the synthetic images are random noise, accuracy will be near random chance (~1/7):

```bash
python scripts/train_centralized.py --synthetic
python scripts/train_federated.py --synthetic
```

Expected performance on real HAM10000 data is significantly higher.

## ⚠️ Notes

- The current Flower simulation API (`flwr.simulation.start_simulation`) is deprecated in Flower 1.39+ but remains functional. A future update may migrate to the `flwr run` app-based workflow.
- Differential Privacy is optional. When enabled, Opacus tracks per-client privacy loss (ε, δ). Lower ε means stronger privacy but usually lower accuracy.

---

## 📝 Citation

If you use HAM10000, please cite:

> P. Tschandl, C. Rosendahl and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” *Scientific Data*, vol. 5, Article 180161, 2018.

---

## 📄 License

This project is released for academic and educational use. The HAM10000 dataset has its own license terms at the Harvard Dataverse repository.
