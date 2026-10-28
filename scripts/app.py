"""Streamlit demo interface for skin lesion classification."""

from __future__ import annotations

import sys
import json
import subprocess
import os
import copy
import yaml
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.transforms import build_transforms
from src.models.model import build_model
from src.utils.config import load_config
from src.models.feature_models import ImageFeaturePredictor


def load_model(config: dict, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config["dataset"]["num_classes"]
    model = build_model(
        config["model"]["backbone"],
        num_classes,
        dropout=config["model"]["dropout"],
        freeze_backbone=config["model"].get("freeze_backbone", True),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict(image: Image.Image, model: torch.nn.Module, device: torch.device, config: dict):
    _, transform = build_transforms(config["dataset"]["image_size"], augment=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    predicted_class = int(probabilities.argmax())
    confidence = float(probabilities[predicted_class])
    class_names = config["dataset"]["class_names"]

    return class_names[predicted_class], confidence, dict(zip(class_names, probabilities.tolist()))


def main():
    st.set_page_config(page_title="Skin Lesion Model Comparison", layout="centered")
    st.title("🔬 Skin Lesion Model Comparison")
    st.markdown(
        """
        This is a **research prototype** for multi-class classification of dermoscopic images.
        Upload a dermoscopic image to obtain a predicted lesion category and confidence score.

        ⚠️ **Not for clinical diagnosis.**
        """
    )

    config = load_config("config.yaml")
    model_dir = Path(config["outputs"]["models_dir"])

    with st.expander("Federated learning: setup and test", expanded=True):
        st.write("The server sends a global model to simulated clients. Each client trains on its own partition, then the server combines model weights in proportion to each client's training examples. The resulting model is evaluated on held-out data.")
        st.code("Global model → clients train locally → weighted aggregation → updated global model", language=None)
        fed = config["federated"]
        st.write(f"Configured run: {fed['num_clients']} clients, {fed['num_rounds']} rounds, {fed['local_epochs']} local epochs, {fed['aggregation'].upper()}, {'IID' if fed['iid'] else 'non-IID'} partitions.")
        st.write(f"Differential privacy in the configured run: {'enabled' if config['differential_privacy']['enabled'] else 'disabled'}. This is a simulation on one computer, not a deployment across separate hospitals.")
        st.caption("Try a two-client, two-round FedAvg simulation on random synthetic images. Differential privacy is disabled in this quick demonstration. Its accuracy and model are not medically meaningful; its files are separate from trained models.")
        demo_dir = Path("artifacts/federated-demo").resolve()
        if st.button("Run federated demonstration"):
            demo = copy.deepcopy(config)
            demo["dataset"].update(processed_dir=str(demo_dir / "data"), raw_dir=str(demo_dir / "raw"), synthetic_samples=140, image_size=32)
            demo["model"].update(backbone="custom_cnn", freeze_backbone=False)
            demo["federated"].update(num_clients=2, num_rounds=2, local_epochs=1, batch_size=8, fraction_fit=1.0, fraction_evaluate=1.0, aggregation="fedavg", iid=True)
            demo["differential_privacy"]["enabled"] = False
            demo["outputs"] = {key: str(demo_dir / key) for key in demo["outputs"]}
            demo_dir.mkdir(parents=True, exist_ok=True)
            config_path = demo_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(demo))
            report_path = demo_dir / "models_dir" / "federated_metrics.json"
            report_path.unlink(missing_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            with st.spinner("Training clients and aggregating their models; this may take several minutes..."):
                try:
                    result = subprocess.run([sys.executable, "scripts/federated_demo.py", "--config", str(config_path)], capture_output=True, text=True, env=env, timeout=600)
                    if result.returncode:
                        st.error("The demonstration did not complete. Diagnostic output:")
                        st.code((result.stderr + "\n" + result.stdout)[-8000:])
                    else:
                        st.success("Federated demonstration completed.")
                except subprocess.TimeoutExpired:
                    st.error("The demonstration timed out after 10 minutes.")
        report_path = demo_dir / "models_dir" / "federated_metrics.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            st.write("Latest synthetic demonstration:", report.get("setup", {}))
            st.dataframe(report["round_history"], hide_index=True)
            st.metric("Held-out synthetic test accuracy", f"{report['final']['test_accuracy']:.1%}")
            st.caption("Random synthetic images only test the software; accuracy does not measure clinical performance.")


    legacy_models = {p.name: p for p in sorted(model_dir.glob("*.pt"))}
    feature_models = {}
    recommended = None
    recommended_score = (-1, -1, -1)
    display_names = {"linear": "Linear baseline", "mlp": "Neural network", "xgboost": "XGBoost",
                     "specialists": "Seven lesion specialists", "nearest": "Nearest-neighbor baseline",
                     "paper": "Authors' original linear probe"}
    for path in sorted(Path("outputs/feature_models").glob("*/*/metadata.json")):
        info = json.loads(path.read_text())
        if info.get("format_version") != 1:
            continue
        label = f"{info['backbone']} / {display_names[info['kind']]}"
        if info.get("training_variant") == "smote":
            label += " / SMOTE"
        elif info.get("training_variant") == "tuned_multiview":
            label += " / Tuned multi-view"
        elif info.get("training_variant") == "calibrated_specialists":
            label += " / Calibrated one-vs-rest"
        elif info.get("training_variant") == "specialists_regularized":
            label += " / Early stopping + weighting"
        elif info.get("training_variant") == "specialists_smote":
            label += " / Early stopping + SMOTE + weighting"
        elif info.get("training_variant") == "panderm_frozen_heads":
            label += " / Frozen PanDerm"
        elif info.get("training_variant") == "panderm_cv_refined":
            label += " / Training-fold tuning" if info['kind']=='xgboost' else " / Smaller 192-64 head"
        elif info.get("training_variant") == "panderm_width_comparison":
            label += f" / {info['width_name'].capitalize()} {info['architecture'][1]}-{info['architecture'][2]} head"
        elif info.get("training_variant") == "panderm_capacity_search":
            label += " / Capacity search winner"
        feature_models[label] = path.parent
        best_file = path.parent.parent / "best_model.json"
        metrics_file = path.parent / "metrics.json"
        if best_file.exists() and metrics_file.exists() and json.loads(best_file.read_text())["model"] == info["kind"]:
            validation = json.loads(metrics_file.read_text())["validation"]
            # Prefer the latest Large experiment, then its own validation winner.
            # Scores from different validation partitions do not establish a ranking.
            score = (int(info['backbone'] == 'panderm_large'), validation["macro_f1"], validation["balanced_accuracy"])
            if score > recommended_score:
                recommended, recommended_score = label, score
    available_models = list(feature_models) + list(legacy_models)
    if not available_models:
        st.info("No trained prediction model is available yet. The federated demonstration above can still be run.")
        return

    selected_model = st.sidebar.selectbox("Select model", available_models,
                                         index=available_models.index(recommended) if recommended else 0)

    @st.cache_resource
    def cached_load_model(path: str):
        return load_model(config, path)

    if selected_model in feature_models:
        folder = feature_models[selected_model]

        @st.cache_resource
        def cached_feature_predictor(path: str, version: int):
            return ImageFeaturePredictor(path, "cuda" if torch.cuda.is_available() else "cpu")

        version = max((folder / "metadata.json").stat().st_mtime_ns,
                      (folder.parent / "scaler.joblib").stat().st_mtime_ns)
        predictor = cached_feature_predictor(str(folder), version)
        if predictor.info.get("backbone", "").startswith("panderm_"):
            variant = "Large" if predictor.info['backbone'] == 'panderm_large' else "Base"
            st.caption(f"Frozen PanDerm {variant} dermatology encoder with a readout fitted using HAM10000 training images. Centralized research model.")
        else:
            st.caption("Frozen ImageNet feature extractor with a classifier trained on HAM10000. Centralized model; no differential-privacy guarantee.")
        if predictor.info.get("training_variant") == "smote":
            st.caption("Training strategy: SMOTE on training features only, without class weighting.")
        elif predictor.info.get("training_variant") == "tuned_multiview":
            st.caption("Nine image views with frozen EfficientNet; validation-selected class weighting and regularization.")
        elif predictor.info.get("training_variant") == "calibrated_specialists":
            st.caption("Seven binary specialists selected and calibrated using training lesion folds. Displayed scores are normalized for class selection.")
        elif predictor.info.get("training_variant") in {"specialists_regularized", "specialists_smote"}:
            st.caption("Seven specialists with lesion-separated early stopping, moderate regularization and training-fold calibration. Displayed scores are normalized for class selection.")
        elif predictor.info.get("training_variant") == "panderm_frozen_heads":
            if predictor.info['backbone'] == 'panderm_large':
                st.caption("Authors' released lesion-separated HAM split: 8,207 training / 575 validation / 1,232 test images. Only the classification readout is trained. Earlier experiments used a different internal split.")
            else:
                st.caption("Frozen PanDerm Base backbone with the original lesion-separated data splits. Only the classification readout uses training labels.")
            if predictor.info.get("kind") == "nearest":
                st.caption("Fixed cosine nearest-neighbor retrieval from training images. Scores are hard votes, not calibrated confidence probabilities. Bare PanDerm provides embeddings only.")
        elif predictor.info.get("training_variant") == "panderm_cv_refined":
            st.caption("Additional model: selected using three lesion-separated folds within the 8,207 training images, then refitted on all training images. The released validation and test partitions were excluded from tuning. Previous models remain available.")
            if predictor.info.get('details',{}).get('representation')=='lda6':
                st.caption("This XGBoost pipeline includes a supervised LDA projection fitted only on training data: 1,024 frozen image features become six discriminative features before the shallow trees.")
        elif predictor.info.get("training_variant") == "panderm_width_comparison":
            units=predictor.info['architecture'][1:3]
            st.caption(f"Two hidden layers: {units[0]} and {units[1]} units; {predictor.info['parameters']:,} trainable head parameters. Selected using training lesion folds, then fitted on all 8,207 training images. Earlier models remain available.")
        elif predictor.info.get("training_variant") == "panderm_capacity_search":
            details=predictor.info['details']
            st.caption(f"Best candidate in a bounded training-fold search: {details['family']}, hidden widths {details['widths']}. Test scores did not select this model; a global optimum is not established.")
            if details['family']=='residual':
                st.caption("The fitted linear path is preserved and frozen. A separate small neural branch learns a correction to its scores; it starts with zero correction.")
        if recommended == selected_model:
            st.sidebar.caption("Selected using validation macro-F1 within this experiment.")
        comparison = folder.parent / "comparison.csv"
        if comparison.exists():
            import pandas as pd
            with st.expander("Model comparison"):
                table = pd.read_csv(comparison)
                columns = [name for name in ["model", "validation_macro_f1", "validation_macro_f05", "test_macro_f1",
                                             "test_macro_f05", "test_balanced_accuracy"] if name in table.columns]
                st.dataframe(table[columns], hide_index=True)
                external = [name for name in ["model", "official_test_macro_f1", "official_test_macro_f05",
                                             "official_test_accuracy"] if name in table.columns]
                if len(external) > 1:
                    st.caption("Additional official challenge test set; excluded from model selection.")
                    st.dataframe(table[external], hide_index=True)
        if (predictor.info.get("kind") == "specialists"
                and (folder.parent / "binary_specialist_metrics.csv").exists()):
            import pandas as pd
            with st.expander("Each specialist: official test results"):
                statistics = pd.read_csv(folder.parent / "binary_specialist_metrics.csv")
                st.caption("Independent one-vs-rest decisions at training-selected thresholds. Binary accuracy includes the other six classes as negatives.")
                st.dataframe(statistics[statistics.split == "official_test"][["lesion", "family", "label_smoothing",
                    "accuracy", "f1", "f05", "precision", "recall", "balanced_accuracy"]], hide_index=True)
            history = folder.parent / "historical_comparison.csv"
            if history.exists():
                with st.expander("Comparison with previous training runs"):
                    statistics = pd.read_csv(history)
                    st.dataframe(statistics[statistics.split == "official_test"][["run", "model", "accuracy",
                        "macro_f1", "macro_f05", "macro_precision", "macro_recall"]], hide_index=True)
    else:
        model, device = cached_load_model(str(legacy_models[selected_model]))

    uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            if selected_model in feature_models:
                pred_class, confidence, probs = predictor.predict(image)
            else:
                pred_class, confidence, probs = predict(image, model, device, config)

        st.subheader(f"Prediction: **{pred_class.upper()}**")
        st.metric(label="Confidence", value=f"{confidence:.2%}")

        st.bar_chart(probs)

        st.caption("Research demonstration only. Class probabilities are not a clinical diagnosis.")


if __name__ == "__main__":
    main()

