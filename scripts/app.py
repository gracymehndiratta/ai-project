"""Streamlit demo interface for skin lesion classification."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.transforms import build_transforms
from src.models.model import build_model
from src.utils.config import load_config


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
    st.set_page_config(page_title="Federated Skin Lesion Classifier", layout="centered")
    st.title("🔬 Federated Privacy-Preserving Skin Lesion Classifier")
    st.markdown(
        """
        This is a **research prototype** for multi-class classification of dermoscopic images.
        Upload a dermoscopic image to obtain a predicted lesion category and confidence score.

        ⚠️ **Not for clinical diagnosis.**
        """
    )

    config = load_config("config.yaml")
    model_dir = Path(config["outputs"]["models_dir"])

    available_models = sorted([p.name for p in model_dir.glob("*.pt")])
    if not available_models:
        st.error("No trained model found in outputs/models/. Please run training first.")
        return

    selected_model = st.sidebar.selectbox("Select model", available_models)
    model_path = model_dir / selected_model

    @st.cache_resource
    def cached_load_model(path: str):
        return load_model(config, path)

    model, device = cached_load_model(str(model_path))

    uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            pred_class, confidence, probs = predict(image, model, device, config)

        st.subheader(f"Prediction: **{pred_class.upper()}**")
        st.metric(label="Confidence", value=f"{confidence:.2%}")

        st.bar_chart(probs)

        st.caption(
            "The model was trained in a federated / privacy-aware setting and is intended "
            "for academic decision-support only."
        )


if __name__ == "__main__":
    main()
