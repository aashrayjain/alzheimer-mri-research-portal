"""Alzheimer's MRI Research Intelligence Portal - Streamlit Web Application."""
from __future__ import annotations

import io
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from PIL import Image

from src.config import (
    DEFAULT_CLASSES,
    DISCLAIMER_TEXT,
    MODEL_CHECKPOINTS,
    get_controlled_config,
    get_portal_config,
)
from src.evaluation_loader import (
    get_classification_report_df,
    get_confusion_matrix_for_model,
    load_evaluation_details,
    load_model_comparison_df,
)
from src.gradcam import compute_gradcam_overlay
from src.inference import predict_mri
from src.model_loader import load_model_checkpoint
from src.preprocessing import load_and_validate_image
from src.reporting import export_payload_to_json, generate_prediction_export_payload

# Configure Page
st.set_page_config(
    page_title="Alzheimer's MRI Research Intelligence Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Quality CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #1e293b;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        color: #cbd5e1;
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .card-title {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading PyTorch model checkpoint...")
def get_cached_model(model_name: str, device_str: str):
    """Loads and caches model instance to prevent reloading on every interaction."""
    device = torch.device(device_str)
    return load_model_checkpoint(model_name, device=device)


def detect_system_device() -> torch.device:
    """Detects available computing hardware."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def render_sidebar():
    """Renders consistent navigation and hardware telemetry."""
    st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to page:",
        [
            "🔬 Page A: Research Overview",
            "🖼️ Page B: MRI Prediction Lab",
            "📊 Page C: Model Performance",
            "✨ Page D: Explainable AI (Grad-CAM)",
            "📁 Page E: Dataset & Experiment Audit",
            "📑 Page F: Report & Export",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Telemetry")
    device = detect_system_device()
    st.sidebar.markdown(f"**Compute Device:** `{device.type.upper()}`")
    st.sidebar.markdown(f"**PyTorch:** `{torch.__version__}`")
    st.sidebar.markdown(f"**Artifacts:** `Verified Local`")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "M.Tech AI & ML Research Project\n"
        "Alzheimer's Stage Classification from Brain MRI"
    )
    return page


# Main App Flow
selected_page = render_sidebar()

# Page A: Research Overview
if selected_page == "🔬 Page A: Research Overview":
    st.markdown('<div class="main-title">Alzheimer\'s MRI Research Intelligence Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep Learning | Comparative Model Evaluation | Explainable AI | MRI Classification</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Research Notice:</strong> {DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)

    st.subheader("1. Executive Research Summary")
    st.markdown(
        """
        This project investigates multi-class classification of Alzheimer's Disease stages using high-resolution brain MRI scans.
        We benchmark three foundational Deep Convolutional Neural Network architectures (**ResNet50**, **DenseNet121**, and **EfficientNet-B0**)
        under rigorous controlled experimental conditions to evaluate diagnostic efficacy, per-class sensitivity, and clinical explainability.
        """
    )

    st.subheader("2. Benchmarking Scorecard (Validation Cohort: 960 Scans)")
    df_comparison = load_model_comparison_df()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """<div class="metric-card">
                <div class="card-title">Top Overall Accuracy</div>
                <div class="card-value">88.44%</div>
                <div style="color: #10b981; font-size: 0.9rem;">ResNet50</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="metric-card">
                <div class="card-title">Top Balanced Accuracy</div>
                <div class="card-value">90.10%</div>
                <div style="color: #10b981; font-size: 0.9rem;">DenseNet121</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="metric-card">
                <div class="card-title">Lowest Validation Loss</div>
                <div class="card-value">0.3052</div>
                <div style="color: #10b981; font-size: 0.9rem;">DenseNet121</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """<div class="metric-card">
                <div class="card-title">Top Macro F1-Score</div>
                <div class="card-value">0.8804</div>
                <div style="color: #10b981; font-size: 0.9rem;">ResNet50</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(
        df_comparison.rename(
            columns={
                "model": "Architecture",
                "validation_accuracy": "Accuracy",
                "validation_macro_f1": "Macro F1",
                "validation_weighted_f1": "Weighted F1",
                "validation_balanced_accuracy": "Balanced Accuracy",
                "best_validation_loss_during_training": "Best Val Loss",
            }
        ).style.format({
            "Accuracy": "{:.2%}",
            "Macro F1": "{:.4f}",
            "Weighted F1": "{:.4f}",
            "Balanced Accuracy": "{:.2%}",
            "Best Val Loss": "{:.4f}",
        }),
        use_container_width=True,
    )

    st.subheader("3. Key Clinical & Research Takeaways")
    with st.expander("📌 Why Accuracy Alone Is Misleading (Class Imbalance Analysis)", expanded=True):
        st.markdown(
            """
            * **Extreme Rarity in ModerateDemented**: In the validation split of 960 scans, `ModerateDemented` accounts for only **10 samples** (1.04%), compared to 480 `NonDemented` scans (50.0%).
            * **DenseNet121 Superiority on Minority Classes**: DenseNet121 achieved **100% recall (10/10)** on `ModerateDemented` scans with a Balanced Accuracy of **90.10%**, making it the most clinically robust architecture despite ResNet50 having marginally higher raw accuracy (88.44% vs 88.13%).
            * **EfficientNet-B0 Limitations**: EfficientNet-B0 achieved only 71.46% accuracy and 46.27% recall on `MildDemented`, showing difficulty in distinguishing subtle early ventricular enlargement and sulcal widening.
            """
        )

# Page B: MRI Prediction Lab
elif selected_page == "🖼️ Page B: MRI Prediction Lab":
    st.markdown('<div class="main-title">MRI Prediction Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Perform live inference on axial brain MRI scans using verified trained checkpoints</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Notice:</strong> {DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("Upload Brain MRI")
        uploaded_file = st.file_uploader(
            "Select a brain MRI image file (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg"],
            help="Upload an axial T1/T2 brain MRI slice.",
        )

        model_choice = st.selectbox(
            "Select Inference Model:",
            ["DenseNet121 (Recommended - Best Balanced)", "ResNet50 (Highest Accuracy)", "EfficientNet-B0", "⚡ Compare All 3 Models Side-by-Side"],
        )

        # Example image fallback button
        if uploaded_file is None:
            st.info("Tip: You can also test a sample image from the dataset.")
            sample_dir = Path("Alzheimer_s Dataset/test/MildDemented")
            if sample_dir.is_dir():
                samples = list(sample_dir.glob("*.jpg"))
                if samples and st.button("Load Sample MRI (MildDemented)"):
                    with open(samples[0], "rb") as f:
                        uploaded_file = io.BytesIO(f.read())

    with col_right:
        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
                image = load_and_validate_image(raw_bytes)
                st.subheader("MRI Scan Preview")
                st.image(image, caption="Uploaded Scan (Resized to 224x224 RGB)", width=240)
            except Exception as e:
                st.error(f"Image validation error: {str(e)}")
                image = None
        else:
            st.info("Please upload an MRI image on the left panel to begin analysis.")
            image = None

    if image is not None:
        st.markdown("---")
        st.subheader("Inference Results")
        device_str = detect_system_device().type

        # Single Model Mode
        if "Side-by-Side" not in model_choice:
            actual_model_name = "DenseNet121" if "DenseNet" in model_choice else ("ResNet50" if "ResNet" in model_choice else "EfficientNet-B0")
            model, metadata = get_cached_model(actual_model_name, device_str)
            prediction = predict_mri(model, image, DEFAULT_CLASSES, device=device_str)

            res_col1, res_col2 = st.columns([1, 1.5])
            with res_col1:
                st.markdown(f"### Predicted Stage:\n## **{prediction['predicted_class']}**")
                st.metric("Top Confidence", f"{prediction['confidence_percentage']}%")
                st.metric("Inference Latency", f"{prediction['latency_ms']} ms")
                st.caption(f"Weights: `{metadata['checkpoint_file']}` ({metadata['checkpoint_size_mb']} MB)")

            with res_col2:
                st.markdown("#### Probability Distribution Across Stages")
                prob_df = pd.DataFrame({
                    "Stage": list(prediction["class_probabilities"].keys()),
                    "Probability": [v * 100 for v in prediction["class_probabilities"].values()],
                })
                st.bar_chart(prob_df.set_index("Stage"))

                # Store in session state for export
                st.session_state["latest_prediction"] = prediction
                st.session_state["latest_model_name"] = actual_model_name
                st.session_state["latest_metadata"] = metadata
                st.session_state["latest_image"] = image

        # All-Models Side-by-Side Mode
        else:
            cols = st.columns(3)
            all_preds = {}
            for idx, m_name in enumerate(["DenseNet121", "ResNet50", "EfficientNet-B0"]):
                model, metadata = get_cached_model(m_name, device_str)
                pred = predict_mri(model, image, DEFAULT_CLASSES, device=device_str)
                all_preds[m_name] = pred

                with cols[idx]:
                    st.markdown(f"### {m_name}")
                    st.markdown(f"**Prediction:** `{pred['predicted_class']}`")
                    st.metric("Confidence", f"{pred['confidence_percentage']}%")
                    st.caption(f"Latency: {pred['latency_ms']} ms")
                    sub_df = pd.DataFrame({
                        "Stage": list(pred["class_probabilities"].keys()),
                        "Prob (%)": [round(v * 100, 1) for v in pred["class_probabilities"].values()],
                    })
                    st.dataframe(sub_df, use_container_width=True, hide_index=True)

# Page C: Model Performance Laboratory
elif selected_page == "📊 Page C: Model Performance":
    st.markdown('<div class="main-title">Model Performance Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Direct evaluation of saved experimental artifacts, confusion matrices, and per-class reports</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Comparative Metrics", "🎯 Confusion Matrices", "📋 Per-Class Classification Reports"])

    with tab1:
        st.subheader("High-Level Validation Metrics Comparison")
        comp_df = load_model_comparison_df()
        st.dataframe(
            comp_df.rename(
                columns={
                    "model": "Model",
                    "validation_accuracy": "Accuracy",
                    "validation_macro_f1": "Macro F1",
                    "validation_weighted_f1": "Weighted F1",
                    "validation_balanced_accuracy": "Balanced Accuracy",
                    "best_validation_loss_during_training": "Best Val Loss",
                }
            ).style.format({
                "Accuracy": "{:.2%}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "Balanced Accuracy": "{:.2%}",
                "Best Val Loss": "{:.4f}",
            }),
            use_container_width=True,
        )

        st.markdown("#### Balanced Accuracy vs Raw Accuracy")
        chart_data = comp_df[["model", "validation_accuracy", "validation_balanced_accuracy"]].copy()
        chart_data["validation_accuracy"] = chart_data["validation_accuracy"] * 100
        chart_data["validation_balanced_accuracy"] = chart_data["validation_balanced_accuracy"] * 100
        chart_data = chart_data.rename(columns={
            "model": "Model",
            "validation_accuracy": "Raw Accuracy (%)",
            "validation_balanced_accuracy": "Balanced Accuracy (%)",
        }).set_index("Model")
        st.bar_chart(chart_data)

    with tab2:
        st.subheader("Confusion Matrix Analysis")
        col_m, col_opt = st.columns([1, 1])
        with col_m:
            chosen_cm_model = st.selectbox("Select Architecture:", ["DenseNet121", "ResNet50", "EfficientNet-B0"], key="cm_model_select")
        with col_opt:
            normalize_cm = st.checkbox("Normalize by True Class (Row %)", value=True)

        cm = get_confusion_matrix_for_model(chosen_cm_model, normalize=normalize_cm)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        fmt = ".1f" if normalize_cm else "d"
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=DEFAULT_CLASSES,
            yticklabels=DEFAULT_CLASSES,
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)
        ax.set_title(f"{chosen_cm_model} Confusion Matrix {'(Normalized %)' if normalize_cm else '(Count)'}", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Per-Class Classification Metrics")
        selected_rep_model = st.selectbox("Select Model for Detailed Report:", ["DenseNet121", "ResNet50", "EfficientNet-B0"], key="rep_model_select")
        rep_df = get_classification_report_df(selected_rep_model)
        st.dataframe(
            rep_df.style.format({
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1-Score": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

# Page D: Explainable AI / Grad-CAM
elif selected_page == "✨ Page D: Explainable AI (Grad-CAM)":
    st.markdown('<div class="main-title">Explainable AI & Visual Saliency (Grad-CAM)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Target-layer gradient activations highlighting salient anatomical features</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Clinical Notice:</strong> {DISCLAIMER_TEXT} Saliency maps show model feature focus, not clinical biomarker boundaries.</div>', unsafe_allow_html=True)

    # Use uploaded image if in session state, else allow upload
    cam_image = st.session_state.get("latest_image")
    if cam_image is None:
        uploaded_cam_file = st.file_uploader("Upload an MRI image for Grad-CAM explanation:", type=["png", "jpg", "jpeg"])
        if uploaded_cam_file:
            cam_image = load_and_validate_image(uploaded_cam_file.getvalue())

    if cam_image is not None:
        cam_col1, cam_col2 = st.columns([1, 2])
        with cam_col1:
            cam_model_name = st.selectbox("Target Architecture:", ["DenseNet121", "ResNet50", "EfficientNet-B0"], index=0)
            target_class_choice = st.selectbox("Target Class for Gradient:", ["Auto (Predicted Class)"] + DEFAULT_CLASSES)
            alpha_val = st.slider("Heatmap Overlay Alpha", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
            colormap_name = st.selectbox("Colormap:", ["jet", "inferno", "viridis", "magma"], index=0)

            run_gradcam_btn = st.button("Generate Grad-CAM Explanation", type="primary")

        with cam_col2:
            if run_gradcam_btn or st.session_state.get("gradcam_ran", False):
                st.session_state["gradcam_ran"] = True
                device_str = detect_system_device().type
                model, _ = get_cached_model(cam_model_name, device_str)

                class_idx = None if target_class_choice == "Auto (Predicted Class)" else DEFAULT_CLASSES.index(target_class_choice)

                with st.spinner("Computing convolutional activations and gradients..."):
                    orig_img, overlay_img, heatmap_arr = compute_gradcam_overlay(
                        model=model,
                        model_name=cam_model_name,
                        image=cam_image,
                        target_class_idx=class_idx,
                        alpha=alpha_val,
                        colormap_name=colormap_name,
                    )

                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    st.image(orig_img, caption="Preprocessed MRI (224x224)", use_container_width=True)
                with img_col2:
                    st.image(heatmap_arr, caption="Grad-CAM Activation Map", use_container_width=True)
                with img_col3:
                    st.image(overlay_img, caption="MRI & Saliency Overlay", use_container_width=True)

                st.success(f"Grad-CAM generated successfully using target layer of **{cam_model_name}**.")
    else:
        st.info("Please upload an MRI image or run inference on Page B first.")

# Page E: Dataset & Experiment Audit
elif selected_page == "📁 Page E: Dataset & Experiment Audit":
    st.markdown('<div class="main-title">Dataset & Experimental Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Full experimental reproducibility audit trail and dataset split configuration</div>', unsafe_allow_html=True)

    controlled_cfg = get_controlled_config()
    portal_cfg = get_portal_config()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Controlled Training Configuration")
        st.json(controlled_cfg)

    with col2:
        st.subheader("Inference & Preprocessing Pipeline")
        st.json(portal_cfg)

    st.subheader("Training Class Distribution & Imbalance Mitigation")
    train_dist_df = pd.DataFrame({
        "Stage Class": controlled_cfg["class_names"],
        "Training Sample Count": controlled_cfg["train_counts"],
        "Proportion (%)": [round(c / sum(controlled_cfg["train_counts"]) * 100, 2) for c in controlled_cfg["train_counts"]],
    })
    st.dataframe(train_dist_df, use_container_width=True, hide_index=True)
    st.caption("Loss Strategy: Weighted Cross-Entropy with normalized square-root inverse-frequency weights.")

# Page F: Report & Export
elif selected_page == "📑 Page F: Report & Export":
    st.markdown('<div class="main-title">Research Report & Data Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Export reproducible benchmarks, per-class reports, and live MRI inference findings</div>', unsafe_allow_html=True)

    st.subheader("1. Download Benchmark Metrics")
    comp_df = load_model_comparison_df()
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Validation Comparison CSV",
        data=csv_bytes,
        file_name="validation_model_comparison.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("2. Export Latest MRI Inference Result")
    if "latest_prediction" in st.session_state:
        pred_dict = st.session_state["latest_prediction"]
        m_name = st.session_state["latest_model_name"]
        m_meta = st.session_state["latest_metadata"]
        dev_info = detect_system_device().type

        payload = generate_prediction_export_payload(m_name, pred_dict, m_meta, dev_info)
        json_str = export_payload_to_json(payload)

        st.json(payload)
        st.download_button(
            "⬇️ Download Prediction Session Summary (JSON)",
            data=json_str,
            file_name=f"mri_prediction_{m_name.lower()}_{int(time.time())}.json",
            mime="application/json",
        )
    else:
        st.info("No inference session found. Run a prediction on 'Page B: MRI Prediction Lab' to enable structured export.")

