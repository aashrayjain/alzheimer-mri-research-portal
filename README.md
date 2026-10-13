# Alzheimer's Disease MRI Classification & Research Portal

An Advanced Machine Learning research project focused on multi-class classification of Alzheimer's Disease stages from brain MRI scans using deep convolutional neural networks (`ResNet50`, `DenseNet121`, and `EfficientNet-B0`).

---

## 📁 Project Structure

```text
AlzheimerResearch/
│
├── .venv/                                      # Python virtual environment (ignored from VCS)
│
├── alzheimer_research_portal_artifacts/        # Research artifacts, trained weights & evaluation data
│   ├── controlled_experiment_config.json      # Controlled training configuration & experimental setup
│   ├── portal_config.json                     # Inference preprocessing & portal configuration
│   ├── validation_model_comparison.csv        # Comparative performance metrics across models
│   ├── validation_evaluation_details.json     # Detailed classification reports & confusion matrices
│   ├── resnet50_best.pth                      # Best checkpoint for ResNet50 (~94.4 MB)
│   ├── densenet121_best.pth                   # Best checkpoint for DenseNet121 (~28.4 MB)
│   └── efficientnet_b0_best.pth               # Best checkpoint for EfficientNet-B0 (~16.4 MB)
│
├── requirements.txt                            # Python package dependencies
└── README.md                                   # Project documentation and artifact overview
```

---

## 🔍 Detailed File & Internal Data Breakdown

### 1. `alzheimer_research_portal_artifacts/`

This folder contains all artifacts produced from the controlled training and validation experiments, including model weights and benchmarking metrics.

#### ⚙️ Configuration Files

* **[`controlled_experiment_config.json`](file:///c:/Users/aashr/main%20pc/MTECH/NIMS%20University%20Rajasthan%20Jaipur/Syllabus/Semester%201/Advance%20Machine%20Learning/AlzheimerResearch/alzheimer_research_portal_artifacts/controlled_experiment_config.json)**
  * **Purpose**: Records hyperparameters and reproducible training settings.
  * **Internal Data**:
    * **Random Seed**: `42`
    * **Total Dataset Size**: `6400` MRI scans
    * **Target Classes**:
      1. `MildDemented` (627 training samples)
      2. `ModerateDemented` (45 training samples)
      3. `NonDemented` (2240 training samples)
      4. `VeryMildDemented` (1568 training samples)
    * **Input Resolution & Batch Size**: `224x224`, `batch_size: 32`
    * **Optimization & Loss**: `AdamW` optimizer, weighted cross-entropy using normalized square-root inverse-frequency weights to mitigate severe class imbalance.
    * **Model Selection Strategy**: Selected checkpoint with the lowest validation loss during training.
    * **Architectures Benchmarked**: `ResNet50`, `EfficientNet-B0`, `DenseNet121`.

* **[`portal_config.json`](file:///c:/Users/aashr/main%20pc/MTECH/NIMS%20University%20Rajasthan%20Jaipur/Syllabus/Semester%201/Advance%20Machine%20Learning/AlzheimerResearch/alzheimer_research_portal_artifacts/portal_config.json)**
  * **Purpose**: Inference configuration for deploying the models into an interactive research portal/app.
  * **Internal Data**:
    * Standardized preprocessing pipeline (`input_size: 224`, `input_channels: 3`, `grayscale_to_rgb: true`).
    * ImageNet normalization constants:
      * Mean: `[0.485, 0.456, 0.406]`
      * Std: `[0.229, 0.224, 0.225]`
    * Model registry mapping names to `.pth` weights.
    * Research disclaimer: Prototype for educational and research evaluation; not intended for standalone clinical diagnosis.

---

#### 📊 Validation & Benchmark Results

* **[`validation_model_comparison.csv`](file:///c:/Users/aashr/main%20pc/MTECH/NIMS%20University%20Rajasthan%20Jaipur/Syllabus/Semester%201/Advance%20Machine%20Learning/AlzheimerResearch/alzheimer_research_portal_artifacts/validation_model_comparison.csv)**
  * **Purpose**: High-level comparison table across key metrics on the validation dataset (960 validation samples).
  * **Summary Metrics**:

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy | Best Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | **88.44%** | **0.8804** | **0.8847** | 85.18% | 0.3273 |
| **DenseNet121** | 88.13% | 0.8761 | 0.8804 | **90.10%** | **0.3052** |
| **EfficientNet-B0** | 71.46% | 0.6913 | 0.7115 | 73.59% | 0.6820 |

* **[`validation_evaluation_details.json`](file:///c:/Users/aashr/main%20pc/MTECH/NIMS%20University%20Rajasthan%20Jaipur/Syllabus/Semester%201/Advance%20Machine%20Learning/AlzheimerResearch/alzheimer_research_portal_artifacts/validation_evaluation_details.json)**
  * **Purpose**: Full per-class classification reports (Precision, Recall, F1-score, Support) and $4 \times 4$ confusion matrices for each architecture.
  * **Highlights**:
    * **DenseNet121**: Highest balanced accuracy (90.10%) and 100% recall on the underrepresented `ModerateDemented` class (10/10 samples identified).
    * **ResNet50**: Strong overall precision (88.68% weighted) and top accuracy (88.44%).
    * **EfficientNet-B0**: Struggles comparatively on subtle early-stage features (`MildDemented` recall: 46.27%).

---

#### 🧠 Trained Model Weights

* **[`resnet50_best.pth`](file:///c:/Users/aashr/main%20pc/MTECH/NIMS%20University%20Rajasthan%20Jaipur/Syllabus/Semester%201/Advance%20Machine%20Learning/AlzheimerResearch/alzheimer_research_portal_artifacts/resnet50_best.pth)**: PyTorch state dictionary checkpoint for ResNet50 (~94.4 MB).
* **[`densenet121_best.pth`](file:///c:/Users/aashr/main%20pc/MTECH/NIMS%20University%20Rajasthan%20Jaipur/Syllabus/Semester%201/Advance%20Machine%20Learning/AlzheimerResearch/alzheimer_research_portal_artifacts/densenet121_best.pth)**: PyTorch state dictionary checkpoint for DenseNet121 (~28.4 MB).
* **[`efficientnet_b0_best.pth`](file:///c:/Users/aashr/main%20pc/MTECH/NIMS%20University%20Rajasthan%20Jaipur/Syllabus/Semester%201/Advance%20Machine%20Learning/AlzheimerResearch/alzheimer_research_portal_artifacts/efficientnet_b0_best.pth)**: PyTorch state dictionary checkpoint for EfficientNet-B0 (~16.4 MB).

---

## 📦 Installation & Environment Setup

Install all dependencies using Python (>=3.10 recommended):

```bash
pip install -r requirements.txt
```

### Key Libraries:
- **PyTorch & TorchVision**: Model definitions, checkpoint loading, image transformations.
- **Pillow / OpenCV**: Image loading and pre-processing.
- **Scikit-Learn**: Validation metric calculation (Balanced Accuracy, Macro/Weighted F1, confusion matrices).
- **Pandas & NumPy**: Tabular data manipulation and numerical operations.
- **Streamlit**: Web portal framework for interactive MRI classification and result visualization.

