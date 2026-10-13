"""Configuration loader and path resolution for Alzheimer Research Portal."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Locate project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "alzheimer_research_portal_artifacts"

CONTROLLED_CONFIG_PATH = ARTIFACTS_DIR / "controlled_experiment_config.json"
PORTAL_CONFIG_PATH = ARTIFACTS_DIR / "portal_config.json"
VAL_COMPARISON_CSV_PATH = ARTIFACTS_DIR / "validation_model_comparison.csv"
VAL_EVAL_DETAILS_JSON_PATH = ARTIFACTS_DIR / "validation_evaluation_details.json"

MODEL_CHECKPOINTS = {
    "ResNet50": ARTIFACTS_DIR / "resnet50_best.pth",
    "DenseNet121": ARTIFACTS_DIR / "densenet121_best.pth",
    "EfficientNet-B0": ARTIFACTS_DIR / "efficientnet_b0_best.pth",
}

DEFAULT_CLASSES: List[str] = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

DISCLAIMER_TEXT = (
    "This application is an experimental deep-learning research prototype for educational "
    "and research evaluation. Its predictions are not medical diagnoses and must not be used "
    "independently for clinical decisions. The system has not been established as a clinically "
    "validated screening or diagnostic tool."
)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Safely loads and parses a JSON file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_portal_config() -> Dict[str, Any]:
    """Retrieves portal configuration."""
    return load_json_file(PORTAL_CONFIG_PATH)


def get_controlled_config() -> Dict[str, Any]:
    """Retrieves controlled experiment configuration."""
    return load_json_file(CONTROLLED_CONFIG_PATH)

