"""Model architecture construction and safe checkpoint loading."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision import models

from src.config import MODEL_CHECKPOINTS, DEFAULT_CLASSES


def build_architecture(model_name: str, num_classes: int = 4) -> nn.Module:
    """Builds the CNN architecture matching the research experiment."""
    normalized_name = model_name.lower().replace("-", "").replace("_", "")

    if "resnet50" in normalized_name:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif "densenet121" in normalized_name:
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif "efficientnetb0" in normalized_name:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture name: {model_name}")

    return model


def load_model_checkpoint(
    model_name: str,
    device: torch.device | str = "cpu",
    checkpoint_path: Path | None = None
) -> Tuple[nn.Module, Dict]:
    """
    Safely loads trained state dict into the constructed architecture.
    Does not allow random weight fallbacks on failure.
    """
    if checkpoint_path is None:
        if model_name not in MODEL_CHECKPOINTS:
            raise KeyError(f"Unknown model identifier: {model_name}")
        checkpoint_path = MODEL_CHECKPOINTS[model_name]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    # Load checkpoint safely
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict checkpoint in {checkpoint_path.name}")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint {checkpoint_path.name} missing 'state_dict'")

    class_names: List[str] = payload.get("class_names", DEFAULT_CLASSES)
    model = build_architecture(model_name, num_classes=len(class_names))

    # Load state dict strictly
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    metadata = {
        "model_name": payload.get("model_name", model_name),
        "class_names": class_names,
        "best_validation_loss": payload.get("best_validation_loss"),
        "best_epoch_in_history": payload.get("best_epoch_in_history"),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
    }

    return model, metadata

