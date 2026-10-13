"""Inference utilities for running MRI classification and computing probabilities."""
from __future__ import annotations

import time
from typing import Dict, List, Any
import torch
import torch.nn as nn
from PIL import Image

from src.preprocessing import preprocess_image


def predict_mri(
    model: nn.Module,
    image: Image.Image,
    class_names: List[str],
    device: torch.device | str = "cpu"
) -> Dict[str, Any]:
    """
    Runs deterministic evaluation inference, measures latency,
    and returns class probabilities formatted for reporting.
    """
    model.eval()
    tensor = preprocess_image(image).to(device)

    start_time = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000

    prob_list = probabilities.cpu().tolist()
    class_probabilities = {
        class_names[i]: float(prob_list[i]) for i in range(len(class_names))
    }

    top_idx = int(torch.argmax(probabilities).item())
    predicted_class = class_names[top_idx]
    confidence = float(prob_list[top_idx])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "class_probabilities": class_probabilities,
        "latency_ms": round(latency_ms, 2),
        "device_used": str(device),
    }

