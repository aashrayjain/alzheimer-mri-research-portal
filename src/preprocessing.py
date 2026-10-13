"""Preprocessing pipelines for MRI inference matching training configurations."""
from __future__ import annotations

import io
from typing import Union
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def get_inference_transform() -> transforms.Compose:
    """Returns standardized inference transformation."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_and_validate_image(image_input: Union[bytes, io.BytesIO, str, Image.Image]) -> Image.Image:
    """
    Validates, decodes, and converts MRI image to 3-channel RGB.
    Safely catches corrupt or unsupported formats.
    """
    try:
        if isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        else:
            image = Image.open(image_input)

        # Force load to catch truncated images
        image.load()
        # Convert grayscale or RGBA cleanly to RGB
        image_rgb = image.convert("RGB")
        return image_rgb
    except (UnidentifiedImageError, OSError, ValueError) as err:
        raise ValueError(f"Unable to read or decode image file: {str(err)}") from err


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Transforms PIL Image to normalized 4D tensor (1, 3, 224, 224).
    """
    transform = get_inference_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

