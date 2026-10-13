"""Gradient-weighted Class Activation Mapping (Grad-CAM) implementation."""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm

from src.preprocessing import preprocess_image


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Finds the last convolutional layer for each supported architecture."""
    normalized = model_name.lower().replace("-", "").replace("_", "")
    if "resnet50" in normalized:
        return model.layer4[-1]
    elif "densenet121" in normalized:
        # In DenseNet121, model.features is a Sequential ending with norm5 (BatchNorm2d)
        # model.features.denseblock4 is the final denseblock
        if hasattr(model.features, "denseblock4"):
            return model.features.denseblock4
        return model.features[-1]
    elif "efficientnetb0" in normalized:
        return model.features[-1]
    else:
        raise ValueError(f"Grad-CAM target layer not configured for: {model_name}")


class GradCAM:
    """Computes Grad-CAM activations safely using forward hooks and tensor gradient hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handle = None

        # Temporarily ensure inplace ReLU is disabled in the model to avoid view-modification autograd errors
        self._disable_inplace_relu(self.model)
        self._register_hooks()

    def _disable_inplace_relu(self, module: nn.Module):
        for sub_module in module.modules():
            if isinstance(sub_module, (nn.ReLU, nn.SiLU, nn.LeakyReLU)):
                sub_module.inplace = False

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Clone and retain activations
            self.activations = output
            def tensor_backward_hook(grad):
                self.gradients = grad
            self.activations.register_hook(tensor_backward_hook)

        self.hook_handle = self.target_layer.register_forward_hook(forward_hook)

    def remove_hooks(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Computes 2D normalized Grad-CAM heatmap array (224, 224)."""
        self.model.eval()
        self.model.zero_grad()

        # Enable grad temporarily for backprop
        with torch.enable_grad():
            tensor = input_tensor.clone().requires_grad_(True)
            output = self.model(tensor)

            if target_class_idx is None:
                target_class_idx = int(output.argmax(dim=1).item())

            score = output[0, target_class_idx]
            score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to extract gradients or activations for Grad-CAM.")

        # Global average pooling on gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU to retain only positive influence
        cam = torch.relu(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Resize heatmap to match input resolution (224, 224)
        cam_img = Image.fromarray((cam_norm * 255).astype(np.uint8))
        cam_resized = cam_img.resize((224, 224), Image.Resampling.BILINEAR)
        return np.array(cam_resized) / 255.0


def compute_gradcam_overlay(
    model: nn.Module,
    model_name: str,
    image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Computes heatmap and blends it with the original MRI image.
    Returns: (resized_original, overlay_image, heatmap_array)
    """
    target_layer = get_target_layer(model, model_name)
    cam = GradCAM(model, target_layer)

    try:
        input_tensor = preprocess_image(image)
        heatmap = cam.generate_heatmap(input_tensor, target_class_idx)
    finally:
        cam.remove_hooks()

    # Colormap heatmap with modern matplotlib compatibility (mpl.colormaps or cm.get_cmap)
    try:
        if hasattr(mpl, "colormaps") and colormap_name in mpl.colormaps:
            colormap = mpl.colormaps[colormap_name]
        elif hasattr(cm, "get_cmap"):
            colormap = cm.get_cmap(colormap_name)
        else:
            colormap = mpl.colormaps.get_cmap(colormap_name)
    except Exception:
        colormap = mpl.colormaps["jet"]

    colored_heatmap = colormap(heatmap)[:, :, :3]  # Strip alpha, range 0..1
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(colored_heatmap)

    original_resized = image.resize((224, 224)).convert("RGB")
    overlay = Image.blend(original_resized, heatmap_pil, alpha=alpha)

    return original_resized, overlay, heatmap

