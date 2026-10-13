"""Unit tests verifying Grad-CAM generation across all models and colormaps."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import build_architecture, load_model_checkpoint
from src.gradcam import compute_gradcam_overlay, get_target_layer


class TestGradCAM(unittest.TestCase):

    def setUp(self):
        # Create synthetic 224x224 RGB image
        self.image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    def test_gradcam_all_architectures(self):
        """Tests that Grad-CAM runs without inplace errors or hook crashes on all 3 architectures."""
        for model_name in ["DenseNet121", "ResNet50", "EfficientNet-B0"]:
            model, _ = load_model_checkpoint(model_name, device="cpu")
            orig, overlay, heatmap = compute_gradcam_overlay(
                model=model,
                model_name=model_name,
                image=self.image,
                target_class_idx=1,
                alpha=0.45,
                colormap_name="jet"
            )

            self.assertIsInstance(orig, Image.Image)
            self.assertIsInstance(overlay, Image.Image)
            self.assertIsInstance(heatmap, np.ndarray)
            self.assertEqual(heatmap.shape, (224, 224))
            self.assertTrue(np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0))

    def test_colormaps_compatibility(self):
        """Verifies that all dropdown colormaps work with modern matplotlib."""
        model = build_architecture("ResNet50", num_classes=4)
        for colormap in ["jet", "inferno", "viridis", "magma"]:
            _, overlay, _ = compute_gradcam_overlay(
                model=model,
                model_name="ResNet50",
                image=self.image,
                target_class_idx=0,
                colormap_name=colormap
            )
            self.assertEqual(overlay.size, (224, 224))


if __name__ == "__main__":
    unittest.main()

