import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predict_mri
from src.config import DEFAULT_CLASSES


class DummyModel(nn.Module):
    """Mock model outputting constant logits for deterministic test."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 4)

    def forward(self, x):
        # Return deterministic logits: index 1 highest (ModerateDemented)
        return torch.tensor([[1.0, 5.0, 2.0, 0.5]], dtype=torch.float32)


class TestInference(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        self.image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_predict_mri_output_structure(self):
        results = predict_mri(self.model, self.image, DEFAULT_CLASSES, device="cpu")

        self.assertEqual(results["predicted_class"], "ModerateDemented")
        self.assertIn("class_probabilities", results)
        self.assertEqual(len(results["class_probabilities"]), 4)

        probs = list(results["class_probabilities"].values())
        self.assertAlmostEqual(sum(probs), 1.0, places=4)
        self.assertGreater(results["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()

