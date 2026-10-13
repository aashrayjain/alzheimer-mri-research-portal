import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_and_validate_image,
    preprocess_image,
    get_inference_transform
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a dummy synthetic RGB and Grayscale image in memory
        self.rgb_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        self.gray_image = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L")

    def test_rgb_conversion(self):
        proc_rgb = load_and_validate_image(self.rgb_image)
        proc_gray = load_and_validate_image(self.gray_image)

        self.assertEqual(proc_rgb.mode, "RGB")
        self.assertEqual(proc_gray.mode, "RGB")

    def test_corrupted_image_handling(self):
        with self.assertRaises(ValueError):
            load_and_validate_image(b"not_an_image_binary_data")

    def test_preprocess_tensor_shape(self):
        tensor = preprocess_image(self.rgb_image)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()

