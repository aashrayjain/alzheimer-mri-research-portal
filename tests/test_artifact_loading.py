import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    get_portal_config,
    get_controlled_config,
    MODEL_CHECKPOINTS,
    DEFAULT_CLASSES
)
from src.evaluation_loader import (
    load_model_comparison_df,
    load_evaluation_details,
    get_confusion_matrix_for_model,
    get_classification_report_df
)


class TestArtifactLoading(unittest.TestCase):

    def test_configs_exist_and_parse(self):
        portal_cfg = get_portal_config()
        controlled_cfg = get_controlled_config()

        self.assertIn("class_names", portal_cfg)
        self.assertEqual(portal_cfg["class_names"], DEFAULT_CLASSES)
        self.assertEqual(portal_cfg["input_size"], 224)
        self.assertEqual(controlled_cfg["seed"], 42)
        self.assertEqual(len(controlled_cfg["train_counts"]), 4)

    def test_checkpoint_files_exist(self):
        for name, path in MODEL_CHECKPOINTS.items():
            self.assertTrue(path.is_file(), f"Missing checkpoint for {name}: {path}")

    def test_comparison_csv_loading(self):
        df = load_model_comparison_df()
        self.assertEqual(len(df), 3)
        self.assertIn("model", df.columns)
        self.assertIn("validation_accuracy", df.columns)

    def test_confusion_matrices_shape(self):
        for model_name in ["ResNet50", "DenseNet121", "EfficientNet-B0"]:
            cm = get_confusion_matrix_for_model(model_name, normalize=False)
            self.assertEqual(cm.shape, (4, 4))
            self.assertEqual(cm.sum(), 960)  # 960 validation samples

            cm_norm = get_confusion_matrix_for_model(model_name, normalize=True)
            self.assertEqual(cm_norm.shape, (4, 4))

    def test_classification_reports(self):
        for model_name in ["ResNet50", "DenseNet121", "EfficientNet-B0"]:
            df_rep = get_classification_report_df(model_name)
            self.assertGreaterEqual(len(df_rep), 4)


if __name__ == "__main__":
    unittest.main()

