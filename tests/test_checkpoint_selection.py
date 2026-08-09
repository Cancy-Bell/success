import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_AURC_token", ROOT / "src" / "run_AURC_token.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CheckpointSelectionTests(unittest.TestCase):
    def test_dev_checkpoint_score_uses_official_token_f1_only(self):
        split_results = {
            "dev": {
                "official_token_macro_f1": 0.71,
                "final_bio_token_macro_f1": 0.99,
            }
        }
        self.assertEqual(MODULE.dev_checkpoint_score(split_results), 0.71)

    def test_new_checkpoint_selection_metadata(self):
        metric, score = MODULE.checkpoint_selection_metadata(
            {
                "selection_metric": "dev_official_token_macro_f1",
                "selection_score": 0.72,
                "dev_official_token_f1": 0.72,
            }
        )
        self.assertEqual(metric, "dev_official_token_macro_f1")
        self.assertEqual(score, 0.72)

    def test_legacy_checkpoint_selection_metadata_is_not_mislabeled(self):
        metric, score = MODULE.checkpoint_selection_metadata({"dev_token_f1": 0.61})
        self.assertEqual(metric, "dev_final_bio_token_macro_f1")
        self.assertEqual(score, 0.61)


if __name__ == "__main__":
    unittest.main()
