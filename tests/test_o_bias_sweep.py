import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_o_bias_sweep", ROOT / "run_o_bias_sweep.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OBiasSweepTests(unittest.TestCase):
    def test_bias_directory_name_is_stable_and_windows_safe(self):
        self.assertEqual(MODULE.bias_directory_name(0.325), "o_bias_0_325")
        self.assertEqual(MODULE.bias_directory_name(-0.25), "o_bias_minus_0_25")

    def test_comparison_uses_best_checkpoint_metrics(self):
        experiment = {
            "initial_o_bias": 0.325,
            "status": "completed",
            "output_dir": "example",
            "final_best_metrics": {
                "best_epoch": 7,
                "splits": {
                    "dev": {
                        "official_token_macro_f1": 0.71,
                        "final_o_bias": 0.321,
                    },
                    "test": {
                        "official_token_macro_f1": 0.70,
                        "official_segment_f1": 0.68,
                    },
                },
            },
        }
        row = MODULE.comparison_rows([experiment])[0]
        self.assertEqual(row["best_epoch"], 7)
        self.assertEqual(row["dev_official_token_macro_f1"], 0.71)
        self.assertEqual(row["test_official_token_macro_f1"], 0.70)
        self.assertEqual(row["learned_o_bias"], 0.321)

    def test_sweep_command_must_include_initial_o_bias(self):
        MODULE.validate_o_bias_command(["python", "train.py", "--initial_o_bias", "0.325"], 0.325)
        with self.assertRaisesRegex(ValueError, "missing --initial_o_bias"):
            MODULE.validate_o_bias_command(["python", "train.py"], 0.325)


if __name__ == "__main__":
    unittest.main()
