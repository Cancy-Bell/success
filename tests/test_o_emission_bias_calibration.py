import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_o_emission_bias", ROOT / "calibrate_o_emission_bias.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OEmissionBiasCalibrationTests(unittest.TestCase):
    def test_bias_grid_uses_decimal_steps(self):
        self.assertEqual(
            MODULE.build_bias_grid(0.0, 0.3, 0.1), [0.0, 0.1, 0.2, 0.3]
        )

    def test_positive_o_bias_can_change_viterbi_path_to_o(self):
        crf = MODULE.LinearChainCRF(5)
        with torch.no_grad():
            crf.start_transitions.zero_()
            crf.end_transitions.zero_()
            crf.transitions.zero_()
        record = {
            "_log_prob_emissions": torch.log(
                torch.tensor([[0.40, 0.45, 0.05, 0.05, 0.05]], dtype=torch.float32)
            )
        }
        no_bias = MODULE.decode_with_bias([record], crf, bias=0.0, batch_size=1)
        positive_bias = MODULE.decode_with_bias([record], crf, bias=0.2, batch_size=1)
        self.assertEqual(no_bias, [[1]])
        self.assertEqual(positive_bias, [[0]])

    def test_log_probabilities_preserve_crf_path_at_zero_bias(self):
        crf = MODULE.LinearChainCRF(5)
        emissions = torch.tensor(
            [[[0.5, 0.7, -0.2, 0.1, -0.3], [0.2, -0.1, 0.8, 0.0, 0.1]]]
        )
        mask = torch.ones((1, 2), dtype=torch.bool)
        expected = crf.decode(emissions, mask)
        log_probabilities = torch.log_softmax(emissions[0], dim=-1)
        actual = MODULE.decode_with_bias(
            [{"_log_prob_emissions": log_probabilities}],
            crf,
            bias=0.0,
            batch_size=1,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
