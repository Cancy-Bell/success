import os
import sys
import unittest

import numpy as np


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from label_utils import B_CON, B_PRO, I_PRO, O, bio_to_spans, official_labels_to_bio
from metrics_utils import compute_all_metrics, official_segment_f1
from syntax_utils import map_spacy_to_wordpieces


class CorePipelineTests(unittest.TestCase):
    def test_wordpiece_mapping_is_matrix_product(self):
        adjacency_spacy = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        alignment_matrix = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        expected = alignment_matrix @ adjacency_spacy @ alignment_matrix.T
        actual = map_spacy_to_wordpieces(adjacency_spacy, alignment_matrix)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (4, 4))

    def test_official_labels_propagate_to_wordpiece_bio(self):
        labels = ["pro", "con", "non"]
        mapping = [0, 0, 1, 2]
        self.assertEqual(
            official_labels_to_bio(labels, mapping),
            [B_PRO, I_PRO, B_CON, O],
        )

    def test_illegal_i_tag_starts_a_new_span(self):
        spans = bio_to_spans([I_PRO, I_PRO, O])
        self.assertEqual(spans, [{"start": 0, "end": 2, "stance": "Pro"}])

    def test_entity_requires_exact_span_and_stance(self):
        record = {
            "id": "sample",
            "gold_bio_ids": [B_PRO, I_PRO],
            "pred_bio_ids": [B_PRO, I_PRO],
            "gold_document_stance": "pro",
            "pred_document_stance": "pro",
            "pred_argument_units": [
                {"start": 0, "end": 2, "final_stance": "Con"}
            ],
        }
        result = compute_all_metrics([record])
        self.assertEqual(result["au_span_f1"], 1.0)
        self.assertEqual(result["entity_f1"], 0.0)
        self.assertEqual(result["au_stance_macro_f1"], 0.0)

    def test_official_segment_overlap_is_strictly_greater_than_half(self):
        gold = [["pro", "pro", "non", "non"]]
        accepted = [["pro", "pro", "non", "non"]]
        # One-token overlap over two-token segments is exactly 0.5 and must fail.
        rejected = [["non", "pro", "pro", "non"]]
        self.assertEqual(official_segment_f1(gold, accepted), 1.0)
        self.assertEqual(official_segment_f1(gold, rejected), 0.0)

    def test_initial_final_bio_and_graph_stance_metrics_are_separate(self):
        record = {
            "id": "feedback",
            "gold_bio_ids": [B_PRO, I_PRO, O, B_CON],
            "initial_bio_ids": [O, O, O, O],
            "pred_bio_ids": [B_PRO, I_PRO, O, B_CON],
            "graph_official_labels": ["pro", "pro", "non", "pro"],
            "fused_official_labels": ["pro", "pro", "non", "con"],
            "gold_document_stance": "pro",
            "pred_document_stance": "pro",
            "pred_argument_units": [
                {"start": 0, "end": 2, "final_stance": "Pro"},
                {"start": 3, "end": 4, "final_stance": "Con"},
            ],
            "au_stance_predictions": [
                {"start": 0, "end": 2, "predicted_stance": "Pro"},
                {"start": 3, "end": 4, "predicted_stance": "Pro"},
            ],
        }
        result = compute_all_metrics([record])
        self.assertGreater(
            result["final_bio_token_macro_f1"],
            result["initial_bio_token_macro_f1"],
        )
        self.assertEqual(result["au_stance_accuracy"], 0.5)
        self.assertEqual(result["au_stance_matched_count"], 2)
        self.assertAlmostEqual(result["initial_official_token_macro_f1"], 2.0 / 15.0)
        self.assertEqual(result["final_official_token_macro_f1"], 1.0)
        self.assertLess(
            result["graph_official_token_macro_f1"],
            result["fused_official_token_macro_f1"],
        )
        self.assertAlmostEqual(result["all_stance_token_accuracy"], 0.75)
        self.assertAlmostEqual(result["argument_stance_token_accuracy"], 2.0 / 3.0)
        self.assertEqual(result["argument_stance_token_count"], 3)
        self.assertEqual(result["gold_au_stance_count"], 2)
        self.assertEqual(result["gold_au_stance_accuracy"], 0.5)

    def test_au_identification_is_token_level_not_exact_span(self):
        record = {
            "id": "token-au",
            "gold_bio_ids": [B_PRO, I_PRO, O, O],
            "initial_bio_ids": [O, O, O, O],
            "pred_bio_ids": [O, B_PRO, I_PRO, O],
            "graph_official_labels": ["non", "pro", "pro", "non"],
            "gold_document_stance": "pro",
            "pred_document_stance": "pro",
            "pred_argument_units": [
                {"start": 1, "end": 3, "final_stance": "Pro"},
            ],
        }
        result = compute_all_metrics([record])
        self.assertEqual(result["au_span_f1"], 0.0)
        self.assertGreater(result["au_token_f1"], 0.0)
        self.assertEqual(result["au_stance_matched_count"], 0)
        self.assertGreater(result["argument_stance_token_count"], 0)
        self.assertGreater(result["argument_stance_token_macro_f1"], 0.0)
        self.assertEqual(result["gold_au_stance_count"], 1)
        self.assertEqual(result["gold_au_stance_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
