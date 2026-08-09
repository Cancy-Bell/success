import os
import sys
import unittest


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from data_utils import (
        align_wordpieces_to_original_tokens,
        locate_pretokenized_offsets,
    )

    DEPENDENCIES_AVAILABLE = True
except (ImportError, OSError):
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "torch unavailable")
class AlignmentRegressionTests(unittest.TestCase):
    def test_legacy_dash_does_not_cause_cascading_offset_drift(self):
        text = "Using a longer time frame\x9716½ years\x97than that"
        tokens = "Using a longer time frame - 16½ years - than that".split()
        offsets, warnings = locate_pretokenized_offsets(text, tokens, "dash")
        self.assertEqual(warnings, [])
        self.assertTrue(all(start >= 0 and end > start for start, end in offsets))
        self.assertEqual(text[offsets[-1][0] : offsets[-1][1]], "that")

    def test_mojibake_apostrophe_is_aligned(self):
        text = "There itâ\x80\x99s done"
        tokens = ["There", "it", "'s", "done"]
        offsets, warnings = locate_pretokenized_offsets(text, tokens, "apostrophe")
        self.assertEqual(warnings, [])
        self.assertTrue(all(start >= 0 for start, _ in offsets))

    def test_unannotated_raw_substring_is_o_not_nearest_stance(self):
        mapping, warnings = align_wordpieces_to_original_tokens(
            wordpiece_offsets=[(0, 8), (9, 12)],
            original_offsets=[(0, 8)],
            sample_id="url",
        )
        self.assertEqual(mapping, [0, -1])
        self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
