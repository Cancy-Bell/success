import unittest

from src.prediction_io import build_aurc_prediction_dict


class PredictionIOTests(unittest.TestCase):
    def test_official_topic_layout_and_three_bio_sequences(self):
        official = {
            "topic-a": [
                {
                    "In-Domain": "Test",
                    "sentence_hash": "sample-1",
                    "sentence": "A sentence.",
                    "tokenized_sentence_spacy_labels": "pro non",
                }
            ],
            "topic-b": [],
        }
        records = [
            {
                "id": "sample-1",
                "topic": "topic-a",
                "sentence_wordpieces": ["A", "sentence", "."],
                "gold_bio": ["B-Pro", "I-Pro", "O"],
                "initial_crf_bio": ["O", "B-Pro", "O"],
                "final_bio": ["B-Pro", "I-Pro", "O"],
                "official_gold_labels": ["pro", "pro", "non"],
                "official_initial_labels": ["non", "pro", "non"],
                "official_pred_labels": ["pro", "pro", "non"],
                "feedback_gate": [[0.5] * 5] * 3,
                "au_stance_probs": [[0.1, 0.9]],
            }
        ]
        output = build_aurc_prediction_dict(official, records)
        self.assertEqual(list(output), ["topic-a", "topic-b"])
        self.assertEqual(len(output["topic-a"]), 1)
        row = output["topic-a"][0]
        self.assertEqual(row["sentence_hash"], "sample-1")
        self.assertEqual(row["model_sentence_wordpieces"], "A sentence .")
        self.assertEqual(row["gold_bio"], "B-Pro I-Pro O")
        self.assertEqual(row["initial_crf_bio"], "O B-Pro O")
        self.assertEqual(row["final_bio"], "B-Pro I-Pro O")
        self.assertNotIn("feedback_gate", row)
        self.assertNotIn("au_stance_probs", row)
        self.assertEqual(
            list(row),
            [
                "In-Domain",
                "sentence_hash",
                "sentence",
                "tokenized_sentence_spacy_labels",
                "model_sentence_wordpieces",
                "gold_bio",
                "initial_crf_bio",
                "final_bio",
                "official_gold_labels",
                "official_initial_labels",
                "official_pred_labels",
            ],
        )

    def test_length_mismatch_is_rejected(self):
        official = {"topic": [{"sentence_hash": "id"}]}
        records = [
            {
                "id": "id",
                "topic": "topic",
                "sentence_wordpieces": ["one", "two"],
                "gold_bio": ["O"],
                "initial_crf_bio": ["O", "O"],
                "final_bio": ["O", "O"],
            }
        ]
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            build_aurc_prediction_dict(official, records)


if __name__ == "__main__":
    unittest.main()
