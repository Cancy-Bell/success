import os
import sys
import unittest


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    import torch
    from transformers import BertConfig

    from graph_utils import build_batch_heterogeneous_edges
    from label_utils import B_CON, B_PRO, I_PRO, O
    from models import TokenBERT
    from run_AURC_token import build_prediction_record

    DEPENDENCIES_AVAILABLE = True
except (ImportError, OSError):
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "torch/transformers unavailable")
class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            num_labels=5,
        )
        self.model = TokenBERT(
            num_labels=5,
            model_name="unused",
            config=config,
            initialize_from_pretrained=False,
            gcn_layers=2,
            gcn_dropout=0.0,
            hetgat_layers=1,
            hetgat_heads=1,
            hetgat_dropout=0.0,
        )
        batch_size, full_length, sentence_length = 2, 10, 6
        self.batch = {
            "input_ids": torch.randint(0, 100, (batch_size, full_length)),
            "attention_mask": torch.ones(batch_size, full_length, dtype=torch.long),
            "token_type_ids": torch.tensor(
                [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]] * batch_size,
                dtype=torch.long,
            ),
            "topic_input_ids": torch.randint(0, 100, (batch_size, full_length)),
            "topic_attention_mask": torch.tensor(
                [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] * batch_size,
                dtype=torch.long,
            ),
            "topic_token_type_ids": torch.zeros(
                batch_size, full_length, dtype=torch.long
            ),
            "sentence_indices": torch.tensor(
                [[4, 5, 6, 7, 0, 0]] * batch_size, dtype=torch.long
            ),
            "sentence_mask": torch.tensor(
                [[True, True, True, True, False, False]] * batch_size
            ),
            "dependency_adj_wordpiece": torch.eye(sentence_length)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
            "topics": ["same-topic", "same-topic"],
            "labels": torch.tensor(
                [
                    [B_PRO, I_PRO, O, B_CON, O, O],
                    [O, O, O, O, O, O],
                ],
                dtype=torch.long,
            ),
            "document_labels": torch.tensor([2, 0], dtype=torch.long),
        }

    def test_complete_three_loss_model_runs_from_first_epoch(self):
        self.model.train()
        output = self.model(**self.batch)
        expected = output["bio_loss"] + output["au_loss"] + output["document_loss"]
        self.assertTrue(torch.allclose(output["total_loss"], expected))
        expected_bio = (
            output["final_bio_loss"]
            + 0.3 * output["initial_bio_loss"]
            + 0.5 * output["official_token_loss"]
        )
        self.assertTrue(torch.allclose(output["bio_loss"], expected_bio))
        self.assertEqual(tuple(output["initial_bio_emissions"].shape), (2, 6, 5))
        self.assertEqual(tuple(output["final_bio_emissions"].shape), (2, 6, 5))
        self.assertIn("final_o_bias", output)
        self.assertTrue(output["final_o_bias"].requires_grad)
        self.assertFalse(
            torch.allclose(
                output["initial_bio_emissions"], output["final_bio_emissions"]
            )
        )
        self.assertTrue(
            all(sample["graph_source"] == "gold_bio" for sample in output["sample_outputs"])
        )
        output["total_loss"].backward()
        self.assertTrue(any(p.grad is not None for p in self.model.topic_bert.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.model.dependency_gcn.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.model.hetgat.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.model.feedback_gate.parameters()))
        self.assertIsNotNone(self.model.final_o_bias.grad)

    def test_official_auxiliary_loss_collapses_bio_to_non_con_pro(self):
        emissions = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 4.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0, 3.0],
                ]
            ]
        )
        labels = torch.tensor([[O, B_PRO, B_CON]], dtype=torch.long)
        mask = torch.ones((1, 3), dtype=torch.bool)
        correct_loss = self.model._official_token_loss(emissions, labels, mask)
        wrong_labels = torch.tensor([[B_PRO, B_CON, O]], dtype=torch.long)
        wrong_loss = self.model._official_token_loss(emissions, wrong_labels, mask)
        self.assertLess(float(correct_loss), float(wrong_loss))

    def test_final_bio_loss_backpropagates_through_reasoning_feedback(self):
        self.model.train()
        output = self.model(**self.batch)
        output["bio_loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool((parameter.grad != 0).any())
                for parameter in self.model.hetgat.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and bool((parameter.grad != 0).any())
                for parameter in self.model.au_stance_classifier.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and bool((parameter.grad != 0).any())
                for parameter in self.model.document_classifier.parameters()
            )
        )

    def test_dual_bert_and_batch_graph_have_no_topic_nodes(self):
        self.assertIsNot(self.model.tokenbert.bert, self.model.topic_bert)
        self.model.train()
        output = self.model(**self.batch)
        debug = output["sample_outputs"][0]["graph_debug"]
        self.assertEqual(debug["node_order"], "all_aus_then_documents")
        self.assertEqual(debug["au_count"], 2)
        self.assertEqual(debug["document_count"], 2)
        self.assertEqual(debug["document_topic_edges"], 2)
        relation_names = set(self.model.hetgat.layers[0].relation_types)
        self.assertEqual(
            relation_names,
            {"self", "au_to_au", "au_to_doc", "doc_to_au", "doc_to_doc"},
        )

    def test_semantic_and_syntax_evidence_share_one_au_relation(self):
        au_states = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        records = [
            {"sample_index": 0, "span": {"start": 0, "end": 1}},
            {"sample_index": 0, "span": {"start": 1, "end": 2}},
        ]
        edges, debug = build_batch_heterogeneous_edges(
            au_representations=au_states,
            au_records=records,
            document_count=1,
            topics=["topic"],
            dependency_adj_wordpiece=torch.ones(1, 2, 2),
            semantic_threshold=0.5,
            top_k=3,
            syntax_hops=1,
        )
        self.assertIn("au_to_au", edges)
        self.assertNotIn("au_semantic", edges)
        self.assertNotIn("au_syntax", edges)
        self.assertTrue(bool((edges["au_to_au"]["edge_weight"] > 2.9).all()))
        self.assertTrue(
            all(edge["semantic_edge"] and edge["syntax_edge"] for edge in debug["au_au_edges"])
        )

    def test_au_masked_attention_and_feedback_outputs(self):
        self.model.train()
        output = self.model(**self.batch)
        sample = output["sample_outputs"][0]
        first_weights = sample["au_attention_weights"][0]
        self.assertAlmostEqual(float(first_weights[:2].sum()), 1.0, places=6)
        self.assertEqual(float(first_weights[2:].sum()), 0.0)
        self.assertEqual(tuple(sample["feedback_gate"].shape), (4, 5))
        self.assertTrue(bool(((sample["feedback_gate"] >= 0) & (sample["feedback_gate"] <= 1)).all()))
        self.assertEqual(tuple(sample["initial_official_probs"].shape), (4, 3))
        self.assertEqual(tuple(sample["graph_official_probs"].shape), (4, 3))
        self.assertEqual(tuple(sample["fused_official_probs"].shape), (4, 3))
        self.assertEqual(tuple(sample["stance_fusion_weights"].shape), (4, 2))
        self.assertTrue(
            torch.allclose(
                sample["stance_fusion_weights"].sum(dim=-1),
                torch.ones(4),
                atol=1e-6,
            )
        )

    def test_eval_uses_initial_predicted_bio_for_graph(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(**self.batch)
        self.assertTrue(
            all(
                sample["graph_source"] == "predicted_bio"
                for sample in output["sample_outputs"]
            )
        )
        for sample in output["sample_outputs"]:
            self.assertIn("initial_bio_ids", sample)
            self.assertIn("final_bio_ids", sample)
            self.assertIn("au_stance_probs", sample)
            self.assertIn("document_probs", sample)

    def test_batch_with_no_gold_aus_still_trains_bio_and_document(self):
        self.model.train()
        no_au_batch = dict(self.batch)
        no_au_batch["labels"] = torch.zeros_like(self.batch["labels"])
        output = self.model(**no_au_batch)
        self.assertEqual(float(output["au_loss"]), 0.0)
        self.assertEqual(output["sample_outputs"][0]["graph_debug"]["au_count"], 0)
        output["total_loss"].backward()
        self.assertTrue(any(p.grad is not None for p in self.model.document_classifier.parameters()))

    def test_prediction_json_keeps_all_three_bio_layers_and_feedback(self):
        self.model.eval()
        with torch.no_grad():
            sample_output = self.model(**self.batch)["sample_outputs"][0]
        metadata = {
            "id": "tiny",
            "topic": "same-topic",
            "text": "a b c d",
            "sentence_wordpieces": ["a", "b", "c", "d"],
            "wordpiece_offsets": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "wordpiece_to_original_token": [0, 1, 2, 3],
            "gold_document_stance": "pro",
            "alignment_warnings": [],
        }
        record = build_prediction_record(
            metadata=metadata,
            gold_bio_ids=[B_PRO, I_PRO, O, B_CON],
            sample_output=sample_output,
        )
        for key in (
            "gold_bio",
            "initial_crf_bio",
            "final_bio",
            "initial_bio_probs",
            "au_stance_probs",
            "document_stance_probs",
            "feedback_gate",
            "final_bio_probs",
            "graph_official_labels",
            "fused_official_labels",
            "stance_fusion_weights",
        ):
            self.assertIn(key, record)
        self.assertEqual(len(record["initial_crf_bio"]), 4)
        self.assertEqual(len(record["final_bio"]), 4)
        self.assertEqual(record["sentence_length"], 4)


if __name__ == "__main__":
    unittest.main()
