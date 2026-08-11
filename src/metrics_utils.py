#!/usr/bin/env python
"""Official AURC and strict AU/entity evaluation metrics."""

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from label_utils import (
    AU_STANCES,
    BIO_LABELS,
    DOCUMENT_LABELS,
    OFFICIAL_LABELS,
    bio_to_spans,
    collapse_bio_sequence,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _prf(true_positive: int, predicted: int, gold: int) -> Tuple[float, float, float]:
    if predicted == 0 and gold == 0:
        return 1.0, 1.0, 1.0
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, gold)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(
    gold_labels: Sequence[object],
    predicted_labels: Sequence[object],
    labels: Sequence[object],
) -> float:
    """Compute explicit-label macro F1 without version-sensitive sklearn defaults."""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("gold/predicted label lengths differ")
    scores = []
    for label in labels:
        true_positive = sum(
            gold == label and predicted == label
            for gold, predicted in zip(gold_labels, predicted_labels)
        )
        false_positive = sum(
            gold != label and predicted == label
            for gold, predicted in zip(gold_labels, predicted_labels)
        )
        false_negative = sum(
            gold == label and predicted != label
            for gold, predicted in zip(gold_labels, predicted_labels)
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        scores.append(_safe_divide(2.0 * precision * recall, precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def _extract_official_segments(labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    segments: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    stance: Optional[str] = None
    for index, raw_label in enumerate(list(labels) + ["non"]):
        label = str(raw_label).lower()
        if label not in ("pro", "con"):
            if start is not None and stance is not None:
                segments.append((start, index, stance))
            start, stance = None, None
        elif stance != label:
            if start is not None and stance is not None:
                segments.append((start, index, stance))
            start, stance = index, label
    return segments


def official_segment_f1(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> float:
    """Paper-defined macro segment F1 with overlap ratio strictly above 0.5.

    Candidate pairs are greedily matched one-to-one from greatest overlap to
    avoid rewarding duplicate predictions. Sentences with neither a gold nor a
    predicted PRO/CON segment receive F1=1, as defined in the AURC paper.
    """
    sentence_scores: List[float] = []
    for gold_labels, predicted_labels in zip(gold_sequences, predicted_sequences):
        gold_segments = _extract_official_segments(gold_labels)
        predicted_segments = _extract_official_segments(predicted_labels)
        candidates = []
        for gold_index, (gold_start, gold_end, gold_stance) in enumerate(gold_segments):
            for pred_index, (pred_start, pred_end, pred_stance) in enumerate(
                predicted_segments
            ):
                overlap = max(0, min(gold_end, pred_end) - max(gold_start, pred_start))
                ratio = _safe_divide(
                    overlap, max(gold_end - gold_start, pred_end - pred_start)
                )
                if ratio > 0.5 and gold_stance == pred_stance:
                    candidates.append((ratio, gold_index, pred_index))
        candidates.sort(reverse=True)
        used_gold, used_predicted = set(), set()
        matches = 0
        for _, gold_index, pred_index in candidates:
            if gold_index in used_gold or pred_index in used_predicted:
                continue
            used_gold.add(gold_index)
            used_predicted.add(pred_index)
            matches += 1
        _, _, sentence_f1 = _prf(
            matches, len(predicted_segments), len(gold_segments)
        )
        sentence_scores.append(sentence_f1)
    return sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0


def official_sentence_prediction(
    token_labels: Sequence[str], sample_id: str = ""
) -> str:
    """Convert official token labels to one sentence label.

    NON is used when no argument label occurs. Otherwise the more frequent of
    PRO/CON is selected. The paper resolves rare ties randomly; a stable hash of
    the sample id provides the same random tie behavior reproducibly.
    """
    pro_count = sum(str(label).lower() == "pro" for label in token_labels)
    con_count = sum(str(label).lower() == "con" for label in token_labels)
    if pro_count == 0 and con_count == 0:
        return "non"
    if pro_count > con_count:
        return "pro"
    if con_count > pro_count:
        return "con"
    digest = hashlib.md5(str(sample_id).encode("utf8")).digest()
    return "pro" if digest[0] % 2 == 0 else "con"


def compute_all_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Compute all mandatory metrics from JSON-like evaluation records."""
    official_gold_sequences: List[List[str]] = []
    official_pred_sequences: List[List[str]] = []
    flattened_official_gold: List[str] = []
    flattened_official_initial: List[str] = []
    flattened_official_pred: List[str] = []
    flattened_official_graph: List[str] = []
    flattened_official_fused: List[str] = []
    flattened_bio_gold: List[str] = []
    flattened_bio_initial: List[str] = []
    flattened_bio_pred: List[str] = []
    official_sentence_gold: List[str] = []
    official_sentence_pred: List[str] = []
    document_gold: List[str] = []
    document_pred: List[str] = []

    span_true_positive = 0
    span_predicted = 0
    span_gold = 0
    entity_true_positive = 0
    entity_predicted = 0
    entity_gold = 0
    stance_gold: List[str] = []
    stance_predicted: List[str] = []
    stance_correct = 0
    gold_au_stance_gold: List[str] = []
    gold_au_stance_predicted: List[str] = []
    gold_au_stance_correct = 0

    for record in records:
        gold_bio_ids = [int(label) for label in record["gold_bio_ids"]]
        pred_bio_ids = [int(label) for label in record["pred_bio_ids"]]
        initial_bio_ids = [
            int(label) for label in record.get("initial_bio_ids", pred_bio_ids)
        ]
        official_gold = collapse_bio_sequence(gold_bio_ids)
        official_pred = collapse_bio_sequence(pred_bio_ids)
        official_initial = collapse_bio_sequence(initial_bio_ids)
        official_graph = list(record.get("graph_official_labels", official_pred))
        official_fused = list(record.get("fused_official_labels", official_pred))
        official_gold_sequences.append(official_gold)
        official_pred_sequences.append(official_pred)
        flattened_official_gold.extend(official_gold)
        flattened_official_initial.extend(official_initial)
        flattened_official_pred.extend(official_pred)
        flattened_official_graph.extend(official_graph)
        flattened_official_fused.extend(official_fused)
        flattened_bio_gold.extend(BIO_LABELS[label] for label in gold_bio_ids)
        flattened_bio_initial.extend(BIO_LABELS[label] for label in initial_bio_ids)
        flattened_bio_pred.extend(BIO_LABELS[label] for label in pred_bio_ids)

        sample_id = str(record.get("id", ""))
        gold_document_stance = str(record["gold_document_stance"]).lower()
        official_sentence_gold.append(gold_document_stance)
        official_sentence_pred.append(
            official_sentence_prediction(official_pred, sample_id=sample_id)
        )
        predicted_document_stance = record.get("pred_document_stance")
        if predicted_document_stance is not None:
            document_gold.append(gold_document_stance)
            document_pred.append(str(predicted_document_stance).lower())

        gold_spans = bio_to_spans(gold_bio_ids)
        predicted_units = list(record.get("pred_argument_units", []))
        graph_au_predictions = list(record.get("au_stance_predictions", []))
        gold_by_span = {
            (int(span["start"]), int(span["end"])): str(span["stance"])
            for span in gold_spans
        }
        for span in gold_spans:
            start, end = int(span["start"]), int(span["end"])
            gold_stance = str(span["stance"])
            token_stances = [
                str(label).lower()
                for label in official_graph[start:end]
                if str(label).lower() in ("pro", "con")
            ]
            pro_count = sum(label == "pro" for label in token_stances)
            con_count = sum(label == "con" for label in token_stances)
            if pro_count > con_count:
                predicted_stance = "Pro"
            elif con_count > pro_count:
                predicted_stance = "Con"
            else:
                predicted_stance = "non"
            gold_au_stance_gold.append(gold_stance)
            gold_au_stance_predicted.append(predicted_stance)
            gold_au_stance_correct += int(gold_stance == predicted_stance)
        predicted_by_span = {
            (int(unit["start"]), int(unit["end"])): str(unit["final_stance"])
            for unit in predicted_units
        }
        gold_span_set = set(gold_by_span)
        predicted_span_set = set(predicted_by_span)
        matching_spans = gold_span_set & predicted_span_set
        span_true_positive += len(matching_spans)
        span_predicted += len(predicted_span_set)
        span_gold += len(gold_span_set)
        graph_stance_by_span = {
            (int(unit["start"]), int(unit["end"])): str(unit["predicted_stance"])
            for unit in graph_au_predictions
        }
        for span in sorted(set(gold_by_span) & set(graph_stance_by_span)):
            gold_stance = gold_by_span[span]
            predicted_stance = graph_stance_by_span[span]
            stance_gold.append(gold_stance)
            stance_predicted.append(predicted_stance)
            stance_correct += int(gold_stance == predicted_stance)

        gold_entities = {
            (span[0], span[1], gold_by_span[span]) for span in gold_span_set
        }
        predicted_entities = {
            (span[0], span[1], predicted_by_span[span])
            for span in predicted_span_set
        }
        entity_true_positive += len(gold_entities & predicted_entities)
        entity_predicted += len(predicted_entities)
        entity_gold += len(gold_entities)

    span_precision, span_recall, span_f1 = _prf(
        span_true_positive, span_predicted, span_gold
    )
    entity_precision, entity_recall, entity_f1 = _prf(
        entity_true_positive, entity_predicted, entity_gold
    )
    initial_bio_f1 = macro_f1(
        flattened_bio_gold, flattened_bio_initial, labels=BIO_LABELS
    )
    final_bio_f1 = macro_f1(
        flattened_bio_gold, flattened_bio_pred, labels=BIO_LABELS
    )
    gold_argument_tokens = [label != "non" for label in flattened_official_gold]
    initial_argument_tokens = [
        label != "non" for label in flattened_official_initial
    ]
    final_argument_tokens = [label != "non" for label in flattened_official_pred]
    initial_argument_tp = sum(
        gold and predicted
        for gold, predicted in zip(gold_argument_tokens, initial_argument_tokens)
    )
    final_argument_tp = sum(
        gold and predicted
        for gold, predicted in zip(gold_argument_tokens, final_argument_tokens)
    )
    initial_argument_precision, initial_argument_recall, initial_argument_f1 = _prf(
        initial_argument_tp,
        sum(initial_argument_tokens),
        sum(gold_argument_tokens),
    )
    final_argument_precision, final_argument_recall, final_argument_f1 = _prf(
        final_argument_tp,
        sum(final_argument_tokens),
        sum(gold_argument_tokens),
    )
    initial_official_f1 = macro_f1(
        flattened_official_gold,
        flattened_official_initial,
        labels=OFFICIAL_LABELS,
    )
    final_official_f1 = macro_f1(
        flattened_official_gold,
        flattened_official_pred,
        labels=OFFICIAL_LABELS,
    )
    all_stance_correct = sum(
        gold == predicted
        for gold, predicted in zip(flattened_official_gold, flattened_official_graph)
    )
    all_stance_accuracy = _safe_divide(
        all_stance_correct, len(flattened_official_gold)
    )
    argument_stance_pairs = [
        (gold, predicted)
        for gold, predicted in zip(flattened_official_gold, flattened_official_graph)
        if gold != "non" or predicted != "non"
    ]
    if argument_stance_pairs:
        argument_stance_gold = [gold for gold, _ in argument_stance_pairs]
        argument_stance_predicted = [
            predicted for _, predicted in argument_stance_pairs
        ]
    else:
        argument_stance_gold = []
        argument_stance_predicted = []
    argument_stance_correct = sum(
        gold == predicted
        for gold, predicted in zip(argument_stance_gold, argument_stance_predicted)
    )
    return {
        "official_token_macro_f1": final_official_f1,
        "initial_official_token_macro_f1": initial_official_f1,
        "final_official_token_macro_f1": final_official_f1,
        "graph_official_token_macro_f1": macro_f1(
            flattened_official_gold,
            flattened_official_graph,
            labels=OFFICIAL_LABELS,
        ),
        "fused_official_token_macro_f1": macro_f1(
            flattened_official_gold,
            flattened_official_fused,
            labels=OFFICIAL_LABELS,
        ),
        "official_token_f1_delta": final_official_f1 - initial_official_f1,
        "official_segment_f1": official_segment_f1(
            official_gold_sequences, official_pred_sequences
        ),
        "official_sentence_f1": macro_f1(
            official_sentence_gold,
            official_sentence_pred,
            labels=OFFICIAL_LABELS,
        ),
        "all_stance_token_macro_f1": macro_f1(
            flattened_official_gold,
            flattened_official_graph,
            labels=OFFICIAL_LABELS,
        ),
        "all_stance_token_accuracy": all_stance_accuracy,
        "argument_stance_token_macro_f1": macro_f1(
            argument_stance_gold,
            argument_stance_predicted,
            labels=OFFICIAL_LABELS,
        ),
        "argument_stance_token_accuracy": _safe_divide(
            argument_stance_correct, len(argument_stance_gold)
        ),
        "argument_stance_token_count": len(argument_stance_gold),
        "gold_au_stance_macro_f1": macro_f1(
            gold_au_stance_gold,
            gold_au_stance_predicted,
            labels=AU_STANCES,
        ),
        "gold_au_stance_accuracy": _safe_divide(
            gold_au_stance_correct, len(gold_au_stance_gold)
        ),
        "gold_au_stance_count": len(gold_au_stance_gold),
        "au_span_precision": span_precision,
        "au_span_recall": span_recall,
        "au_span_f1": span_f1,
        "initial_au_token_precision": initial_argument_precision,
        "initial_au_token_recall": initial_argument_recall,
        "initial_au_token_f1": initial_argument_f1,
        "final_au_token_precision": final_argument_precision,
        "final_au_token_recall": final_argument_recall,
        "final_au_token_f1": final_argument_f1,
        "au_token_precision": final_argument_precision,
        "au_token_recall": final_argument_recall,
        "au_token_f1": final_argument_f1,
        "au_stance_macro_f1": macro_f1(
            stance_gold, stance_predicted, labels=AU_STANCES
        ),
        "au_stance_accuracy": _safe_divide(stance_correct, len(stance_gold)),
        "au_stance_matched_count": len(stance_gold),
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "document_stance_macro_f1": macro_f1(
            document_gold, document_pred, labels=DOCUMENT_LABELS
        ),
        "initial_bio_token_macro_f1": initial_bio_f1,
        "bio_token_macro_f1": final_bio_f1,
        "final_bio_token_macro_f1": final_bio_f1,
        "feedback_token_f1_delta": final_bio_f1 - initial_bio_f1,
    }
