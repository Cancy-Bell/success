"""Prediction serialization compatible with the official AURC JSON layout."""

import copy
import json
import os
from typing import Dict, List, Mapping, Sequence


SEQUENCE_FIELDS = (
    "gold_bio",
    "initial_crf_bio",
    "final_bio",
    "official_gold_labels",
    "official_initial_labels",
    "official_pred_labels",
)


def _space_separated(value: object, field_name: str) -> str:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Prediction field {!r} must be a sequence.".format(field_name))
    return " ".join(str(item) for item in value)


def build_aurc_prediction_dict(
    aurc_data: Mapping[str, Sequence[Mapping[str, object]]],
    records: Sequence[Mapping[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    """Merge predictions into official rows and group them by AURC topic.

    Only rows represented in ``records`` are emitted (for example, the 1,200
    In-Domain Test rows).  Their ordering follows the official source file.
    The output intentionally keeps the compact official-row shape: original
    AURC fields plus model WordPieces, gold BIO, Initial CRF BIO, Final BIO,
    and official-label projections.  Rich diagnostics remain in the original
    JSONL prediction files.
    """
    prediction_by_id: Dict[str, Mapping[str, object]] = {}
    for record in records:
        sample_id = str(record.get("id", ""))
        if not sample_id:
            raise ValueError("Every prediction record must contain a non-empty id.")
        if sample_id in prediction_by_id:
            raise ValueError("Duplicate prediction id: {}".format(sample_id))
        prediction_by_id[sample_id] = record

    output: Dict[str, List[Dict[str, object]]] = {
        str(topic): [] for topic in aurc_data
    }
    matched_ids = set()
    for topic, official_rows in aurc_data.items():
        for official_row in official_rows:
            sample_id = str(official_row.get("sentence_hash", ""))
            prediction = prediction_by_id.get(sample_id)
            if prediction is None:
                continue
            prediction_topic = str(prediction.get("topic", topic))
            if prediction_topic != str(topic):
                raise ValueError(
                    "Topic mismatch for {}: official={!r}, prediction={!r}".format(
                        sample_id, topic, prediction_topic
                    )
                )

            wordpieces = prediction.get("sentence_wordpieces", [])
            sequence_lengths = {
                field_name: len(prediction.get(field_name, []))
                for field_name in ("gold_bio", "initial_crf_bio", "final_bio")
            }
            expected_length = len(wordpieces)
            if any(length != expected_length for length in sequence_lengths.values()):
                raise ValueError(
                    "BIO/WordPiece length mismatch for {}: wordpieces={}, {}".format(
                        sample_id, expected_length, sequence_lengths
                    )
                )

            merged = copy.deepcopy(dict(official_row))
            merged["model_sentence_wordpieces"] = _space_separated(
                wordpieces, "sentence_wordpieces"
            )
            for field_name in SEQUENCE_FIELDS:
                merged[field_name] = _space_separated(
                    prediction.get(field_name, []), field_name
                )
            output[str(topic)].append(merged)
            matched_ids.add(sample_id)

    missing_ids = sorted(set(prediction_by_id) - matched_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            "{} prediction ids were not found in official AURC data: {}".format(
                len(missing_ids), preview
            )
        )
    return output


def write_aurc_prediction_json(
    path: str,
    aurc_data: Mapping[str, Sequence[Mapping[str, object]]],
    records: Sequence[Mapping[str, object]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    output = build_aurc_prediction_dict(aurc_data=aurc_data, records=records)
    with open(path, "w", encoding="utf8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
