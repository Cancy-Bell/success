#!/usr/bin/env python
"""Shared three-class AURC token labels, stance maps, and span recovery."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


NON = 0
CON = 1
PRO = 2

# Keep the historical names as compatibility aliases for prediction files and
# analysis utilities.  The model itself is now three-class throughout.
O = NON
BIO_LABELS = ["non", "con", "pro"]
BIO_LABEL_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}

AU_STANCES = ["Pro", "Con"]
AU_STANCE_TO_ID = {label.lower(): index for index, label in enumerate(AU_STANCES)}

OFFICIAL_LABELS = ["non", "con", "pro"]
OFFICIAL_LABEL_TO_ID = {label: index for index, label in enumerate(OFFICIAL_LABELS)}

DOCUMENT_LABELS = ["non", "con", "pro"]
DOCUMENT_LABEL_TO_ID = {label: index for index, label in enumerate(DOCUMENT_LABELS)}


def bio_id_to_stance(label_id: int) -> Optional[str]:
    if int(label_id) == PRO:
        return "Pro"
    if int(label_id) == CON:
        return "Con"
    return None


def collapse_bio_id(label_id: int) -> str:
    """Return the official non/con/pro label for a three-class token id."""
    label_id = int(label_id)
    if not 0 <= label_id < len(OFFICIAL_LABELS):
        raise ValueError("unknown three-class token label id: {}".format(label_id))
    return OFFICIAL_LABELS[label_id]


def collapse_bio_sequence(label_ids: Sequence[int]) -> List[str]:
    return [collapse_bio_id(label_id) for label_id in label_ids]


def repair_bio_sequence(label_ids: Sequence[int]) -> List[int]:
    """Validate and return a three-class non/con/pro token sequence.

    The legacy function name is retained so existing callers and JSON tooling
    continue to work after removal of B/I labels.
    """
    repaired = [int(label) for label in label_ids]
    invalid = [label for label in repaired if label not in (NON, CON, PRO)]
    if invalid:
        raise ValueError("invalid three-class token labels: {}".format(invalid[:5]))
    return repaired


def bio_to_spans(label_ids: Sequence[int]) -> List[Dict[str, object]]:
    """Recover AU spans as contiguous runs of equal non-neutral stance.

    Without B/I labels, adjacent AUs with the same stance are necessarily one
    run. Each result has ``start``, ``end`` (exclusive), and ``stance``.
    """
    labels = repair_bio_sequence(label_ids)
    spans: List[Dict[str, object]] = []
    current_start: Optional[int] = None
    current_stance: Optional[str] = None

    def close(end: int) -> None:
        nonlocal current_start, current_stance
        if current_start is not None and current_stance is not None:
            spans.append({"start": current_start, "end": end, "stance": current_stance})
        current_start = None
        current_stance = None

    for index, label in enumerate(labels):
        stance = bio_id_to_stance(label)
        if stance is None:
            close(index)
            continue
        if current_stance != stance:
            close(index)
            current_start = index
            current_stance = stance
    close(len(labels))
    return spans


def official_labels_to_bio(
    official_labels: Sequence[str],
    wordpiece_to_original_token: Sequence[int],
) -> List[int]:
    """Propagate official non/con/pro token labels directly to WordPieces."""
    bio_labels: List[int] = []
    for original_index in wordpiece_to_original_token:
        if original_index < 0 or original_index >= len(official_labels):
            bio_labels.append(O)
            continue
        label = str(official_labels[original_index]).lower()
        bio_labels.append(OFFICIAL_LABEL_TO_ID.get(label, NON))
    return bio_labels

