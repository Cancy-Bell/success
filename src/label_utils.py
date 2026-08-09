#!/usr/bin/env python
"""Shared AURC BIO labels, stance maps, and robust span recovery."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


O = 0
B_PRO = 1
I_PRO = 2
B_CON = 3
I_CON = 4

BIO_LABELS = ["O", "B-Pro", "I-Pro", "B-Con", "I-Con"]
BIO_LABEL_TO_ID = {label: index for index, label in enumerate(BIO_LABELS)}

AU_STANCES = ["Pro", "Con"]
AU_STANCE_TO_ID = {label.lower(): index for index, label in enumerate(AU_STANCES)}

OFFICIAL_LABELS = ["non", "con", "pro"]
OFFICIAL_LABEL_TO_ID = {label: index for index, label in enumerate(OFFICIAL_LABELS)}

DOCUMENT_LABELS = ["non", "con", "pro"]
DOCUMENT_LABEL_TO_ID = {label: index for index, label in enumerate(DOCUMENT_LABELS)}


def bio_id_to_stance(label_id: int) -> Optional[str]:
    if int(label_id) in (B_PRO, I_PRO):
        return "Pro"
    if int(label_id) in (B_CON, I_CON):
        return "Con"
    return None


def collapse_bio_id(label_id: int) -> str:
    """Collapse five-class BIO to the official three AURC labels."""
    stance = bio_id_to_stance(int(label_id))
    return "non" if stance is None else stance.lower()


def collapse_bio_sequence(label_ids: Sequence[int]) -> List[str]:
    return [collapse_bio_id(label_id) for label_id in label_ids]


def repair_bio_sequence(label_ids: Sequence[int]) -> List[int]:
    """Repair illegal I-tags by deterministically treating them as B-tags.

    The CRF is transition-constrained, but this fixed repair policy also makes
    externally supplied paths and legacy checkpoints safe.
    """
    repaired: List[int] = []
    previous_stance: Optional[str] = None
    for raw_label in label_ids:
        label = int(raw_label)
        stance = bio_id_to_stance(label)
        if label == I_PRO and previous_stance != "Pro":
            label = B_PRO
        elif label == I_CON and previous_stance != "Con":
            label = B_CON
        repaired.append(label)
        previous_stance = bio_id_to_stance(label)
    return repaired


def bio_to_spans(label_ids: Sequence[int]) -> List[Dict[str, object]]:
    """Recover half-open WordPiece AU spans from a BIO path.

    Each result has ``start``, ``end`` (exclusive), and ``stance``. Invalid
    I-tags are first repaired with :func:`repair_bio_sequence`.
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
        is_begin = label in (B_PRO, B_CON)
        if stance is None:
            close(index)
            continue
        if is_begin or current_stance != stance:
            close(index)
            current_start = index
            current_stance = stance
    close(len(labels))
    return spans


def official_labels_to_bio(
    official_labels: Sequence[str],
    wordpiece_to_original_token: Sequence[int],
) -> List[int]:
    """Propagate official token labels to WordPieces as five-class BIO.

    For a B-labeled original token split into multiple WordPieces, only the
    first WordPiece receives B and all remaining pieces receive I.
    """
    seen_original_tokens = set()
    bio_labels: List[int] = []
    for original_index in wordpiece_to_original_token:
        if original_index < 0 or original_index >= len(official_labels):
            bio_labels.append(O)
            continue
        label = str(official_labels[original_index]).lower()
        if label not in ("pro", "con"):
            bio_labels.append(O)
            seen_original_tokens.add(original_index)
            continue

        first_piece = original_index not in seen_original_tokens
        segment_start = (
            original_index == 0
            or str(official_labels[original_index - 1]).lower() != label
        )
        if label == "pro":
            bio_labels.append(B_PRO if first_piece and segment_start else I_PRO)
        else:
            bio_labels.append(B_CON if first_piece and segment_start else I_CON)
        seen_original_tokens.add(original_index)
    return bio_labels

