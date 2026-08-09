#!/usr/bin/env python
"""Dependency parsing and spaCy-to-WordPiece graph alignment helpers.

The two operations in this module are intentionally kept separate:

1. ``A_wordpiece = M @ A_spacy @ M.T`` maps a dependency graph between
   tokenizer granularities.
2. GCN adjacency normalization is performed later, inside ``models.py``.

Keeping the operations separate prevents the mapping matrix from being
mistaken for a graph-normalization step.
"""

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


def _span_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    """Return the number of overlapping characters in two half-open spans."""
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def build_spacy_dependency_adjacency(doc) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Build a bidirectional spaCy dependency adjacency with self-loops.

    Parameters
    ----------
    doc:
        A parsed spaCy ``Doc`` for the raw sentence only. Topic text and BERT
        special tokens must never be passed here.

    Returns
    -------
    adjacency:
        ``A_spacy`` with shape ``[num_spacy_tokens, num_spacy_tokens]``.
    edges:
        Directed dependency edges before WordPiece mapping (self-loops omitted
        from this human-readable list).
    """
    token_count = len(doc)
    adjacency = np.eye(token_count, dtype=np.float32)
    edges: List[Tuple[int, int]] = []
    for token in doc:
        head_index = int(token.head.i)
        token_index = int(token.i)
        if head_index == token_index:
            continue
        adjacency[token_index, head_index] = 1.0
        adjacency[head_index, token_index] = 1.0
        edges.append((token_index, head_index))
        edges.append((head_index, token_index))
    return adjacency, edges


def build_alignment_matrix(
    wordpiece_offsets: Sequence[Tuple[int, int]],
    spacy_offsets: Sequence[Tuple[int, int]],
    sample_id: str = "",
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Align sentence WordPieces to spaCy tokens by raw-text offsets.

    ``M[k, i] == 1`` means sentence WordPiece ``k`` belongs to spaCy token
    ``i``. A maximum-overlap assignment is used when normalization creates an
    ambiguous offset. If no overlap exists, the nearest spaCy token midpoint is
    selected and a warning is returned rather than crashing the whole run.
    """
    wp_count = len(wordpiece_offsets)
    spacy_count = len(spacy_offsets)
    alignment_matrix = np.zeros((wp_count, spacy_count), dtype=np.float32)
    wp_to_spacy = [-1] * wp_count
    warnings: List[str] = []

    if wp_count == 0:
        return alignment_matrix, wp_to_spacy, warnings
    if spacy_count == 0:
        warning = "sample={} has WordPieces but no spaCy tokens".format(sample_id)
        LOGGER.warning(warning)
        warnings.append(warning)
        return alignment_matrix, wp_to_spacy, warnings

    spacy_midpoints = [0.5 * (start + end) for start, end in spacy_offsets]
    for wp_index, wp_span in enumerate(wordpiece_offsets):
        overlaps = [_span_overlap(wp_span, spacy_span) for spacy_span in spacy_offsets]
        best_overlap = max(overlaps)
        if best_overlap > 0:
            # Prefer the first maximum. Proper tokenizations should have a
            # unique maximum; this deterministic tie-break handles punctuation
            # normalization without making the run nondeterministic.
            spacy_index = overlaps.index(best_overlap)
        else:
            wp_midpoint = 0.5 * (wp_span[0] + wp_span[1])
            spacy_index = min(
                range(spacy_count),
                key=lambda index: abs(spacy_midpoints[index] - wp_midpoint),
            )
            warning = (
                "sample={} WordPiece {} span {} had no spaCy overlap; "
                "fell back to nearest spaCy token {} span {}"
            ).format(
                sample_id,
                wp_index,
                tuple(wp_span),
                spacy_index,
                tuple(spacy_offsets[spacy_index]),
            )
            LOGGER.warning(warning)
            warnings.append(warning)

        alignment_matrix[wp_index, spacy_index] = 1.0
        wp_to_spacy[wp_index] = spacy_index

    return alignment_matrix, wp_to_spacy, warnings


def map_spacy_to_wordpieces(
    adjacency_spacy: np.ndarray,
    alignment_matrix: np.ndarray,
) -> np.ndarray:
    """Compute ``A_wordpiece = M @ A_spacy @ M.T`` exactly.

    This function does *not* normalize the result for a GCN. The returned
    matrix retains dependency and same-spaCy-token connectivity inherited via
    the self-loops already present in ``A_spacy``.
    """
    if alignment_matrix.ndim != 2 or adjacency_spacy.ndim != 2:
        raise ValueError("alignment_matrix and adjacency_spacy must be rank-2")
    if adjacency_spacy.shape[0] != adjacency_spacy.shape[1]:
        raise ValueError("A_spacy must be square")
    if alignment_matrix.shape[1] != adjacency_spacy.shape[0]:
        raise ValueError(
            "M/A_spacy mismatch: {} versus {}".format(
                alignment_matrix.shape, adjacency_spacy.shape
            )
        )

    # alignment_matrix: [sentence_wordpieces, spacy_tokens]
    # adjacency_spacy:  [spacy_tokens, spacy_tokens]
    # adjacency_wordpiece: [sentence_wordpieces, sentence_wordpieces]
    adjacency_wordpiece = alignment_matrix @ adjacency_spacy @ alignment_matrix.T
    return adjacency_wordpiece.astype(np.float32, copy=False)


def build_wordpiece_dependency_graph(
    doc,
    wordpiece_offsets: Sequence[Tuple[int, int]],
    sample_id: str = "",
) -> Dict[str, object]:
    """Build all explicitly named stages of the syntax alignment pipeline."""
    spacy_offsets = [(int(token.idx), int(token.idx + len(token.text))) for token in doc]
    adjacency_spacy, dependency_edges_spacy = build_spacy_dependency_adjacency(doc)
    alignment_matrix, wp_to_spacy, warnings = build_alignment_matrix(
        wordpiece_offsets=wordpiece_offsets,
        spacy_offsets=spacy_offsets,
        sample_id=sample_id,
    )
    adjacency_wordpiece = map_spacy_to_wordpieces(
        adjacency_spacy=adjacency_spacy,
        alignment_matrix=alignment_matrix,
    )

    dependency_edges_wordpiece = [
        (source, target)
        for source in range(adjacency_wordpiece.shape[0])
        for target in range(adjacency_wordpiece.shape[1])
        if source != target and adjacency_wordpiece[source, target] > 0
    ]
    return {
        "A_spacy": adjacency_spacy,
        "alignment_matrix": alignment_matrix,
        "A_wordpiece": adjacency_wordpiece,
        "wordpiece_to_spacy": wp_to_spacy,
        "spacy_offsets": spacy_offsets,
        "dependency_edges_spacy": dependency_edges_spacy,
        "dependency_edges_wordpiece": dependency_edges_wordpiece,
        "warnings": warnings,
    }

