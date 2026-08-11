#!/usr/bin/env python
"""Feature conversion for the official AURC JSON data and joint topic input."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from label_utils import (
    DOCUMENT_LABEL_TO_ID,
    O,
    official_labels_to_bio,
)
from syntax_utils import build_wordpiece_dependency_graph


LOGGER = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    """Extended version of the official ``InputFeatures`` container."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    topic_input_ids: torch.Tensor
    topic_attention_mask: torch.Tensor
    topic_token_type_ids: torch.Tensor
    sentence_indices: torch.Tensor
    sentence_mask: torch.Tensor
    label_ids: torch.Tensor
    dependency_adj_wordpiece: torch.Tensor
    document_label_id: torch.Tensor
    metadata: Dict[str, object]


class AURCFeatureDataset(Dataset):
    def __init__(self, features: Sequence[InputFeatures]):
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> InputFeatures:
        return self.features[index]


def collate_aurc_features(features: Sequence[InputFeatures]) -> Dict[str, object]:
    """Stack tensor fields while retaining JSON-ready per-sample metadata."""
    tensor_fields = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "topic_input_ids",
        "topic_attention_mask",
        "topic_token_type_ids",
        "sentence_indices",
        "sentence_mask",
        "label_ids",
        "dependency_adj_wordpiece",
        "document_label_id",
    )
    batch = {
        field: torch.stack([getattr(feature, field) for feature in features], dim=0)
        for field in tensor_fields
    }
    batch["metadata"] = [feature.metadata for feature in features]
    return batch


def load_spacy_pipeline(model_name: str, allow_fallback: bool = False):
    """Load a dependency parser, with an explicit opt-in fallback for smoke tests."""
    try:
        import spacy

        pipeline = spacy.load(model_name)
    except Exception as error:
        if not allow_fallback:
            raise RuntimeError(
                "Could not load spaCy model {!r}. Install it with "
                "`python -m spacy download {}` or pass --allow_spacy_fallback "
                "for a self-loop-only diagnostic run."
                .format(model_name, model_name)
            ) from error
        import spacy

        LOGGER.warning(
            "spaCy model %s is unavailable; using a blank English tokenizer. "
            "Dependency graphs will contain only inherited self connections.",
            model_name,
        )
        pipeline = spacy.blank("en")

    if "parser" not in pipeline.pipe_names and not allow_fallback:
        raise RuntimeError(
            "spaCy pipeline {!r} has no dependency parser; a parser is required."
            .format(model_name)
        )
    return pipeline


def _token_variants(token: str) -> List[str]:
    replacements = {
        "``": '"',
        "''": '"',
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    variants = [token]
    if token in replacements:
        variants.append(replacements[token])
    # The released AURC sentences contain both Unicode punctuation and a few
    # legacy mojibake/code-page characters, while the prepared token strings
    # often contain their ASCII spaCy equivalents.
    if token == "-":
        variants.extend(["–", "—", "−", "\x96", "\x97"])
    elif token == '"':
        variants.extend(
            ["“", "”", "„", "â€œ", "â€\x9d", "â\x80\x9c", "â\x80\x9d"]
        )
    elif token == "'":
        variants.extend(
            ["’", "‘", "\x91", "\x92", "â€˜", "â€™", "â\x80\x98", "â\x80\x99"]
        )
    elif token.startswith("'"):
        suffix = token[1:]
        variants.extend(
            [
                "’" + suffix,
                "‘" + suffix,
                "\x91" + suffix,
                "\x92" + suffix,
                "â€˜" + suffix,
                "â€™" + suffix,
                "â\x80\x98" + suffix,
                "â\x80\x99" + suffix,
            ]
        )
    return variants


def locate_pretokenized_offsets(
    text: str,
    tokens: Sequence[str],
    sample_id: str,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Locate official prepared spaCy tokens in the untouched raw sentence."""
    offsets: List[Tuple[int, int]] = []
    warnings: List[str] = []
    cursor = 0
    lower_text = text.lower()
    for token_index, token in enumerate(tokens):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        best_start = -1
        best_variant = token
        for variant in _token_variants(token):
            if text.startswith(variant, cursor):
                best_start = cursor
                best_variant = variant
                break
            if lower_text.startswith(variant.lower(), cursor):
                best_start = cursor
                best_variant = text[cursor : cursor + len(variant)]
                break
        if best_start < 0:
            candidates = []
            for variant in _token_variants(token):
                exact = text.find(variant, cursor)
                folded = lower_text.find(variant.lower(), cursor)
                for position in (exact, folded):
                    if position >= 0:
                        candidates.append((position, variant))
            if candidates:
                best_start, best_variant = min(candidates, key=lambda item: item[0])
            else:
                # Do not invent an offset or advance the cursor. The official
                # prepared token stream occasionally omits/normalizes raw-text
                # material (notably URLs and legacy punctuation). Advancing an
                # approximate cursor here would make every later token drift.
                warning = (
                    "sample={} could not locate official token {}={!r}; "
                    "leaving it unaligned without shifting later tokens"
                ).format(sample_id, token_index, token)
                LOGGER.warning(warning)
                warnings.append(warning)
                offsets.append((-1, -1))
                continue
        best_end = min(len(text), best_start + len(best_variant))
        offsets.append((best_start, best_end))
        cursor = best_end
    return offsets, warnings


def align_wordpieces_to_original_tokens(
    wordpiece_offsets: Sequence[Tuple[int, int]],
    original_offsets: Sequence[Tuple[int, int]],
    sample_id: str,
) -> Tuple[List[int], List[str]]:
    """Map each WordPiece to the maximum-overlap official original token."""
    mapping: List[int] = []
    warnings: List[str] = []
    unaligned_wordpieces: List[int] = []
    for wp_index, (wp_start, wp_end) in enumerate(wordpiece_offsets):
        overlaps = [
            max(0, min(wp_end, token_end) - max(wp_start, token_start))
            for token_start, token_end in original_offsets
        ]
        if overlaps and max(overlaps) > 0:
            mapping.append(overlaps.index(max(overlaps)))
            continue
        # Raw substrings omitted by the official prepared annotation stream
        # (for example a URL) have no defensible gold stance. Mark them O via
        # -1 instead of copying a potentially argumentative nearest token.
        mapping.append(-1)
        unaligned_wordpieces.append(wp_index)
    if unaligned_wordpieces:
        preview = unaligned_wordpieces[:8]
        warning = (
            "sample={} has {} WordPieces outside the official annotation "
            "token stream (indices {}{}); assigning token label=non"
        ).format(
            sample_id,
            len(unaligned_wordpieces),
            preview,
            "..." if len(unaligned_wordpieces) > len(preview) else "",
        )
        LOGGER.warning(warning)
        warnings.append(warning)
    return mapping, warnings


class AURCFeatureBuilder:
    """Convert official AURC rows into joint Topic/Sentence BERT features."""

    def __init__(
        self,
        tokenizer,
        nlp,
        max_sequence_length: int,
        topic_max_length: int = 16,
        save_alignment_debug: bool = False,
        alignment_debug_limit: int = 10,
        progress_log_every: int = 250,
    ):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "A fast HuggingFace tokenizer is required for offset_mapping "
                "and reliable Topic/Sentence separation."
            )
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.max_sequence_length = int(max_sequence_length)
        self.topic_max_length = int(topic_max_length)
        if self.topic_max_length < 3:
            raise ValueError("topic_max_length must allow [CLS], Topic and [SEP]")
        self.save_alignment_debug = bool(save_alignment_debug)
        self.alignment_debug_limit = int(alignment_debug_limit)
        self.progress_log_every = max(1, int(progress_log_every))
        self.debug_records: List[Dict[str, object]] = []

    def _build_feature(self, topic: str, row: Dict[str, object], doc) -> InputFeatures:
        sentence = str(row["sentence"])
        sample_id = str(row["sentence_hash"])
        encoding = self.tokenizer(
            topic,
            sentence,
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        sequence_ids = encoding.sequence_ids()
        input_ids = list(encoding["input_ids"])
        attention_mask = list(encoding["attention_mask"])
        token_type_ids = list(encoding.get("token_type_ids", [0] * len(input_ids)))
        all_offsets = [tuple(map(int, span)) for span in encoding["offset_mapping"]]
        topic_encoding = self.tokenizer(
            topic,
            add_special_tokens=True,
            max_length=self.topic_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        topic_input_ids = list(topic_encoding["input_ids"])
        topic_attention_mask = list(topic_encoding["attention_mask"])
        topic_token_type_ids = list(
            topic_encoding.get("token_type_ids", [0] * len(topic_input_ids))
        )
        sentence_indices = [
            index
            for index, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
            and attention_mask[index] == 1
            and all_offsets[index][1] > all_offsets[index][0]
        ]
        sentence_offsets = [all_offsets[index] for index in sentence_indices]
        sentence_tokens = self.tokenizer.convert_ids_to_tokens(
            [input_ids[index] for index in sentence_indices]
        )

        official_tokens = str(row.get("tokenized_sentence_spacy", "")).split()
        official_labels = str(row.get("tokenized_sentence_spacy_labels", "")).split()
        row_warnings: List[str] = []
        if len(official_tokens) != len(official_labels):
            warning = (
                "sample={} has {} official tokens but {} labels; padding/truncating "
                "labels with non"
            ).format(sample_id, len(official_tokens), len(official_labels))
            LOGGER.warning(warning)
            row_warnings.append(warning)
            official_labels = (official_labels + ["non"] * len(official_tokens))[
                : len(official_tokens)
            ]

        original_offsets, locate_warnings = locate_pretokenized_offsets(
            text=sentence,
            tokens=official_tokens,
            sample_id=sample_id,
        )
        wordpiece_to_original, label_alignment_warnings = (
            align_wordpieces_to_original_tokens(
                wordpiece_offsets=sentence_offsets,
                original_offsets=original_offsets,
                sample_id=sample_id,
            )
        )
        gold_bio = official_labels_to_bio(
            official_labels=official_labels,
            wordpiece_to_original_token=wordpiece_to_original,
        )

        syntax = build_wordpiece_dependency_graph(
            doc=doc,
            wordpiece_offsets=sentence_offsets,
            sample_id=sample_id,
        )
        sentence_length = len(sentence_indices)
        if syntax["A_wordpiece"].shape != (sentence_length, sentence_length):
            raise AssertionError(
                "sample={} A_wordpiece shape {} != ({}, {})".format(
                    sample_id,
                    syntax["A_wordpiece"].shape,
                    sentence_length,
                    sentence_length,
                )
            )

        padded_sentence_indices = sentence_indices + [0] * (
            self.max_sequence_length - sentence_length
        )
        sentence_mask = [1] * sentence_length + [0] * (
            self.max_sequence_length - sentence_length
        )
        padded_gold_bio = gold_bio + [O] * (self.max_sequence_length - sentence_length)
        adjacency_padded = torch.zeros(
            (self.max_sequence_length, self.max_sequence_length), dtype=torch.float
        )
        if sentence_length:
            adjacency_padded[:sentence_length, :sentence_length] = torch.from_numpy(
                syntax["A_wordpiece"]
            )

        document_label = str(row.get("sentence_level_stance", "non")).lower()
        if document_label not in DOCUMENT_LABEL_TO_ID:
            warning = "sample={} has unknown document stance {!r}; using non".format(
                sample_id, document_label
            )
            LOGGER.warning(warning)
            row_warnings.append(warning)
            document_label = "non"

        row_warnings.extend(locate_warnings)
        row_warnings.extend(label_alignment_warnings)
        row_warnings.extend(syntax["warnings"])

        metadata: Dict[str, object] = {
            "id": sample_id,
            "topic": topic,
            "text": sentence,
            "sentence_length": sentence_length,
            "sentence_wordpieces": sentence_tokens,
            "wordpiece_offsets": [list(span) for span in sentence_offsets],
            "wordpiece_to_original_token": wordpiece_to_original,
            "wordpiece_to_spacy": syntax["wordpiece_to_spacy"],
            "original_tokens": official_tokens,
            "original_token_offsets": [list(span) for span in original_offsets],
            "gold_document_stance": document_label,
            "alignment_warnings": row_warnings,
        }

        should_debug = self.save_alignment_debug and (
            len(self.debug_records) < self.alignment_debug_limit or bool(row_warnings)
        )
        if should_debug:
            debug_record = {
                "id": sample_id,
                "topic": topic,
                "text": sentence,
                "spacy_tokens": [token.text for token in doc],
                "spacy_offsets": [list(span) for span in syntax["spacy_offsets"]],
                "wordpieces": sentence_tokens,
                "wordpiece_offsets": [list(span) for span in sentence_offsets],
                "wordpiece_to_spacy": syntax["wordpiece_to_spacy"],
                "dependency_edges_spacy": [
                    list(edge) for edge in syntax["dependency_edges_spacy"]
                ],
                "dependency_edges_wordpiece": [
                    list(edge) for edge in syntax["dependency_edges_wordpiece"]
                ],
                "alignment_matrix": syntax["alignment_matrix"].tolist(),
                "A_spacy": syntax["A_spacy"].tolist(),
                "A_wordpiece": syntax["A_wordpiece"].tolist(),
                "warnings": row_warnings,
            }
            self.debug_records.append(debug_record)

        return InputFeatures(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            topic_input_ids=torch.tensor(topic_input_ids, dtype=torch.long),
            topic_attention_mask=torch.tensor(topic_attention_mask, dtype=torch.long),
            topic_token_type_ids=torch.tensor(topic_token_type_ids, dtype=torch.long),
            sentence_indices=torch.tensor(padded_sentence_indices, dtype=torch.long),
            sentence_mask=torch.tensor(sentence_mask, dtype=torch.bool),
            label_ids=torch.tensor(padded_gold_bio, dtype=torch.long),
            dependency_adj_wordpiece=adjacency_padded,
            document_label_id=torch.tensor(
                DOCUMENT_LABEL_TO_ID[document_label], dtype=torch.long
            ),
            metadata=metadata,
        )

    def build_splits(
        self,
        aurc_data: Dict[str, Sequence[Dict[str, object]]],
        target_domain: str,
        spacy_batch_size: int = 64,
    ) -> Dict[str, List[InputFeatures]]:
        rows: List[Tuple[str, Dict[str, object], str]] = []
        for topic, topic_rows in aurc_data.items():
            for row in topic_rows:
                split = str(row.get(target_domain, ""))
                if split in ("Train", "Dev", "Test"):
                    rows.append((topic, row, split))

        split_features: Dict[str, List[InputFeatures]] = {
            "train": [],
            "dev": [],
            "test": [],
        }
        disabled = [name for name in ("ner",) if name in self.nlp.pipe_names]
        documents = self.nlp.pipe(
            (str(row["sentence"]) for _, row, _ in rows),
            batch_size=int(spacy_batch_size),
            disable=disabled,
        )
        for row_index, ((topic, row, split), doc) in enumerate(
            zip(rows, documents), start=1
        ):
            feature = self._build_feature(topic=topic, row=row, doc=doc)
            split_features[split.lower()].append(feature)
            if row_index % self.progress_log_every == 0 or row_index == len(rows):
                LOGGER.info(
                    "AURC preprocessing: %d/%d samples (train=%d dev=%d test=%d)",
                    row_index,
                    len(rows),
                    len(split_features["train"]),
                    len(split_features["dev"]),
                    len(split_features["test"]),
                )
        return split_features

    def save_debug_records(self, output_path: str) -> None:
        if not self.save_alignment_debug:
            return
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf8") as handle:
            for record in self.debug_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
