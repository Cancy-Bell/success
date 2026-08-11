#!/usr/bin/env python
"""Official AURC token entry point extended to the end-to-end graph model."""

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, get_linear_schedule_with_warmup

from data_utils import (
    AURCFeatureBuilder,
    AURCFeatureDataset,
    collate_aurc_features,
    load_spacy_pipeline,
)
from label_utils import (
    AU_STANCES,
    BIO_LABELS,
    DOCUMENT_LABELS,
    OFFICIAL_LABELS,
    bio_to_spans,
    collapse_bio_sequence,
)
from metrics_utils import compute_all_metrics
from models import TokenBERT
from prediction_io import write_aurc_prediction_json


LOGGER = logging.getLogger("aurc")
CHECKPOINT_SELECTION_METRIC = "dev_official_token_macro_f1"
LOSS_NAMES = (
    "total_loss",
    "bio_loss",
    "final_bio_loss",
    "initial_bio_loss",
    "official_token_loss",
    "au_loss",
    "document_loss",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: Dict[str, object], device: torch.device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_forward(
    model: TokenBERT,
    batch: Dict[str, object],
) -> Dict[str, object]:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        topic_input_ids=batch["topic_input_ids"],
        topic_attention_mask=batch["topic_attention_mask"],
        topic_token_type_ids=batch["topic_token_type_ids"],
        sentence_indices=batch["sentence_indices"],
        sentence_mask=batch["sentence_mask"],
        dependency_adj_wordpiece=batch["dependency_adj_wordpiece"],
        topics=[str(item["topic"]) for item in batch["metadata"]],
        labels=batch["label_ids"],
        document_labels=batch["document_label_id"],
    )


def training(
    train_dataloader: DataLoader,
    model: TokenBERT,
    device: torch.device,
    optimizer,
    scheduler,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    scaler,
    use_amp: bool,
) -> Dict[str, float]:
    """Optimize three-class token, AU stance, and Document stance losses."""
    model.train()
    running = {name: 0.0 for name in LOSS_NAMES}
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, raw_batch in enumerate(train_dataloader):
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model_forward(model=model, batch=batch)
            scaled_loss = outputs["loss"] / gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        for name in LOSS_NAMES:
            running[name] += float(outputs[name].detach().cpu().item())
        steps += 1

        should_update = (
            (step + 1) % gradient_accumulation_steps == 0
            or step + 1 == len(train_dataloader)
        )
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    denominator = max(1, steps)
    result = {name: value / denominator for name, value in running.items()}
    result["final_o_bias"] = float(model.final_o_bias.detach().cpu().item())
    return result


def _span_with_offsets(span: Dict[str, object], metadata: Dict[str, object]):
    start = int(span["start"])
    end = int(span["end"])
    offsets = metadata["wordpiece_offsets"]
    mapping = metadata["wordpiece_to_original_token"]
    text = str(metadata["text"])
    if 0 <= start < end <= len(offsets):
        char_start = int(offsets[start][0])
        char_end = int(offsets[end - 1][1])
    else:
        char_start = char_end = 0
    original_indices = [
        int(index) for index in mapping[start:end] if int(index) >= 0
    ]
    original_start = min(original_indices) if original_indices else None
    original_end = max(original_indices) + 1 if original_indices else None
    return {
        "start": start,
        "end": end,
        "wordpiece_start": start,
        "wordpiece_end": end,
        "original_token_start": original_start,
        "original_token_end": original_end,
        "char_start": char_start,
        "char_end": char_end,
        "text": text[char_start:char_end],
        "stance": str(span["stance"]),
    }


def build_prediction_record(
    metadata: Dict[str, object],
    gold_bio_ids: Sequence[int],
    sample_output: Dict[str, object],
) -> Dict[str, object]:
    gold_bio_ids = [int(label) for label in gold_bio_ids]
    initial_bio_ids = [int(label) for label in sample_output["initial_bio_ids"]]
    final_bio_ids = [int(label) for label in sample_output["final_bio_ids"]]
    sentence_length = len(gold_bio_ids)
    gold_units = [
        _span_with_offsets(span, metadata) for span in bio_to_spans(gold_bio_ids)
    ]
    initial_units = [
        _span_with_offsets(span, metadata) for span in bio_to_spans(initial_bio_ids)
    ]
    final_units = []
    for span in bio_to_spans(final_bio_ids):
        unit = _span_with_offsets(span, metadata)
        unit["final_stance"] = unit.pop("stance")
        final_units.append(unit)

    au_values = sample_output["au_stance_probs"].cpu().tolist()
    au_stance_predictions = []
    for index, span in enumerate(sample_output["au_spans"]):
        unit = _span_with_offsets(span, metadata)
        probabilities = au_values[index]
        unit.update(
            {
                "initial_bio_stance": unit.pop("stance"),
                "predicted_stance": AU_STANCES[int(np.argmax(probabilities))],
                "stance_probs": dict(zip(AU_STANCES, map(float, probabilities))),
                "attention_weights": sample_output["au_attention_weights"][index]
                .cpu()
                .tolist(),
            }
        )
        au_stance_predictions.append(unit)

    document_values = sample_output["document_probs"].cpu().tolist()
    predicted_document_stance = DOCUMENT_LABELS[int(np.argmax(document_values))]
    initial_official_probs = sample_output["initial_official_probs"].cpu().tolist()
    graph_official_probs = sample_output["graph_official_probs"].cpu().tolist()
    fused_official_probs = sample_output["fused_official_probs"].cpu().tolist()
    stance_fusion_weights = sample_output["stance_fusion_weights"].cpu().tolist()
    graph_official_labels = [
        OFFICIAL_LABELS[int(np.argmax(values))] for values in graph_official_probs
    ]
    fused_official_labels = [
        OFFICIAL_LABELS[int(np.argmax(values))] for values in fused_official_probs
    ]
    return {
        "id": metadata["id"],
        "topic": metadata["topic"],
        "text": metadata["text"],
        "sentence_length": sentence_length,
        "sentence_wordpieces": metadata["sentence_wordpieces"],
        "wordpiece_offsets": metadata["wordpiece_offsets"],
        "gold_bio_ids": gold_bio_ids,
        "initial_bio_ids": initial_bio_ids,
        "final_bio_ids": final_bio_ids,
        # pred_bio_ids remains an explicit compatibility alias for metrics/tools.
        "pred_bio_ids": final_bio_ids,
        "gold_bio": [BIO_LABELS[label] for label in gold_bio_ids],
        "initial_crf_bio": [BIO_LABELS[label] for label in initial_bio_ids],
        "final_bio": [BIO_LABELS[label] for label in final_bio_ids],
        "initial_bio_probs": sample_output["initial_bio_probs"].cpu().tolist(),
        "final_bio_probs": sample_output["final_bio_probs"].cpu().tolist(),
        "feedback_gate": sample_output["feedback_gate"].cpu().tolist(),
        "initial_official_probs": initial_official_probs,
        "graph_official_probs": graph_official_probs,
        "fused_official_probs": fused_official_probs,
        "stance_fusion_weights": stance_fusion_weights,
        "official_gold_labels": collapse_bio_sequence(gold_bio_ids),
        "official_initial_labels": collapse_bio_sequence(initial_bio_ids),
        "official_pred_labels": collapse_bio_sequence(final_bio_ids),
        "graph_official_labels": graph_official_labels,
        "fused_official_labels": fused_official_labels,
        "gold_argument_units": gold_units,
        "initial_argument_units": initial_units,
        "pred_argument_units": final_units,
        "au_stance_predictions": au_stance_predictions,
        "au_stance_probs": [item["stance_probs"] for item in au_stance_predictions],
        "gold_document_stance": metadata["gold_document_stance"],
        "pred_document_stance": predicted_document_stance,
        "document_stance_probs": dict(
            zip(DOCUMENT_LABELS, map(float, document_values))
        ),
        "document_attention_weights": sample_output[
            "document_attention_weights"
        ].cpu().tolist(),
        "graph_source": sample_output["graph_source"],
        "graph_debug": sample_output["graph_debug"],
        "alignment_warnings": metadata.get("alignment_warnings", []),
    }


def evaluation(
    sample_dataloader: DataLoader,
    model: TokenBERT,
    device: torch.device,
    use_amp: bool,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Evaluate complete feedback model using initial predicted BIO graph spans."""
    model.eval()
    loss_totals = {name: 0.0 for name in LOSS_NAMES}
    records: List[Dict[str, object]] = []
    steps = 0
    with torch.no_grad():
        for raw_batch in sample_dataloader:
            metadata = raw_batch["metadata"]
            batch = move_batch_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model_forward(model=model, batch=batch)
            for name in LOSS_NAMES:
                loss_totals[name] += float(outputs[name].detach().cpu().item())
            steps += 1
            for index, (sample_metadata, sample_output) in enumerate(
                zip(metadata, outputs["sample_outputs"])
            ):
                valid_length = int(sample_metadata["sentence_length"])
                gold_ids = raw_batch["label_ids"][index, :valid_length].tolist()
                records.append(
                    build_prediction_record(
                        metadata=sample_metadata,
                        gold_bio_ids=gold_ids,
                        sample_output=sample_output,
                    )
                )

    result = compute_all_metrics(records)
    denominator = max(1, steps)
    result.update(
        {name: total / denominator for name, total in loss_totals.items()}
    )
    result["loss"] = result["total_loss"]
    result["final_o_bias"] = float(model.final_o_bias.detach().cpu().item())
    return result, records


def write_jsonl(path: str, records: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_sentence_lengths_csv(path: str, records: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "topic", "sentence_length", "sentence"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.get("id", ""),
                    "topic": record.get("topic", ""),
                    "sentence_length": record.get("sentence_length", 0),
                    "sentence": record.get("text", ""),
                }
            )


def save_metrics_history(
    history: Sequence[Dict[str, object]], json_path: str, csv_path: str
) -> None:
    with open(json_path, "w", encoding="utf8") as handle:
        json.dump(list(history), handle, ensure_ascii=False, indent=2)
    fieldnames = sorted({key for row in history for key in row.keys()})
    with open(csv_path, "w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def print_metrics(split: str, metrics: Dict[str, float]) -> None:
    LOGGER.info(
        "%s: loss=%.4f | Initial/Final 3-class Token F1=%.4f/%.4f "
        "(delta=%+.4f) | Graph/Fused stance Token F1=%.4f/%.4f "
        "| official segment/sentence F1=%.4f/%.4f | AU token P/R/F1=%.4f/%.4f/%.4f "
        "| All stance Acc/F1=%.4f/%.4f | Arg stance Acc/F1=%.4f/%.4f | "
        "Initial Gold-AU stance Acc/F1=%.4f/%.4f | "
        "Graph Gold-AU stance Acc/F1=%.4f/%.4f (m=%d) | "
        "Entity F1=%.4f | Document F1=%.4f | O-bias=%.4f",
        split,
        metrics["total_loss"],
        metrics["initial_official_token_macro_f1"],
        metrics["final_official_token_macro_f1"],
        metrics["official_token_f1_delta"],
        metrics["graph_official_token_macro_f1"],
        metrics["fused_official_token_macro_f1"],
        metrics["official_segment_f1"],
        metrics["official_sentence_f1"],
        metrics["au_token_precision"],
        metrics["au_token_recall"],
        metrics["au_token_f1"],
        metrics["all_stance_token_accuracy"],
        metrics["all_stance_token_macro_f1"],
        metrics["argument_stance_token_accuracy"],
        metrics["argument_stance_token_macro_f1"],
        metrics["initial_gold_au_stance_accuracy"],
        metrics["initial_gold_au_stance_macro_f1"],
        metrics["gold_au_stance_accuracy"],
        metrics["gold_au_stance_macro_f1"],
        int(metrics["gold_au_stance_count"]),
        metrics["entity_f1"],
        metrics["document_stance_macro_f1"],
        metrics["final_o_bias"],
    )
    LOGGER.info(
        "%s token view: Initial/Final 3-class Token F1=%.4f/%.4f "
        "(delta=%+.4f) | strict AU span F1=%.4f",
        split,
        metrics["initial_bio_token_macro_f1"],
        metrics["final_bio_token_macro_f1"],
        metrics["feedback_token_f1_delta"],
        metrics["au_span_f1"],
    )
    LOGGER.info(
        "%s losses: BIOCombined=%.4f FinalCRF=%.4f InitialCRF=%.4f "
        "OfficialTokenCE=%.4f AU=%.4f Document=%.4f",
        split,
        metrics["bio_loss"],
        metrics["final_bio_loss"],
        metrics["initial_bio_loss"],
        metrics["official_token_loss"],
        metrics["au_loss"],
        metrics["document_loss"],
    )


def save_checkpoint(
    path: str,
    model: TokenBERT,
    config: BertConfig,
    args: argparse.Namespace,
    epoch: int,
    dev_official_token_f1: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": int(epoch),
            "selection_metric": CHECKPOINT_SELECTION_METRIC,
            "selection_score": float(dev_official_token_f1),
            "dev_official_token_f1": float(dev_official_token_f1),
            "training_args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str, model: TokenBERT, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def dev_checkpoint_score(split_results: Dict[str, Dict[str, float]]) -> float:
    """Return the sole early-stop/checkpoint score: Dev Official Token macro-F1."""
    return float(split_results["dev"]["official_token_macro_f1"])


def checkpoint_selection_metadata(checkpoint: Dict[str, object]) -> Tuple[str, float]:
    """Read new checkpoints while retaining explicit legacy-checkpoint support."""
    metric = str(
        checkpoint.get("selection_metric", "dev_final_bio_token_macro_f1")
    )
    if "selection_score" in checkpoint:
        score = float(checkpoint["selection_score"])
    elif "dev_official_token_f1" in checkpoint:
        score = float(checkpoint["dev_official_token_f1"])
    else:
        # Checkpoints produced before Official-Token selection used this key for
        # Legacy checkpoints used the former Dev five-class BIO macro-F1 key.
        score = float(checkpoint.get("dev_token_f1", 0.0))
    return metric, score


def build_model(config: BertConfig, args: argparse.Namespace) -> TokenBERT:
    return TokenBERT(
        model_name=args.pretrained_weights,
        num_labels=3,
        output_hidden_states=False,
        use_crf=args.crf,
        config=config,
        initialize_from_pretrained=True,
        gcn_layers=args.gcn_layers,
        gcn_dropout=args.gcn_dropout,
        hetgat_layers=args.hetgat_layers,
        hetgat_heads=args.hetgat_heads,
        hetgat_dropout=args.hetgat_dropout,
        au_semantic_threshold=args.au_semantic_threshold,
        au_top_k=args.au_top_k,
        au_syntax_hops=args.au_syntax_hops,
        num_document_labels=len(DOCUMENT_LABELS),
        initial_o_bias=args.initial_o_bias,
        local_files_only=args.local_files_only,
    )


def build_optimizer_and_scheduler(
    model: TokenBERT,
    args: argparse.Namespace,
    train_loader: DataLoader,
    total_epochs: int,
):
    """Build one optimizer/scheduler containing the complete registered model."""
    if total_epochs < 1:
        raise ValueError("training must contain at least one epoch")
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    optimizer_grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in named_parameters
                if not any(item in name for item in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in named_parameters
                if any(item in name for item in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_training_steps = optimizer_steps_per_epoch * total_epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    LOGGER.info(
        "Optimizer initialized: epochs=%d trainable_parameters=%d "
        "steps=%d warmup_steps=%d",
        total_epochs,
        sum(parameter.numel() for _, parameter in named_parameters),
        total_training_steps,
        warmup_steps,
    )
    return optimizer, scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fine-Grained Argument Unit Recognition and Classification."
    )
    # Official arguments retained.
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--card_number", type=int, default=0, help="GPU card number.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs.")
    parser.add_argument("--num_labels", type=int, default=3, choices=[3])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=64)
    parser.add_argument("--topic_max_length", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument(
        "--target_domain", choices=["In-Domain", "Cross-Domain"], default="In-Domain"
    )
    parser.add_argument("--input_file", default="AURC_DATA_dict.json")
    parser.add_argument("--data_dir", default="./data/")
    parser.add_argument("--output_dir", default="./models/")
    parser.add_argument(
        "--pretrained_weights", default="bert-large-cased-whole-word-masking"
    )
    parser.add_argument("--fine_tuning", action="store_true", default=True)
    parser.add_argument("--freeze_bert", action="store_true", default=False)
    parser.add_argument("--crf", action="store_true", default=True)
    parser.add_argument("--no_crf", action="store_false", dest="crf")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--no_eval", action="store_false", dest="eval")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--no_save_model", action="store_false", dest="save_model")
    parser.add_argument(
        "--save_prediction",
        "--save_predictions",
        action="store_true",
        dest="save_predictions",
        default=True,
    )
    parser.add_argument(
        "--no_save_predictions", action="store_false", dest="save_predictions"
    )
    parser.add_argument(
        "--save_detailed_predictions",
        action="store_true",
        help=(
            "Also save rich JSONL diagnostic predictions with probabilities, "
            "attention weights and intermediate fields. By default only compact "
            "official-style AURC JSON files are saved."
        ),
    )

    # End-to-end graph parameters, centralized here.
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--spacy_batch_size", type=int, default=64)
    parser.add_argument("--preprocess_log_every", type=int, default=250)
    parser.add_argument("--allow_spacy_fallback", action="store_true", default=False)
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--gcn_dropout", type=float, default=0.1)
    parser.add_argument("--hetgat_layers", type=int, default=2)
    parser.add_argument("--hetgat_heads", type=int, default=1)
    parser.add_argument("--hetgat_dropout", type=float, default=0.1)
    parser.add_argument("--au_semantic_threshold", type=float, default=0.5)
    parser.add_argument("--au_top_k", type=int, default=3)
    parser.add_argument("--au_syntax_hops", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--initial_o_bias",
        type=float,
        default=0.0,
        help="Initial learnable bias added to the Final-CRF non emission.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument(
        "--bert_dropout",
        type=float,
        default=0.1,
        help="Dropout used by BERT hidden states and self-attention probabilities.",
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_alignment_debug", action="store_true", default=False)
    parser.add_argument("--alignment_debug_limit", type=int, default=10)
    parser.add_argument("--local_files_only", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if args.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be >= 1")
    if args.train and not args.save_model:
        raise ValueError(
            "training requires checkpoint saving so final Dev/Test can be run "
            "from the best Dev Official Token macro-F1 model"
        )

    set_seed(args.seed)
    device = torch.device(
        "cuda:{}".format(args.card_number) if torch.cuda.is_available() else "cpu"
    )
    LOGGER.info("device=%s", device)

    task = "_".join(["aurc", args.target_domain[:2].lower()])
    output_dir = os.path.abspath(args.output_dir)
    predictions_dir = os.path.join(output_dir, "predictions")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_token_model.pt")
    config_path = os.path.join(checkpoint_dir, "{}_config.json".format(task))
    metrics_json_path = os.path.join(output_dir, "metrics_history.json")
    metrics_csv_path = os.path.join(output_dir, "metrics_history.csv")
    LOGGER.info("Best checkpoint criterion = Dev Official Token macro-F1")
    LOGGER.info("best_model_path=%s", best_model_path)

    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, "r", encoding="utf8") as handle:
        aurc_data = json.load(handle)
    LOGGER.info("Loaded %d official AURC topics from %s", len(aurc_data), input_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_weights,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    nlp = load_spacy_pipeline(
        model_name=args.spacy_model,
        allow_fallback=args.allow_spacy_fallback,
    )
    feature_builder = AURCFeatureBuilder(
        tokenizer=tokenizer,
        nlp=nlp,
        max_sequence_length=args.max_sequence_length,
        topic_max_length=args.topic_max_length,
        save_alignment_debug=args.save_alignment_debug,
        alignment_debug_limit=args.alignment_debug_limit,
        progress_log_every=args.preprocess_log_every,
    )
    split_features = feature_builder.build_splits(
        aurc_data=aurc_data,
        target_domain=args.target_domain,
        spacy_batch_size=args.spacy_batch_size,
    )
    feature_builder.save_debug_records(
        os.path.join(output_dir, "alignment_debug.jsonl")
    )
    for split, features in split_features.items():
        LOGGER.info("%s features=%d", split, len(features))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        AURCFeatureDataset(split_features["train"]),
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_aurc_features,
    )
    train_eval_loader = DataLoader(
        AURCFeatureDataset(split_features["train"]),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_aurc_features,
    )
    dev_loader = DataLoader(
        AURCFeatureDataset(split_features["dev"]),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_aurc_features,
    )
    test_loader = DataLoader(
        AURCFeatureDataset(split_features["test"]),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_aurc_features,
    )

    config = BertConfig.from_pretrained(
        args.pretrained_weights,
        num_labels=3,
        local_files_only=args.local_files_only,
    )
    if not 0.0 <= args.bert_dropout < 1.0:
        raise ValueError("--bert_dropout must be in the interval [0, 1).")
    config.hidden_dropout_prob = args.bert_dropout
    config.attention_probs_dropout_prob = args.bert_dropout
    model = build_model(config=config, args=args)
    if args.train and args.gradient_checkpointing:
        model.tokenbert.bert.gradient_checkpointing_enable()
        model.topic_bert.gradient_checkpointing_enable()
        LOGGER.info("Gradient checkpointing enabled for both BERT encoders")
    if args.freeze_bert:
        for parameter in model.tokenbert.bert.parameters():
            parameter.requires_grad = False
        for parameter in model.topic_bert.parameters():
            parameter.requires_grad = False
        LOGGER.warning("Both BERT encoders were frozen by --freeze_bert")
    model.to(device)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    LOGGER.info("CUDA automatic mixed precision=%s", use_amp)
    LOGGER.info(
        "Full end-to-end training from epoch 1: Dual-BERT + GCN + "
        "Semantic-Syntax Fusion + Initial 3-class CRF + batch HetGAT + "
        "stance feedback + Final 3-class CRF"
    )
    LOGGER.info(
        "Loss = FinalCRF + InitialCRF + OfficialTokenNLL + AU stance + "
        "Document stance (all coefficients = 1.0)"
    )
    LOGGER.info(
        "Final O-emission bias: learnable scalar initialized to %.3f",
        args.initial_o_bias,
    )
    config.to_json_file(config_path)
    with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    if args.train:
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            args=args,
            train_loader=train_loader,
            total_epochs=args.epochs,
        )

        metrics_history: List[Dict[str, object]] = []
        best_dev_official_token_f1 = float("-inf")
        best_epoch = None
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            LOGGER.info("Epoch %d | %s", epoch, dt.datetime.now())
            optimization_metrics = training(
                train_dataloader=train_loader,
                model=model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=args.max_grad_norm,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                scaler=scaler,
                use_amp=use_amp,
            )
            LOGGER.info(
                "Optimization losses: total=%.4f BIOCombined=%.4f "
                "FinalCRF=%.4f InitialCRF=%.4f OfficialTokenCE=%.4f "
                "AU=%.4f Document=%.4f",
                optimization_metrics["total_loss"],
                optimization_metrics["bio_loss"],
                optimization_metrics["final_bio_loss"],
                optimization_metrics["initial_bio_loss"],
                optimization_metrics["official_token_loss"],
                optimization_metrics["au_loss"],
                optimization_metrics["document_loss"],
            )

            split_results = {}
            for split, loader in (
                ("train", train_eval_loader),
                ("dev", dev_loader),
                ("test", test_loader),
            ):
                split_metrics, records = evaluation(
                    sample_dataloader=loader,
                    model=model,
                    device=device,
                    use_amp=use_amp,
                )
                split_results[split] = split_metrics
                print_metrics(split.upper(), split_metrics)
                history_row = {"epoch": epoch, "split": split}
                history_row.update(split_metrics)
                if split == "train":
                    history_row.update(
                        {
                            "optimization_{}".format(key): value
                            for key, value in optimization_metrics.items()
                        }
                    )
                metrics_history.append(history_row)
                if args.save_predictions and args.save_detailed_predictions:
                    write_jsonl(
                        os.path.join(
                            predictions_dir,
                            "{}_epoch_{:02d}.jsonl".format(split, epoch),
                        ),
                        records,
                    )
            save_metrics_history(
                metrics_history, metrics_json_path, metrics_csv_path
            )

            current_score = dev_checkpoint_score(split_results)
            improved = current_score > (
                best_dev_official_token_f1 + args.early_stop_min_delta
            )
            if improved:
                best_dev_official_token_f1 = current_score
                best_epoch = epoch
                patience_counter = 0
                LOGGER.info("Dev Official Token F1 improved; saving checkpoint.")
                if args.save_model:
                    save_checkpoint(
                        best_model_path,
                        model=model,
                        config=config,
                        args=args,
                        epoch=epoch,
                        dev_official_token_f1=current_score,
                    )
            else:
                patience_counter += 1
                LOGGER.info("Dev Official Token F1 did not improve.")
            LOGGER.info(
                "Early Stopping Monitor: metric=Dev Official Token F1 current=%.4f "
                "best=%.4f best_epoch=%s patience=%d/%d",
                current_score,
                best_dev_official_token_f1,
                best_epoch,
                patience_counter,
                args.early_stop_patience,
            )
            if patience_counter >= args.early_stop_patience:
                LOGGER.info("Early stopping triggered by Dev Official Token F1.")
                LOGGER.info(
                    "Best epoch: %s | Best Dev Official Token F1: %.4f",
                    best_epoch,
                    best_dev_official_token_f1,
                )
                break

        if not os.path.isfile(best_model_path):
            raise RuntimeError(
                "No Dev Token-F1 checkpoint was saved. Keep --save_model enabled."
            )
        checkpoint = load_checkpoint(best_model_path, model=model, device=device)
        selection_metric, selection_score = checkpoint_selection_metadata(checkpoint)
        LOGGER.info(
            "Reloaded best checkpoint: metric=%s epoch=%s score=%.4f",
            selection_metric,
            checkpoint.get("epoch"),
            selection_score,
        )
        final_best_metrics = {}
        for split, loader in (("dev", dev_loader), ("test", test_loader)):
            final_metrics, final_records = evaluation(
                sample_dataloader=loader,
                model=model,
                device=device,
                use_amp=use_amp,
            )
            final_best_metrics[split] = final_metrics
            print_metrics("{}_BEST".format(split.upper()), final_metrics)
            write_sentence_lengths_csv(
                os.path.join(
                    predictions_dir, "{}_sentence_lengths.csv".format(split)
                ),
                final_records,
            )
            write_aurc_prediction_json(
                os.path.join(predictions_dir, "{}_best.json".format(split)),
                aurc_data,
                final_records,
            )
            write_aurc_prediction_json(
                os.path.join(predictions_dir, "{}_best_aurc.json".format(split)),
                aurc_data,
                final_records,
            )
            if args.save_detailed_predictions:
                write_jsonl(
                    os.path.join(
                        predictions_dir, "{}_best_detailed.jsonl".format(split)
                    ),
                    final_records,
                )
        with open(
            os.path.join(output_dir, "final_best_metrics.json"),
            "w",
            encoding="utf8",
        ) as handle:
            json.dump(
                {
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                    "best_epoch": checkpoint.get("epoch"),
                    "best_dev_official_token_f1": (
                        checkpoint.get("dev_official_token_f1")
                        if selection_metric == CHECKPOINT_SELECTION_METRIC
                        else None
                    ),
                    "splits": final_best_metrics,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    elif args.eval:
        checkpoint = load_checkpoint(best_model_path, model=model, device=device)
        selection_metric, selection_score = checkpoint_selection_metadata(checkpoint)
        final_best_metrics = {}
        for split, loader in (("train", train_eval_loader), ("dev", dev_loader), ("test", test_loader)):
            final_metrics, final_records = evaluation(
                sample_dataloader=loader,
                model=model,
                device=device,
                use_amp=use_amp,
            )
            final_best_metrics[split] = final_metrics
            print_metrics("{}_BEST".format(split.upper()), final_metrics)
            if split in ("dev", "test"):
                write_sentence_lengths_csv(
                    os.path.join(
                        predictions_dir, "{}_sentence_lengths.csv".format(split)
                    ),
                    final_records,
                )
                write_aurc_prediction_json(
                    os.path.join(
                        predictions_dir, "{}_best.json".format(split)
                    ),
                    aurc_data,
                    final_records,
                )
                write_aurc_prediction_json(
                    os.path.join(
                        predictions_dir, "{}_best_aurc.json".format(split)
                    ),
                    aurc_data,
                    final_records,
                )
                if args.save_detailed_predictions:
                    write_jsonl(
                        os.path.join(
                            predictions_dir, "{}_best_detailed.jsonl".format(split)
                        ),
                        final_records,
                    )
        with open(
            os.path.join(output_dir, "final_best_metrics.json"),
            "w",
            encoding="utf8",
        ) as handle:
            json.dump(
                {
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                    "best_epoch": checkpoint.get("epoch"),
                    "best_dev_official_token_f1": (
                        checkpoint.get("dev_official_token_f1")
                        if selection_metric == CHECKPOINT_SELECTION_METRIC
                        else None
                    ),
                    "splits": final_best_metrics,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
