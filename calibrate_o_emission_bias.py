#!/usr/bin/env python
"""Calibrate an additive O-label emission bias without retraining AURC.

The saved ``final_bio_probs`` are sufficient for exact CRF re-decoding because
``log(softmax(emissions))`` differs from the original emissions only by one
label-independent constant per token.  Such constants do not change the
Viterbi path.  Only the small CRF parameter tensors are copied from the model
checkpoint; the 2.7 GB Dual-BERT weights are memory-mapped and never evaluated.
"""

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from label_utils import BIO_LABELS, bio_to_spans, collapse_bio_sequence  # noqa: E402
from metrics_utils import compute_all_metrics  # noqa: E402
from models import LinearChainCRF  # noqa: E402


LOGGER = logging.getLogger("aurc_o_bias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune an additive Final-BIO O-emission bias on Dev Official Token "
            "macro-F1, then apply the selected value to Test. No training occurs."
        )
    )
    parser.add_argument(
        "--experiment_dir",
        default=None,
        help=(
            "Completed experiment directory containing checkpoints/ and "
            "predictions/. If omitted, the newest completed batch-size "
            "experiment is discovered automatically."
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dev_predictions", default=None)
    parser.add_argument("--test_predictions", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bias_min", type=float, default=0.0)
    parser.add_argument("--bias_max", type=float, default=1.0)
    parser.add_argument(
        "--bias_step",
        type=float,
        default=0.025,
        help="Fine default grid; direct execution includes the observed optimum.",
    )
    parser.add_argument("--decode_batch_size", type=int, default=256)
    parser.add_argument(
        "--no_save_predictions",
        action="store_false",
        dest="save_predictions",
        default=True,
    )
    return parser.parse_args()


def discover_experiment_dir(project_root: Path) -> Path:
    comparison_root = project_root / "models" / "batch_size_comparison"
    candidates = []
    if comparison_root.is_dir():
        for checkpoint in comparison_root.glob(
            "run_*/batch_size_*/checkpoints/best_token_model.pt"
        ):
            experiment_dir = checkpoint.parent.parent
            dev_path = experiment_dir / "predictions" / "dev_best.jsonl"
            test_path = experiment_dir / "predictions" / "test_best.jsonl"
            result_path = experiment_dir / "experiment_result.json"
            if not (dev_path.is_file() and test_path.is_file()):
                continue
            completed = True
            if result_path.is_file():
                with result_path.open("r", encoding="utf8") as handle:
                    completed = json.load(handle).get("status") == "completed"
            if completed:
                candidates.append(experiment_dir)
    if not candidates:
        raise FileNotFoundError(
            "Could not discover a completed experiment with checkpoint, "
            "dev_best.jsonl, and test_best.jsonl. Pass --experiment_dir."
        )
    return max(
        candidates,
        key=lambda path: (path / "checkpoints" / "best_token_model.pt").stat().st_mtime,
    )


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    experiment_dir = (
        Path(args.experiment_dir).expanduser().resolve()
        if args.experiment_dir
        else discover_experiment_dir(PROJECT_ROOT)
    )
    paths = {
        "experiment_dir": experiment_dir,
        "checkpoint": Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else experiment_dir / "checkpoints" / "best_token_model.pt",
        "dev": Path(args.dev_predictions).expanduser().resolve()
        if args.dev_predictions
        else experiment_dir / "predictions" / "dev_best.jsonl",
        "test": Path(args.test_predictions).expanduser().resolve()
        if args.test_predictions
        else experiment_dir / "predictions" / "test_best.jsonl",
        "output": Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else experiment_dir / "o_emission_bias_calibration",
    }
    for name in ("checkpoint", "dev", "test"):
        if not paths[name].is_file():
            raise FileNotFoundError("{} does not exist: {}".format(name, paths[name]))
    return paths


def build_bias_grid(minimum: float, maximum: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--bias_step must be > 0")
    if maximum < minimum:
        raise ValueError("--bias_max must be >= --bias_min")
    start = Decimal(str(minimum))
    stop = Decimal(str(maximum))
    increment = Decimal(str(step))
    values = []
    current = start
    while current <= stop:
        values.append(float(current))
        current += increment
    if not values:
        raise ValueError("O-emission bias grid is empty")
    return values


def load_crf(checkpoint_path: Path) -> LinearChainCRF:
    LOGGER.info("Memory-mapping checkpoint to read CRF parameters: %s", checkpoint_path)
    load_kwargs = {"map_location": "cpu"}
    # mmap prevents allocating all Dual-BERT tensors while loading the 2.7 GB file.
    try:
        checkpoint = torch.load(
            str(checkpoint_path), mmap=True, weights_only=False, **load_kwargs
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), mmap=True, **load_kwargs)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    prefix = "crf."
    crf_state = {
        key[len(prefix) :]: value.detach().cpu()
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not crf_state:
        raise KeyError("Checkpoint contains no crf.* parameters")
    crf = LinearChainCRF(len(BIO_LABELS)).cpu().eval()
    crf.load_state_dict(crf_state, strict=True)
    del crf_state, state_dict, checkpoint
    gc.collect()
    return crf


def _light_record(record: Dict[str, object]) -> Dict[str, object]:
    """Keep only fields consumed by compute_all_metrics plus cached emissions."""
    probabilities = torch.tensor(record["final_bio_probs"], dtype=torch.float32)
    probabilities = probabilities.clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "id": record.get("id", ""),
        "gold_bio_ids": [int(item) for item in record["gold_bio_ids"]],
        "initial_bio_ids": [int(item) for item in record["initial_bio_ids"]],
        "pred_bio_ids": [int(item) for item in record["pred_bio_ids"]],
        "gold_document_stance": record["gold_document_stance"],
        "pred_document_stance": record.get("pred_document_stance"),
        "au_stance_predictions": record.get("au_stance_predictions", []),
        "pred_argument_units": record.get("pred_argument_units", []),
        "_log_prob_emissions": probabilities.log(),
    }


def load_light_records(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(_light_record(json.loads(line)))
            except Exception as exc:
                raise ValueError("Invalid record at {}:{}".format(path, line_number)) from exc
    if not records:
        raise ValueError("Prediction file is empty: {}".format(path))
    return records


def decode_with_bias(
    records: Sequence[Dict[str, object]],
    crf: LinearChainCRF,
    bias: float,
    batch_size: int,
) -> List[List[int]]:
    if batch_size < 1:
        raise ValueError("--decode_batch_size must be >= 1")
    all_paths: List[List[int]] = []
    for offset in range(0, len(records), batch_size):
        chunk = records[offset : offset + batch_size]
        maximum_length = max(
            int(record["_log_prob_emissions"].shape[0]) for record in chunk
        )
        emissions = torch.zeros(
            (len(chunk), maximum_length, len(BIO_LABELS)), dtype=torch.float32
        )
        mask = torch.zeros((len(chunk), maximum_length), dtype=torch.bool)
        for index, record in enumerate(chunk):
            values = record["_log_prob_emissions"]
            length = int(values.shape[0])
            emissions[index, :length] = values
            mask[index, :length] = True
        emissions[:, :, 0] += float(bias) * mask.to(dtype=emissions.dtype)
        decoded = crf.decode(emissions, mask=mask)
        all_paths.extend(decoded)
    return all_paths


def _metric_records(
    base_records: Sequence[Dict[str, object]], paths: Sequence[Sequence[int]]
) -> List[Dict[str, object]]:
    calibrated = []
    for base, path in zip(base_records, paths):
        item = {key: value for key, value in base.items() if not key.startswith("_")}
        path = [int(label) for label in path]
        item["pred_bio_ids"] = path
        item["pred_argument_units"] = [
            {
                "start": int(span["start"]),
                "end": int(span["end"]),
                "final_stance": str(span["stance"]),
            }
            for span in bio_to_spans(path)
        ]
        calibrated.append(item)
    return calibrated


def metrics_for_bias(
    records: Sequence[Dict[str, object]],
    crf: LinearChainCRF,
    bias: float,
    batch_size: int,
) -> Tuple[Dict[str, float], List[List[int]]]:
    paths = decode_with_bias(records, crf, bias=bias, batch_size=batch_size)
    metrics = compute_all_metrics(_metric_records(records, paths))
    return metrics, paths


def _full_span(span: Dict[str, object], record: Dict[str, object]) -> Dict[str, object]:
    start, end = int(span["start"]), int(span["end"])
    offsets = record.get("wordpiece_offsets", [])
    text = str(record.get("text", ""))
    if 0 <= start < end <= len(offsets):
        char_start = int(offsets[start][0])
        char_end = int(offsets[end - 1][1])
    else:
        char_start = char_end = 0
    return {
        "start": start,
        "end": end,
        "wordpiece_start": start,
        "wordpiece_end": end,
        "original_token_start": None,
        "original_token_end": None,
        "char_start": char_start,
        "char_end": char_end,
        "text": text[char_start:char_end],
        "final_stance": str(span["stance"]),
    }


def save_calibrated_predictions(
    source_path: Path,
    destination_path: Path,
    paths: Sequence[Sequence[int]],
    bias: float,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    path_iterator = iter(paths)
    with source_path.open("r", encoding="utf8") as source, destination_path.open(
        "w", encoding="utf8"
    ) as destination:
        count = 0
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            calibrated_path = [int(label) for label in next(path_iterator)]
            probabilities = torch.tensor(
                record["final_bio_probs"], dtype=torch.float32
            ).clamp_min(torch.finfo(torch.float32).tiny)
            biased_logits = probabilities.log()
            biased_logits[:, 0] += float(bias)
            calibrated_probs = torch.softmax(biased_logits, dim=-1).tolist()
            record["uncalibrated_final_bio_ids"] = record["final_bio_ids"]
            record["o_emission_bias"] = float(bias)
            record["final_bio_ids"] = calibrated_path
            record["pred_bio_ids"] = calibrated_path
            record["final_bio"] = [BIO_LABELS[label] for label in calibrated_path]
            record["final_bio_probs"] = calibrated_probs
            record["official_pred_labels"] = collapse_bio_sequence(calibrated_path)
            record["pred_argument_units"] = [
                _full_span(span, record) for span in bio_to_spans(calibrated_path)
            ]
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        try:
            next(path_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("Decoded path count exceeds source prediction count")
    if count != len(paths):
        raise ValueError(
            "Source prediction count {} differs from decoded path count {}".format(
                count, len(paths)
            )
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    paths = resolve_paths(args)
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("experiment_dir=%s", paths["experiment_dir"])
    LOGGER.info("output_dir=%s", output_dir)

    biases = build_bias_grid(args.bias_min, args.bias_max, args.bias_step)
    crf = load_crf(paths["checkpoint"])
    dev_records = load_light_records(paths["dev"])
    LOGGER.info("Loaded %d Dev records; scanning %d bias values", len(dev_records), len(biases))

    scan_rows = []
    dev_paths_by_bias = {}
    for bias in biases:
        metrics, decoded_paths = metrics_for_bias(
            dev_records, crf, bias=bias, batch_size=args.decode_batch_size
        )
        row = {"o_emission_bias": float(bias), **metrics}
        scan_rows.append(row)
        dev_paths_by_bias[float(bias)] = decoded_paths
        LOGGER.info(
            "Dev beta=%+.3f | Official Token F1=%.6f | Final BIO F1=%.6f "
            "| AU Span P/R/F1=%.6f/%.6f/%.6f",
            bias,
            metrics["official_token_macro_f1"],
            metrics["final_bio_token_macro_f1"],
            metrics["au_span_precision"],
            metrics["au_span_recall"],
            metrics["au_span_f1"],
        )

    best_row = max(
        scan_rows,
        key=lambda row: (
            float(row["official_token_macro_f1"]),
            -abs(float(row["o_emission_bias"])),
            -float(row["o_emission_bias"]),
        ),
    )
    best_bias = float(best_row["o_emission_bias"])
    baseline_row = next(
        (row for row in scan_rows if math.isclose(row["o_emission_bias"], 0.0)),
        None,
    )
    if baseline_row is None:
        baseline_metrics, _ = metrics_for_bias(
            dev_records, crf, bias=0.0, batch_size=args.decode_batch_size
        )
        baseline_row = {"o_emission_bias": 0.0, **baseline_metrics}

    test_records = load_light_records(paths["test"])
    test_baseline, _ = metrics_for_bias(
        test_records, crf, bias=0.0, batch_size=args.decode_batch_size
    )
    test_best, test_best_paths = metrics_for_bias(
        test_records, crf, bias=best_bias, batch_size=args.decode_batch_size
    )
    dev_best_paths = dev_paths_by_bias.get(best_bias)
    if dev_best_paths is None:
        _, dev_best_paths = metrics_for_bias(
            dev_records, crf, bias=best_bias, batch_size=args.decode_batch_size
        )

    fieldnames = sorted({key for row in scan_rows for key in row})
    with (output_dir / "dev_o_bias_scan.csv").open(
        "w", encoding="utf8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scan_rows)
    with (output_dir / "dev_o_bias_scan.json").open("w", encoding="utf8") as handle:
        json.dump(scan_rows, handle, ensure_ascii=False, indent=2)

    summary = {
        "selection_metric": "dev_official_token_macro_f1",
        "retraining_performed": False,
        "checkpoint": str(paths["checkpoint"]),
        "dev_predictions_source": str(paths["dev"]),
        "test_predictions_source": str(paths["test"]),
        "bias_grid": biases,
        "selected_o_emission_bias": best_bias,
        "dev": {
            "baseline": baseline_row,
            "calibrated": best_row,
            "official_token_f1_delta": float(best_row["official_token_macro_f1"])
            - float(baseline_row["official_token_macro_f1"]),
        },
        "test": {
            "baseline": test_baseline,
            "calibrated": test_best,
            "official_token_f1_delta": float(test_best["official_token_macro_f1"])
            - float(test_baseline["official_token_macro_f1"]),
        },
    }
    with (output_dir / "o_emission_bias_calibration.json").open(
        "w", encoding="utf8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.save_predictions:
        save_calibrated_predictions(
            paths["dev"],
            output_dir / "dev_calibrated.jsonl",
            dev_best_paths,
            best_bias,
        )
        save_calibrated_predictions(
            paths["test"],
            output_dir / "test_calibrated.jsonl",
            test_best_paths,
            best_bias,
        )

    LOGGER.info(
        "Selected beta=%+.3f on Dev: Official Token F1 %.6f -> %.6f (%+.6f)",
        best_bias,
        baseline_row["official_token_macro_f1"],
        best_row["official_token_macro_f1"],
        summary["dev"]["official_token_f1_delta"],
    )
    LOGGER.info(
        "Fixed-beta Test: Official Token F1 %.6f -> %.6f (%+.6f)",
        test_baseline["official_token_macro_f1"],
        test_best["official_token_macro_f1"],
        summary["test"]["official_token_f1_delta"],
    )
    LOGGER.info("Calibration results saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
