#!/usr/bin/env python
"""Sequentially sweep the trainable Final-CRF O-emission bias.

This script is intentionally runnable without command-line arguments.  The
default experiment keeps physical batch size 32 and every other training
setting fixed, then trains initial O-bias values 0.25, 0.325, and 0.40.  Each
run has an independent result directory.  Model selection and sweep ranking
use Dev Official Token macro-F1 only; Test metrics are reported but never used
to select the bias.
"""

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

from run_batch_size_experiments import build_command, run_experiment, save_json


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial_o_bias_values",
        type=float,
        nargs="+",
        default=[0.25, 0.325, 0.40],
        help="Initial O-emission biases to train sequentially.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--target_domain", default="In-Domain")
    parser.add_argument(
        "--bert_path",
        default=r"D:\software\huggingface\bert-large-cased-whole-word-masking",
    )
    parser.add_argument(
        "--output_root",
        default=str(project_root / "models" / "o_bias_sweep"),
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional result-folder name; defaults to a fresh timestamped run.",
    )
    parser.add_argument("--max_sequence_length", type=int, default=64)
    parser.add_argument("--topic_max_length", type=int, default=16)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    return parser.parse_args()


def bias_directory_name(value: float) -> str:
    """Return a stable, Windows-safe directory component for a bias value."""
    text = format(value, ".12g")
    return "o_bias_{}".format(text.replace("-", "minus_").replace(".", "_"))


def metric_at(
    result: Dict[str, object], split: str, metric: str
) -> Optional[float]:
    final_metrics = result.get("final_best_metrics")
    if not isinstance(final_metrics, dict):
        return None
    splits = final_metrics.get("splits")
    if not isinstance(splits, dict):
        return None
    split_metrics = splits.get(split)
    if not isinstance(split_metrics, dict):
        return None
    value = split_metrics.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def comparison_rows(experiments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for experiment in experiments:
        final_metrics = experiment.get("final_best_metrics")
        best_epoch = None
        final_bias = None
        if isinstance(final_metrics, dict):
            best_epoch = final_metrics.get("best_epoch")
            final_bias = metric_at(experiment, "dev", "final_o_bias")
        rows.append(
            {
                "initial_o_bias": experiment.get("initial_o_bias"),
                "learned_o_bias": final_bias,
                "status": experiment.get("status"),
                "best_epoch": best_epoch,
                "dev_official_token_macro_f1": metric_at(
                    experiment, "dev", "official_token_macro_f1"
                ),
                "test_official_token_macro_f1": metric_at(
                    experiment, "test", "official_token_macro_f1"
                ),
                "test_official_segment_f1": metric_at(
                    experiment, "test", "official_segment_f1"
                ),
                "test_official_sentence_f1": metric_at(
                    experiment, "test", "official_sentence_f1"
                ),
                "test_final_bio_token_macro_f1": metric_at(
                    experiment, "test", "final_bio_token_macro_f1"
                ),
                "test_au_span_f1": metric_at(experiment, "test", "au_span_f1"),
                "test_au_stance_macro_f1": metric_at(
                    experiment, "test", "au_stance_macro_f1"
                ),
                "test_entity_f1": metric_at(experiment, "test", "entity_f1"),
                "output_dir": experiment.get("output_dir"),
            }
        )
    return rows


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_o_bias_command(command: List[str], initial_o_bias: float) -> None:
    if "--initial_o_bias" not in command:
        raise ValueError(
            "Generated command is missing --initial_o_bias for value {}".format(
                initial_o_bias
            )
        )
    option_index = command.index("--initial_o_bias")
    if option_index + 1 >= len(command):
        raise ValueError("--initial_o_bias is missing its numeric value.")
    actual_value = float(command[option_index + 1])
    if actual_value != float(initial_o_bias):
        raise ValueError(
            "Generated command has --initial_o_bias={} but expected {}".format(
                actual_value, initial_o_bias
            )
        )


def update_summary(run_dir: Path, summary: Dict[str, object]) -> None:
    experiments = summary["experiments"]
    assert isinstance(experiments, list)
    rows = comparison_rows(experiments)
    completed = [
        row
        for row in rows
        if row["status"] == "completed"
        and row["dev_official_token_macro_f1"] is not None
    ]
    if completed:
        best = max(completed, key=lambda row: row["dev_official_token_macro_f1"])
        summary["best_by_dev_official_token_f1"] = {
            "initial_o_bias": best["initial_o_bias"],
            "learned_o_bias": best["learned_o_bias"],
            "best_epoch": best["best_epoch"],
            "dev_official_token_macro_f1": best[
                "dev_official_token_macro_f1"
            ],
            "output_dir": best["output_dir"],
        }
    save_json(run_dir / "o_bias_comparison.json", summary)
    save_csv(run_dir / "o_bias_comparison.csv", rows)


def main() -> int:
    args = parse_args()
    if not args.initial_o_bias_values:
        raise ValueError("At least one --initial_o_bias_values value is required.")

    project_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    summary: Dict[str, object] = {
        "python": sys.executable,
        "project_root": str(project_root),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "selection_rule": "maximum Dev Official Token macro-F1",
        "batch_size": args.batch_size,
        "initial_o_bias_values": args.initial_o_bias_values,
        "fixed_settings": {
            "epochs": args.epochs,
            "learning_rate": 5e-6,
            "weight_decay": 0.02,
            "warmup_ratio": 0.1,
            "bert_dropout": 0.2,
            "gcn_dropout": 0.2,
            "hetgat_dropout": 0.2,
            "hetgat_heads": 1,
        },
        "loss_coefficients": {
            "final_crf": 1.0,
            "initial_crf": 1.0,
            "official_token": 1.0,
            "au_stance": 1.0,
            "document_stance": 1.0,
        },
        "experiments": [],
    }
    save_json(
        output_root / "latest_run.json",
        {"run_name": run_name, "run_dir": str(run_dir)},
    )
    update_summary(run_dir, summary)

    for initial_o_bias in args.initial_o_bias_values:
        bias_args = SimpleNamespace(
            target_domain=args.target_domain,
            bert_path=args.bert_path,
            epochs=args.epochs,
            max_sequence_length=args.max_sequence_length,
            topic_max_length=args.topic_max_length,
            spacy_model=args.spacy_model,
            initial_o_bias=initial_o_bias,
        )
        output_dir = run_dir / bias_directory_name(initial_o_bias)
        command = build_command(
            project_root, output_dir, args.batch_size, bias_args
        )
        validate_o_bias_command(command, initial_o_bias)
        print(
            "O-bias sweep: initial_o_bias={} ({}/{})".format(
                initial_o_bias,
                len(summary["experiments"]) + 1,
                len(args.initial_o_bias_values),
            ),
            flush=True,
        )
        result = run_experiment(
            command=command,
            project_root=project_root,
            output_dir=output_dir,
            batch_size=args.batch_size,
        )
        result["initial_o_bias"] = initial_o_bias
        save_json(output_dir / "experiment_result.json", result)
        summary["experiments"].append(result)
        update_summary(run_dir, summary)

        if result["return_code"] != 0:
            print(
                "initial_o_bias={} failed. Completed results and logs were "
                "preserved; stopping the sweep.".format(initial_o_bias),
                flush=True,
            )
            return int(result["return_code"])

    best = summary.get("best_by_dev_official_token_f1")
    print("All O-bias experiments completed.", flush=True)
    print("Comparison JSON: {}".format(run_dir / "o_bias_comparison.json"))
    print("Comparison CSV: {}".format(run_dir / "o_bias_comparison.csv"))
    if isinstance(best, dict):
        print(
            "Best initial_o_bias={} | Dev Official Token F1={:.6f}".format(
                best["initial_o_bias"], best["dev_official_token_macro_f1"]
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
