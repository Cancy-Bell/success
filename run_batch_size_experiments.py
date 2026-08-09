#!/usr/bin/env python
"""Run AURC batch-size experiments sequentially without overwriting results.

Run this file with the ``aurc`` Conda interpreter. By default it executes the
official entry point first with batch size 32 and, only after a successful
exit, with batch size 64. Each experiment receives an independent output
directory and a complete UTF-8 console log.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--target_domain", default="In-Domain")
    parser.add_argument(
        "--bert_path",
        default=r"D:\software\huggingface\bert-large-cased-whole-word-masking",
    )
    parser.add_argument(
        "--output_root",
        default=str(project_root / "models" / "batch_size_comparison"),
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional result-folder name; defaults to a fresh timestamped run.",
    )
    parser.add_argument("--max_sequence_length", type=int, default=64)
    parser.add_argument("--topic_max_length", type=int, default=16)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument(
        "--initial_o_bias",
        type=float,
        default=0.325,
        help="Initial learnable bias added to the Final-CRF O emission.",
    )
    return parser.parse_args()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def build_command(
    project_root: Path,
    output_dir: Path,
    batch_size: int,
    args: argparse.Namespace,
) -> List[str]:
    return [
        sys.executable,
        str(project_root / "src" / "run_AURC_token.py"),
        "--train",
        "--crf",
        "--target_domain",
        args.target_domain,
        "--data_dir",
        str(project_root / "data"),
        "--input_file",
        "AURC_DATA_dict.json",
        "--output_dir",
        str(output_dir),
        "--pretrained_weights",
        str(Path(args.bert_path).resolve()),
        "--local_files_only",
        "--epochs",
        str(args.epochs),
        "--max_sequence_length",
        str(args.max_sequence_length),
        "--topic_max_length",
        str(args.topic_max_length),
        # Conservative fine-tuning settings for the small AURC training set.
        "--learning_rate",
        "5e-6",
        "--weight_decay",
        "0.02",
        "--warmup_ratio",
        "0.1",
        "--bert_dropout",
        "0.2",
        "--train_batch_size",
        str(batch_size),
        "--eval_batch_size",
        str(batch_size),
        "--test_batch_size",
        str(batch_size),
        "--spacy_model",
        args.spacy_model,
        "--gcn_layers",
        "2",
        "--gcn_dropout",
        "0.2",
        "--hetgat_layers",
        "2",
        "--hetgat_heads",
        "1",
        "--hetgat_dropout",
        "0.2",
        "--au_semantic_threshold",
        "0.5",
        "--au_top_k",
        "3",
        "--au_syntax_hops",
        "1",
        "--early_stop_patience",
        "5",
        "--early_stop_min_delta",
        "0.0",
        "--initial_crf_loss_weight",
        "0.3",
        "--official_token_loss_weight",
        "0.5",
        "--initial_o_bias",
        str(args.initial_o_bias),
        "--save_predictions",
        "--save_alignment_debug",
        "--preprocess_log_every",
        "250",
    ]


def run_experiment(
    command: List[str],
    project_root: Path,
    output_dir: Path,
    batch_size: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "console.log"
    start_time = time.time()
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    print("=" * 80, flush=True)
    print("Starting physical batch_size={}".format(batch_size), flush=True)
    print("Output: {}".format(output_dir), flush=True)
    print("Log: {}".format(log_path), flush=True)
    print("=" * 80, flush=True)

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    # Do not set expandable_segments on Windows: the Windows CUDA allocator in
    # the supported PyTorch build reports that option as unsupported.
    log_handle = None
    log_error = None
    last_log_flush = time.monotonic()
    try:
        # A large userspace buffer plus periodic flushes avoids the Windows/H:
        # drive Errno 22 observed when flushing the file after every log line.
        log_handle = log_path.open("w", encoding="utf8", buffering=1024 * 1024)
        log_handle.write("COMMAND:\n{}\n\n".format(subprocess.list2cmdline(command)))
    except OSError as error:
        log_error = "Could not open console log: {!r}".format(error)
        print("WARNING: {}".format(log_error), flush=True)

    process = subprocess.Popen(
        command,
        cwd=str(project_root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            # This launcher writes everything to stdout, which PyCharm renders
            # with its normal (white) console color.
            print(line, end="", flush=True)
            if log_handle is None:
                continue
            try:
                log_handle.write(line)
                if time.monotonic() - last_log_flush >= 30.0:
                    log_handle.flush()
                    last_log_flush = time.monotonic()
            except OSError as error:
                # A secondary console mirror must never terminate expensive
                # model training. Metrics/predictions/checkpoints are written
                # independently by run_AURC_token.py.
                log_error = "Console log disabled after write failure: {!r}".format(
                    error
                )
                print("WARNING: {}".format(log_error), flush=True)
                try:
                    log_handle.close()
                except OSError:
                    pass
                log_handle = None
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    finally:
        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.close()
            except OSError as error:
                log_error = "Console log close failed: {!r}".format(error)
                print("WARNING: {}".format(log_error), flush=True)
    return_code = process.wait()

    finished_at = dt.datetime.now().isoformat(timespec="seconds")
    result: Dict[str, object] = {
        "batch_size": batch_size,
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(time.time() - start_time, 1),
        "output_dir": str(output_dir),
        "console_log": str(log_path),
        "console_log_complete": log_error is None,
        "console_log_error": log_error,
        "command": command,
    }
    final_metrics_path = output_dir / "final_best_metrics.json"
    if final_metrics_path.is_file():
        with final_metrics_path.open("r", encoding="utf8") as handle:
            result["final_best_metrics"] = json.load(handle)
    save_json(output_dir / "experiment_result.json", result)
    return result


def main() -> int:
    args = parse_args()
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
        "batch_sizes": args.batch_sizes,
        "regularization": {
            "learning_rate": 5e-6,
            "weight_decay": 0.02,
            "warmup_ratio": 0.1,
            "bert_dropout": 0.2,
            "gcn_dropout": 0.2,
            "hetgat_dropout": 0.2,
            "hetgat_heads": 1,
            "initial_crf_loss_weight": 0.3,
            "official_token_loss_weight": 0.5,
            "initial_o_bias": args.initial_o_bias,
        },
        "experiments": [],
    }
    save_json(
        output_root / "latest_run.json",
        {"run_name": run_name, "run_dir": str(run_dir)},
    )

    for batch_size in args.batch_sizes:
        output_dir = run_dir / "batch_size_{}".format(batch_size)
        command = build_command(project_root, output_dir, batch_size, args)
        result = run_experiment(
            command=command,
            project_root=project_root,
            output_dir=output_dir,
            batch_size=batch_size,
        )
        summary["experiments"].append(result)
        save_json(run_dir / "batch_size_comparison.json", summary)
        if result["return_code"] != 0:
            print(
                "batch_size={} failed; preserving its log and stopping before "
                "the next experiment.".format(batch_size),
                flush=True,
            )
            return int(result["return_code"])
        print(
            "batch_size={} completed and saved. Starting the next configured "
            "batch size if present.".format(batch_size),
            flush=True,
        )

    print("All batch-size experiments completed.", flush=True)
    print("Comparison summary: {}".format(run_dir / "batch_size_comparison.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
