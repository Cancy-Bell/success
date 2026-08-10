#!/usr/bin/env python
"""Convert an AURC prediction JSONL file to the official topic-grouped JSON."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prediction_io import write_aurc_prediction_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_jsonl", help="Input prediction JSONL path.")
    parser.add_argument(
        "--aurc_data",
        default=str(ROOT / "data" / "AURC_DATA_dict.json"),
        help="Official AURC_DATA_dict.json used as the row template.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path; defaults to <input_stem>_aurc.json.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON at {} line {}: {}".format(path, line_number, error)
                ) from error
    return records


def main() -> int:
    args = parse_args()
    prediction_path = Path(args.prediction_jsonl).resolve()
    aurc_path = Path(args.aurc_data).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else prediction_path.with_name(prediction_path.stem + "_aurc.json")
    )
    with aurc_path.open("r", encoding="utf8") as handle:
        aurc_data = json.load(handle)
    records = read_jsonl(prediction_path)
    write_aurc_prediction_json(str(output_path), aurc_data, records)
    print("Converted {} prediction records.".format(len(records)))
    print("Official-format output: {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
