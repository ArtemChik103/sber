from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.evidence import FORBIDDEN_SOURCE_COLUMNS
from guardian_of_truth.utils import sha256_hexdigest


ALLOWED_COLUMNS = ("row_key", "prompt", "model_answer")


def export_queries(source: str | Path, output: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    missing = {"prompt", "model_answer"} - set(frame.columns)
    if missing:
        raise ValueError(f"Query export source is missing required columns: {sorted(missing)}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = pd.DataFrame(
        {
            "row_key": frame.apply(lambda row: sha256_hexdigest(row["prompt"], row["model_answer"]), axis=1),
            "prompt": frame["prompt"].fillna("").astype(str),
            "model_answer": frame["model_answer"].fillna("").astype(str),
        }
    )

    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for record in exported.to_dict(orient="records"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        exported.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    stripped = sorted((set(frame.columns) & FORBIDDEN_SOURCE_COLUMNS) | (set(frame.columns) - set(ALLOWED_COLUMNS) - {"row_key"}))
    return {
        "source": str(source),
        "output": str(output),
        "rows": int(len(exported)),
        "columns": list(ALLOWED_COLUMNS),
        "stripped_columns": stripped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unlabeled prompt/model_answer rows for retrieval-only fetching.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(json.dumps(export_queries(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
