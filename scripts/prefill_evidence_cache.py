from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from guardian_of_truth.evidence import EvidenceRetriever


def prefill(csv_path: str | Path, db_path: str | Path, *, limit: int | None = None) -> dict[str, int]:
    frame = pd.read_csv(csv_path)
    forbidden = {"correct_answer", "comment"}
    present_forbidden = forbidden & set(frame.columns)
    if present_forbidden:
        raise ValueError(f"Evidence prefill source contains forbidden columns: {sorted(present_forbidden)}")
    required = {"prompt", "model_answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evidence prefill source is missing required columns: {sorted(missing)}")
    if limit is not None:
        frame = frame.head(limit)
    retriever = EvidenceRetriever(db_path)
    if not retriever.available:
        raise FileNotFoundError(f"Evidence DB does not exist: {db_path}")
    hits = 0
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="prefill-evidence"):
        snippets = retriever.retrieve(str(getattr(row, "prompt")), str(getattr(row, "model_answer")))
        hits += int(bool(snippets))
    return {"rows": int(len(frame)), "rows_with_evidence": int(hits)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill local evidence_cache for a scoring CSV without using labels.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--db-path", default="model/evidence.db")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(json.dumps(prefill(args.csv_path, args.db_path, limit=args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
