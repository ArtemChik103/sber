from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardian_of_truth.evidence import build_kb


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local SQLite FTS5 evidence KB from JSONL/CSV sources.")
    parser.add_argument("--source", action="append", required=True, help="JSONL/CSV source. May be passed multiple times.")
    parser.add_argument("--db-path", default="model/evidence.db")
    parser.add_argument("--kb-version", default="local_v1")
    parser.add_argument("--min-chars", type=int, default=150)
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()

    summary = build_kb(
        [Path(path) for path in args.source],
        args.db_path,
        kb_version=args.kb_version,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    print(json.dumps({"db_path": args.db_path, "kb_version": args.kb_version, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
