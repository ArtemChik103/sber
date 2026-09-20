from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _latency(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in frame.columns or frame[column].empty:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    series = frame[column].astype(float)
    return {
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
    }


def report_overlay_sanity(scored_csv: str | Path, changed_output: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(scored_csv)
    delta = frame.get("refute_overlay_delta", pd.Series([0.0] * len(frame))).astype(float)
    changed = frame[delta > 1e-12].copy()
    columns = [
        "prompt",
        "model_answer",
        "base_is_hallucination_proba",
        "is_hallucination_proba",
        "refute_overlay_delta",
        "refute_overlay_policy",
        "refute_overlay_reason",
        "refute_overlay_expected_kind",
        "evidence_retrieval_hit_count",
        "evidence_top_evidence_overlap",
        "evidence_aligned_hit_count",
        "evidence_aligned_year_refuted_count",
        "evidence_aligned_number_refuted_count",
        "evidence_answer_year_in_prompt",
        "evidence_answer_number_in_prompt",
        "evidence_compact_json",
    ]
    existing = [column for column in columns if column in changed.columns]
    changed_path = Path(changed_output)
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    changed[existing].to_csv(changed_path, index=False)
    return {
        "scored_csv": str(scored_csv),
        "rows": int(len(frame)),
        "changed_rows": int(len(changed)),
        "changed_ratio": float(len(changed) / max(1, len(frame))),
        "never_decreased": bool((delta >= -1e-12).all()),
        "reason_counts": frame.get("refute_overlay_reason", pd.Series(dtype=str)).value_counts().astype(int).to_dict(),
        "expected_kind_counts": frame.get("refute_overlay_expected_kind", pd.Series(dtype=str)).value_counts().astype(int).to_dict(),
        "latency": {
            "total": _latency(frame, "t_total_sec"),
            "model": _latency(frame, "t_model_sec"),
            "overhead": _latency(frame, "t_overhead_sec"),
        },
        "changed_output": str(changed_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report overlay sanity checks for scored CSVs with or without labels.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--changed-output", required=True)
    args = parser.parse_args()

    report = report_overlay_sanity(args.scored_csv, args.changed_output)
    path = Path(args.json_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
