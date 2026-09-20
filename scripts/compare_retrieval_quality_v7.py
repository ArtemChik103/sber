from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardian_of_truth.utils import sha256_hexdigest
from scripts.analyze_retrieval_quality_v7 import analyze


def _key_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["_row_key"] = frame.apply(lambda row: sha256_hexdigest(row.get("prompt", ""), row.get("model_answer", "")), axis=1)
    return frame


def compare(before: str | Path, after: str | Path) -> tuple[dict, pd.DataFrame]:
    before_summary, _ = analyze(before, top_n=0)
    after_summary, _ = analyze(after, top_n=0)
    left = _key_frame(before)
    right = _key_frame(after)
    merged = left.merge(right, on="_row_key", suffixes=("_before", "_after"))
    merged["_hit_delta"] = merged.get("evidence_retrieval_hit_count_after", 0).fillna(0).astype(float) - merged.get(
        "evidence_retrieval_hit_count_before", 0
    ).fillna(0).astype(float)
    merged["_overlap_delta"] = merged.get("evidence_top_evidence_overlap_after", 0).fillna(0).astype(float) - merged.get(
        "evidence_top_evidence_overlap_before", 0
    ).fillna(0).astype(float)
    merged["_aligned_delta"] = merged.get("evidence_aligned_hit_count_after", 0).fillna(0).astype(float) - merged.get(
        "evidence_aligned_hit_count_before", 0
    ).fillna(0).astype(float)
    changed = merged[
        (merged["_hit_delta"] != 0)
        | (merged["_aligned_delta"] != 0)
        | (merged["_overlap_delta"].abs() >= 0.05)
        | (merged.get("is_hallucination_proba_before", 0).astype(float) != merged.get("is_hallucination_proba_after", 0).astype(float))
    ].copy()
    summary = {
        "before": before_summary,
        "after": after_summary,
        "bad_alignment_delta": int(after_summary["bad_alignment_count"] - before_summary["bad_alignment_count"]),
        "bad_alignment_reduction_ratio": float(
            (before_summary["bad_alignment_count"] - after_summary["bad_alignment_count"]) / max(1, before_summary["bad_alignment_count"])
        ),
        "false_negative_bad_alignment_delta": int(
            after_summary["false_negatives_with_bad_alignment"] - before_summary["false_negatives_with_bad_alignment"]
        ),
        "changed_rows": int(len(changed)),
    }
    columns = [
        "prompt_before",
        "model_answer_before",
        "is_hallucination_before",
        "is_hallucination_proba_before",
        "is_hallucination_proba_after",
        "evidence_status_before",
        "evidence_status_after",
        "evidence_retrieval_hit_count_before",
        "evidence_retrieval_hit_count_after",
        "evidence_top_evidence_overlap_before",
        "evidence_top_evidence_overlap_after",
        "evidence_aligned_hit_count_before",
        "evidence_aligned_hit_count_after",
        "_hit_delta",
        "_overlap_delta",
        "_aligned_delta",
        "evidence_compact_json_before",
        "evidence_compact_json_after",
    ]
    existing = [column for column in columns if column in changed.columns]
    return summary, changed[existing]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v7 retrieval quality before and after.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    summary, changed = compare(args.before, args.after)
    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    changed.to_csv(csv_path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
