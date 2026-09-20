from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.refute_overlay import expected_answer_kind


def _compact(row: pd.Series) -> dict[str, Any]:
    try:
        return json.loads(str(row.get("evidence_compact_json", "{}") or "{}"))
    except json.JSONDecodeError:
        return {}


def _num(row: pd.Series, column: str) -> float:
    try:
        value = row.get(column, 0.0)
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def report(scored_csv: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(scored_csv)
    changed = frame[frame.get("refute_overlay_delta", pd.Series([0.0] * len(frame))).fillna(0).astype(float) > 0].copy()
    rows: list[dict[str, Any]] = []
    accepted_reasons: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    titles: Counter[str] = Counter()
    hard_failures: list[str] = []
    no_conflict_rows = 0
    unknown_relation_fires = 0
    unsupported_count_unit_fires = 0
    answer_supported_fires = 0

    for idx, row in changed.iterrows():
        compact = _compact(row)
        conflicts = [item for item in compact.get("relation_conflicts", []) if isinstance(item, dict)]
        rejected = [item for item in compact.get("rejected_relation_candidates", []) if isinstance(item, dict)]
        expected_kind = expected_answer_kind(str(row.get("prompt", "")))
        relation_kind = str(compact.get("prompt_relation_kind", "unknown"))
        kinds[expected_kind] += 1
        if not conflicts:
            no_conflict_rows += 1
            hard_failures.append(f"row_{idx}:missing_relation_conflicts")
        if relation_kind == "unknown":
            has_exact_subject = any(str(item.get("reason", "")).endswith("single_year") for item in conflicts)
            if not has_exact_subject:
                unknown_relation_fires += 1
                hard_failures.append(f"row_{idx}:unknown_relation_without_exact_subject_support")
        for item in conflicts:
            accepted_reasons[str(item.get("reason", "unknown"))] += 1
            titles[str(item.get("title", ""))] += 1
            if item.get("kind") == "count" and not item.get("unit_terms"):
                unsupported_count_unit_fires += 1
                hard_failures.append(f"row_{idx}:count_overlay_without_unit_terms")
        for item in rejected:
            rejected_reasons[str(item.get("reason", "unknown"))] += 1
            if item.get("reason") == "reject_answer_value_supported":
                answer_supported_fires += 1
        rows.append(
            {
                "row_index": idx,
                "prompt": row.get("prompt"),
                "model_answer": row.get("model_answer"),
                "is_hallucination": row.get("is_hallucination"),
                "is_hallucination_proba": row.get("is_hallucination_proba"),
                "base_is_hallucination_proba": row.get("base_is_hallucination_proba"),
                "refute_overlay_delta": row.get("refute_overlay_delta"),
                "refute_overlay_reason": row.get("refute_overlay_reason"),
                "expected_kind": expected_kind,
                "relation_kind": relation_kind,
                "relation_conflicts": json.dumps(conflicts, ensure_ascii=False),
                "rejected_relation_candidates": json.dumps(rejected, ensure_ascii=False),
                "overlay_titles": " | ".join(str(item.get("title", "")) for item in conflicts),
                "conflict_windows": " || ".join(str(item.get("window", "")) for item in conflicts),
            }
        )

    labels = pd.to_numeric(changed.get("is_hallucination"), errors="coerce") if not changed.empty else pd.Series(dtype=float)
    true_positives = int((labels == 1).sum())
    false_positives = int((labels == 0).sum())
    report_data = {
        "total_overlay_changed_rows": int(len(changed)),
        "changed_precision": float(true_positives / max(1, len(changed))),
        "changed_false_positives": false_positives,
        "changed_true_positives": true_positives,
        "changed_rows_by_expected_kind": dict(kinds),
        "accepted_relation_conflict_reasons": dict(accepted_reasons),
        "rejected_relation_candidate_reasons": dict(rejected_reasons),
        "rows_where_overlay_fired_with_no_relation_conflicts": int(no_conflict_rows),
        "numeric_year_fp_titles": titles.most_common(30),
        "answer_value_supported_but_overlay_fired": int(answer_supported_fires),
        "relation_kind_unknown_overlay_fires": int(unknown_relation_fires),
        "count_overlay_without_unit_compatibility": int(unsupported_count_unit_fires),
        "hard_failure": bool(hard_failures),
        "hard_failures": hard_failures[:100],
    }
    return report_data, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report v10 relation-level refute overlay validity.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    report_data, rows = report(args.scored_csv)
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.to_csv(args.csv_output, index=False)
    print(json.dumps(report_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
