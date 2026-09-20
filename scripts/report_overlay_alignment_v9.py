from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.refute_overlay import expected_answer_kind


def _num(row: pd.Series, column: str) -> float:
    try:
        value = row.get(column, 0.0)
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compact(row: pd.Series) -> dict[str, Any]:
    try:
        return json.loads(str(row.get("evidence_compact_json", "{}") or "{}"))
    except json.JSONDecodeError:
        return {}


def _overlay_snippets(row: pd.Series) -> list[dict[str, Any]]:
    snippets = _compact(row).get("overlay_eligible_snippets", [])
    return [item for item in snippets if isinstance(item, dict)]


def report(scored_csv: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(scored_csv)
    changed = frame[frame["refute_overlay_delta"].fillna(0).astype(float) > 0].copy()
    rows: list[dict[str, Any]] = []
    tiers: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    titles: Counter[str] = Counter()
    hard_failures: list[str] = []
    no_strong_support = 0

    for idx, row in changed.iterrows():
        snippets = _overlay_snippets(row)
        kind = expected_answer_kind(str(row.get("prompt", "")))
        kinds[kind] += 1
        if not snippets:
            hard_failures.append(f"row_{idx}:missing_overlay_eligible_snippets")
        row_tiers = [str(item.get("alignment_tier", "")) for item in snippets]
        row_reasons = [str(item.get("alignment_reason", "")) for item in snippets]
        row_titles = [str(item.get("title", "")) for item in snippets]
        tiers.update(row_tiers or ["missing"])
        reasons.update(row_reasons or ["missing"])
        titles.update(row_titles)
        if any(tier != "strong" for tier in row_tiers) or not row_tiers:
            hard_failures.append(f"row_{idx}:non_strong_overlay_tier")
        if not any(
            "title" in reason
            or "entity" in reason
            or "quoted" in reason
            for reason in row_reasons
        ):
            no_strong_support += 1
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
                "expected_kind": kind,
                "alignment_tier": " | ".join(row_tiers),
                "alignment_reason": " | ".join(row_reasons),
                "overlay_titles": " | ".join(row_titles),
                "overlay_snippet_count": len(snippets),
            }
        )

    out = pd.DataFrame(rows)
    labels = pd.to_numeric(changed.get("is_hallucination"), errors="coerce") if not changed.empty else pd.Series(dtype=float)
    true_positives = int((labels == 1).sum())
    false_positives = int((labels == 0).sum())
    numeric_fp = int(((labels == 0) & changed["refute_overlay_expected_kind"].astype(str).isin({"year", "date_or_month", "count"})).sum()) if not changed.empty and "refute_overlay_expected_kind" in changed else false_positives
    report_data = {
        "total_overlay_changed_rows": int(len(changed)),
        "changed_precision": float(true_positives / max(1, len(changed))),
        "changed_false_positives": false_positives,
        "changed_true_positives": true_positives,
        "changed_rows_by_expected_kind": dict(kinds),
        "changed_rows_by_alignment_tier": dict(tiers),
        "changed_rows_by_alignment_reason": dict(reasons),
        "numeric_year_false_positives": numeric_fp,
        "top_overlay_triggering_titles": titles.most_common(30),
        "rows_where_overlay_fired_without_strong_title_entity_support": int(no_strong_support),
        "hard_failure": bool(hard_failures),
        "hard_failures": hard_failures[:50],
    }
    return report_data, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Report v9 overlay alignment safety.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    report_data, rows = report(Path(args.scored_csv))
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.to_csv(args.csv_output, index=False)
    print(json.dumps(report_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
