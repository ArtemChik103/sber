from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.evidence import (
    count_unit_from_prompt,
    count_unit_hit,
    numeric_values_in_window,
    predicate_hit,
    relation_kind_from_prompt,
    split_relation_windows,
    subject_anchor_hit,
)
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


def _category(row: pd.Series, decisions: list[str]) -> str:
    label = _num(row, "is_hallucination")
    accepted = "accept_relation_conflict" in decisions
    if label == 1 and accepted:
        return "true_positive_valid"
    if label == 1:
        return "true_positive_accidental"
    if accepted:
        return "false_positive_ambiguous"
    return "false_positive_invalid"


def _decision_for_window(
    *,
    prompt: str,
    answer: str,
    expected_kind: str,
    relation_kind: str,
    title: str,
    window: str,
    values: dict[str, list[str]],
) -> tuple[str, str]:
    unit_terms = count_unit_from_prompt(prompt, answer)
    pred = predicate_hit(relation_kind, window)
    anchored = subject_anchor_hit(prompt, title, window)
    if not anchored:
        return "reject_no_predicate", "missing_subject_anchor"
    if expected_kind in {"year", "date_or_month"}:
        if len(values["years"]) > 3 and not pred:
            return "reject_many_dates", "broad_year_list"
        if not pred and relation_kind != "unknown":
            return "reject_no_predicate", "predicate_mismatch"
        if relation_kind == "unknown" and len(values["years"]) != 1:
            return "reject_contextual_year", "ambiguous_unknown_relation"
        return "accept_relation_conflict", "subject_predicate_year"
    if expected_kind == "count":
        if not count_unit_hit(unit_terms, window):
            return "reject_wrong_unit", "unit_mismatch"
        if values["years"]:
            return "reject_wrong_unit", "date_or_year_number"
        if len(values["numbers"]) > 5:
            return "reject_broad_list", "many_numbers"
        return "accept_relation_conflict", "compatible_count_unit"
    return "reject_no_predicate", "non_numeric_expected_kind"


def analyze(scored_csv: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(scored_csv)
    changed = frame[frame.get("refute_overlay_delta", pd.Series([0.0] * len(frame))).fillna(0).astype(float) > 0].copy()
    rows: list[dict[str, Any]] = []
    decisions: list[str] = []

    for idx, row in changed.iterrows():
        compact = _compact(row)
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("model_answer", ""))
        expected_kind = expected_answer_kind(prompt)
        relation_kind = relation_kind_from_prompt(prompt)
        answer_values = compact.get("answer_refute_values") or {}
        snippets = compact.get("overlay_eligible_snippets") or compact.get("aligned_snippets") or []
        row_decisions: list[str] = []
        for snippet in snippets:
            if not isinstance(snippet, dict):
                continue
            title = str(snippet.get("title", ""))
            for window in split_relation_windows(str(snippet.get("text", ""))):
                values = numeric_values_in_window(window)
                if not values["years"] and not values["numbers"]:
                    continue
                decision, reason = _decision_for_window(
                    prompt=prompt,
                    answer=answer,
                    expected_kind=expected_kind,
                    relation_kind=relation_kind,
                    title=title,
                    window=window,
                    values=values,
                )
                row_decisions.append(decision)
                decisions.append(decision)
                rows.append(
                    {
                        "row_index": idx,
                        "prompt": prompt,
                        "model_answer": answer,
                        "is_hallucination": row.get("is_hallucination"),
                        "expected_kind": expected_kind,
                        "answer_refute_values": json.dumps(answer_values, ensure_ascii=False),
                        "overlay_titles": title,
                        "evidence_candidates": json.dumps(values, ensure_ascii=False),
                        "window": window,
                        "detected_predicate_class": relation_kind,
                        "detected_count_unit_kind": "compatible" if count_unit_hit(count_unit_from_prompt(prompt, answer), window) else "unknown_or_wrong",
                        "current_overlay_reason": row.get("refute_overlay_reason"),
                        "proposed_relation_decision": decision,
                        "decision_reason": reason,
                    }
                )
        if not row_decisions:
            rows.append(
                {
                    "row_index": idx,
                    "prompt": prompt,
                    "model_answer": answer,
                    "is_hallucination": row.get("is_hallucination"),
                    "expected_kind": expected_kind,
                    "answer_refute_values": json.dumps(answer_values, ensure_ascii=False),
                    "overlay_titles": " | ".join(str(item.get("title", "")) for item in snippets if isinstance(item, dict)),
                    "evidence_candidates": "{}",
                    "window": "",
                    "detected_predicate_class": relation_kind,
                    "detected_count_unit_kind": "",
                    "current_overlay_reason": row.get("refute_overlay_reason"),
                    "proposed_relation_decision": "reject_no_predicate",
                    "decision_reason": "no_numeric_relation_window",
                }
            )
            row_decisions.append("reject_no_predicate")
        rows[-1]["category"] = _category(row, row_decisions)

    out = pd.DataFrame(rows)
    report = {
        "overlay_rows": int(len(changed)),
        "diagnostic_rows": int(len(out)),
        "decision_counts": out.get("proposed_relation_decision", pd.Series(dtype=str)).value_counts().astype(int).to_dict(),
        "category_counts": out.get("category", pd.Series(dtype=str)).value_counts().astype(int).to_dict(),
    }
    return report, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v9 numeric/year refute validity with v10 relation taxonomy.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    report, rows = analyze(args.scored_csv)
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.to_csv(args.csv_output, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
