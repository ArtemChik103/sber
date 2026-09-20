from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.utils import sha256_hexdigest


def _key(row: pd.Series) -> str:
    return sha256_hexdigest(row.get("prompt", ""), row.get("model_answer", ""))


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


def _aligned_titles(row: pd.Series) -> list[str]:
    compact = _compact(row)
    snippets = compact.get("overlay_eligible_snippets") or compact.get("aligned_snippets") or []
    return [str(item.get("title", "")) for item in snippets if isinstance(item, dict)]


def _alignment_reasons(row: pd.Series) -> list[str]:
    compact = _compact(row)
    snippets = compact.get("overlay_eligible_snippets") or compact.get("aligned_snippets") or []
    return [str(item.get("alignment_reason", "")) for item in snippets if isinstance(item, dict)]


def _ap(frame: pd.DataFrame, column: str) -> float | None:
    if "is_hallucination" not in frame.columns or column not in frame.columns:
        return None
    labels = pd.to_numeric(frame["is_hallucination"], errors="coerce")
    scores = pd.to_numeric(frame[column], errors="coerce")
    mask = labels.notna() & scores.notna()
    if labels[mask].nunique() < 2:
        return None
    return float(average_precision_score(labels[mask], scores[mask]))


def _category(before: pd.Series, after: pd.Series) -> str:
    label = int(_num(after, "is_hallucination"))
    before_lift = _num(before, "refute_overlay_delta") > 0
    after_lift = _num(after, "refute_overlay_delta") > 0
    before_score = _num(before, "is_hallucination_proba")
    after_score = _num(after, "is_hallucination_proba")
    before_base = _num(before, "base_is_hallucination_proba")
    after_base = _num(after, "base_is_hallucination_proba")
    if before_lift and not after_lift and label == 1:
        return "lost_good_overlay_lift"
    if after_lift and not before_lift and label == 0:
        return "new_bad_overlay_lift"
    if after_base + 1e-12 < before_base:
        return "base_score_regression"
    if after_base > before_base + 1e-12:
        return "base_score_improvement"
    if after_score != before_score and not after_lift:
        return "alignment_noise_no_overlay"
    return "stable"


def analyze(before_csv: Path, after_csv: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)
    before_by_key = {_key(row): row for _, row in before.iterrows()}
    rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    top_new_fp_titles: Counter[str] = Counter()
    top_lost_tp_titles: Counter[str] = Counter()
    grouped_regressions: dict[str, Counter[str]] = defaultdict(Counter)

    for _, after_row in after.iterrows():
        key = _key(after_row)
        before_row = before_by_key.get(key)
        if before_row is None:
            continue
        category = _category(before_row, after_row)
        before_reason = str(before_row.get("refute_overlay_reason", ""))
        after_reason = str(after_row.get("refute_overlay_reason", ""))
        transitions[f"{before_reason} -> {after_reason}"] += 1
        label = int(_num(after_row, "is_hallucination"))
        if category == "new_bad_overlay_lift":
            top_new_fp_titles.update(_aligned_titles(after_row))
        if category == "lost_good_overlay_lift":
            top_lost_tp_titles.update(_aligned_titles(before_row))
        if category == "base_score_regression":
            grouped_regressions["profile"][str(after_row.get("question_profile") or after_row.get("profile") or "unknown")] += 1
            grouped_regressions["score_path"][str(after_row.get("score_path") or "unknown")] += 1
            grouped_regressions["audit_status"][str(after_row.get("audit_status") or "unknown")] += 1
        rows.append(
            {
                "row_key": key,
                "category": category,
                "label": label if "is_hallucination" in after.columns else None,
                "prompt": after_row.get("prompt"),
                "model_answer": after_row.get("model_answer"),
                "profile": after_row.get("question_profile") or after_row.get("profile"),
                "score_path": after_row.get("score_path"),
                "audit_status": after_row.get("audit_status"),
                "before_score": _num(before_row, "is_hallucination_proba"),
                "after_score": _num(after_row, "is_hallucination_proba"),
                "before_base_score": _num(before_row, "base_is_hallucination_proba"),
                "after_base_score": _num(after_row, "base_is_hallucination_proba"),
                "before_overlay_delta": _num(before_row, "refute_overlay_delta"),
                "after_overlay_delta": _num(after_row, "refute_overlay_delta"),
                "before_overlay_reason": before_reason,
                "after_overlay_reason": after_reason,
                "before_aligned_titles": " | ".join(_aligned_titles(before_row)),
                "after_aligned_titles": " | ".join(_aligned_titles(after_row)),
                "before_alignment_reasons": " | ".join(_alignment_reasons(before_row)),
                "after_alignment_reasons": " | ".join(_alignment_reasons(after_row)),
            }
        )

    out = pd.DataFrame(rows)
    report = {
        "rows": int(len(out)),
        "before_final_ap": _ap(before, "is_hallucination_proba"),
        "after_final_ap": _ap(after, "is_hallucination_proba"),
        "before_base_ap": _ap(before, "base_is_hallucination_proba"),
        "after_base_ap": _ap(after, "base_is_hallucination_proba"),
        "rows_where_overlay_lift_was_lost": int(((out["before_overlay_delta"] > 0) & (out["after_overlay_delta"] <= 0)).sum()),
        "rows_where_new_overlay_lift_appeared": int(((out["before_overlay_delta"] <= 0) & (out["after_overlay_delta"] > 0)).sum()),
        "true_positive_lifts_lost": int(((out["category"] == "lost_good_overlay_lift")).sum()),
        "false_positive_lifts_added": int(((out["category"] == "new_bad_overlay_lift")).sum()),
        "category_counts": out["category"].value_counts().astype(int).to_dict(),
        "final_score_regressions_by_profile": dict(grouped_regressions["profile"]),
        "final_score_regressions_by_score_path": dict(grouped_regressions["score_path"]),
        "final_score_regressions_by_audit_status": dict(grouped_regressions["audit_status"]),
        "refute_overlay_reason_transitions": dict(transitions.most_common()),
        "top_new_false_positive_aligned_titles": top_new_fp_titles.most_common(30),
        "top_lost_true_positive_aligned_titles": top_lost_tp_titles.most_common(30),
    }
    return report, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v9 alignment/overlay score impact.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    report, rows = analyze(Path(args.before), Path(args.after))
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.to_csv(args.csv_output, index=False)


if __name__ == "__main__":
    main()
