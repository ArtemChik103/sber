from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.entity_overlay import diagnostic_columns, parse_compact_json
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.refute_overlay import expected_answer_kind


BUCKETS = [(0.0, 0.1, "0-0.1"), (0.1, 0.25, "0.1-0.25"), (0.25, 0.5, "0.25-0.5"), (0.5, 1.01, "0.5-1.0")]


def score_bucket(value: float) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= value < hi:
            return label
    return "unknown"


def analyze_false_negatives(scored_csv: str | Path, *, csv_output: str | Path, top_n: int = 200) -> dict[str, Any]:
    frame = pd.read_csv(scored_csv)
    extractor = FeatureExtractor()
    if "_profile" not in frame.columns:
        frame["_profile"] = frame["prompt"].map(lambda value: extractor._question_profile(str(value)))
    _ensure_entity_diagnostics(frame)
    frame["_score_bucket_v5"] = frame["is_hallucination_proba"].astype(float).map(score_bucket)
    frame["_expected_kind_v5"] = frame["prompt"].map(lambda value: expected_answer_kind(str(value)))
    frame["_top_evidence_titles_v5"] = frame["evidence_compact_json"].fillna("{}").map(_top_titles)
    frame["_taxonomy_label_v5"] = frame.apply(_taxonomy_label, axis=1)

    false_negatives = frame[(frame["is_hallucination"] == 1) & (frame["is_hallucination_proba"] < 0.5)].copy()
    false_positives = frame[(frame["is_hallucination"] == 0) & (frame["is_hallucination_proba"] >= 0.5)].copy()

    output_columns = [
        "prompt",
        "model_answer",
        "is_hallucination",
        "is_hallucination_proba",
        "_profile",
        "score_path",
        "audit_status",
        "evidence_status",
        "evidence_retrieval_hit_count",
        "evidence_top_evidence_overlap",
        "evidence_core_supported",
        "evidence_core_refuted",
        "evidence_entity_refuted_count",
        "evidence_number_refuted_count",
        "evidence_year_refuted_count",
        "evidence_aligned_hit_count",
        "evidence_aligned_title_entity_match",
        "evidence_answer_core_entities",
        "evidence_aligned_evidence_entities",
        "evidence_answer_core_entity_count",
        "evidence_answer_core_entity_in_aligned_evidence_count",
        "evidence_answer_core_entity_missing_from_aligned_evidence_count",
        "evidence_aligned_evidence_entity_count",
        "evidence_aligned_alternative_entity_count",
        "evidence_aligned_entity_mismatch_candidate",
        "evidence_aligned_entity_confidence",
        "evidence_compact_json",
        "_expected_kind_v5",
        "_score_bucket_v5",
        "_top_evidence_titles_v5",
        "_taxonomy_label_v5",
    ]
    existing = [column for column in output_columns if column in false_negatives.columns]
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    false_negatives.sort_values("is_hallucination_proba", ascending=True)[existing].head(top_n).to_csv(csv_path, index=False)

    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "false_negatives": int(len(false_negatives)),
        "false_positives": int(len(false_positives)),
        "false_negatives_by_profile": false_negatives["_profile"].value_counts().astype(int).to_dict(),
        "false_negatives_by_score_path": false_negatives.get("score_path", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "false_negatives_by_audit_status": false_negatives.get("audit_status", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "false_negatives_by_evidence_status": false_negatives.get("evidence_status", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "false_negatives_by_score_bucket": false_negatives["_score_bucket_v5"].value_counts().sort_index().astype(int).to_dict(),
        "false_negatives_by_taxonomy_label": false_negatives["_taxonomy_label_v5"].value_counts().astype(int).to_dict(),
        "fn_evidence_core_refuted_gt0": int((pd.to_numeric(false_negatives.get("evidence_core_refuted", 0), errors="coerce").fillna(0) > 0).sum()),
        "fn_evidence_entity_refuted_count_gt0": int((pd.to_numeric(false_negatives.get("evidence_entity_refuted_count", 0), errors="coerce").fillna(0) > 0).sum()),
        "fn_evidence_aligned_title_entity_match_gt0": int((pd.to_numeric(false_negatives.get("evidence_aligned_title_entity_match", 0), errors="coerce").fillna(0) > 0).sum()),
        "csv_output": str(csv_path),
    }
    return summary


def _ensure_entity_diagnostics(frame: pd.DataFrame) -> None:
    if "evidence_answer_core_entity_count" in frame.columns and "evidence_answer_core_entities" in frame.columns:
        return
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            diagnostic_columns(
                str(row.get("prompt", "")),
                str(row.get("model_answer", "")),
                str(row.get("evidence_compact_json", "{}")),
                aligned_title_entity_match=_num(row, "evidence_aligned_title_entity_match"),
                top_overlap=_num(row, "evidence_top_evidence_overlap"),
            )
        )
    diag_frame = pd.DataFrame(rows, index=frame.index)
    for column in diag_frame.columns:
        frame[column] = diag_frame[column]


def _taxonomy_label(row: pd.Series) -> str:
    profile = str(row.get("_profile", "generic"))
    expected = str(row.get("_expected_kind_v5", "none"))
    if expected in {"year", "date_or_month", "count"} or profile in {"when", "count"}:
        return "numeric_already_handled_or_not_applicable"
    if profile == "generic" and _word_count(str(row.get("model_answer", ""))) > 20:
        return "unsupported_long_generic"
    if _num(row, "evidence_aligned_hit_count") < 1 or _num(row, "evidence_top_evidence_overlap") < 0.12:
        return "bad_retrieval_alignment"
    if profile == "title_name" and _num(row, "evidence_aligned_entity_mismatch_candidate") > 0:
        return "wrong_title_name_candidate"
    if profile == "where" and _num(row, "evidence_aligned_entity_mismatch_candidate") > 0:
        return "wrong_location_candidate"
    if profile in {"who", "by_whom"} and _num(row, "evidence_aligned_entity_mismatch_candidate") > 0:
        return "wrong_entity_candidate"
    if _num(row, "evidence_core_refuted") > 0 or _num(row, "evidence_claim_refuted_count") > 0:
        return "wrong_relation_candidate"
    return "unknown"


def _top_titles(value: str) -> str:
    compact = parse_compact_json(value)
    titles = []
    for group in ("snippets", "aligned_snippets"):
        for item in compact.get(group, []):
            if isinstance(item, dict) and item.get("title"):
                titles.append(str(item["title"]))
    return " | ".join(list(dict.fromkeys(titles))[:8])


def _num(row: Any, column: str) -> float:
    value = row.get(column, 0.0)
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _word_count(value: str) -> int:
    return len(str(value or "").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze post-v4 false negatives for the v5 entity overlay sprint.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    summary = analyze_false_negatives(args.scored_csv, csv_output=args.csv_output, top_n=args.top_n)
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
