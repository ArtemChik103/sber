from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.evidence import is_generic_subject_token, subject_terms_from_prompt
from guardian_of_truth.feature_extractor import FeatureExtractor


def _float(row: pd.Series, column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(row: pd.Series, column: str, default: str = "") -> str:
    value = row.get(column, default)
    return default if pd.isna(value) else str(value)


def _compact(row: pd.Series) -> dict[str, Any]:
    raw = _text(row, "evidence_compact_json", "{}")
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def _titles(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("title", "")) for item in items if str(item.get("title", "")).strip()]


def _title_subject_overlap(prompt: str, title: str) -> int:
    subject = subject_terms_from_prompt(prompt)
    title_tokens = {token for token in subject_terms_from_prompt(title) if not is_generic_subject_token(token)}
    return len(subject & title_tokens)


def _bad_alignment_reason(row: pd.Series) -> str:
    if _text(row, "evidence_status") != "ok":
        return ""
    compact = _compact(row)
    snippets = compact.get("snippets") or []
    aligned = compact.get("aligned_snippets") or []
    top_overlap = _float(row, "evidence_top_evidence_overlap")
    aligned_count = _float(row, "evidence_aligned_hit_count")
    title_match = _float(row, "evidence_aligned_title_entity_match")
    prompt = _text(row, "prompt")

    reasons: list[str] = []
    if top_overlap < 0.08:
        reasons.append("low_top_overlap")
    if aligned_count > 0 and title_match == 0 and top_overlap < 0.12:
        reasons.append("weak_aligned_no_title_entity")
    if snippets:
        top_title = str(snippets[0].get("title", ""))
        if _title_subject_overlap(prompt, top_title) == 0:
            reasons.append("zero_top_title_subject_overlap")
    if aligned:
        weak_aligned = True
        for item in aligned:
            reason = str(item.get("alignment_reason", ""))
            if reason in {"title_subject_overlap", "quoted_phrase_hit", "answer_core_entity_plus_subject", "text_subject_overlap_3"}:
                weak_aligned = False
                break
        if weak_aligned:
            reasons.append("aligned_by_weak_text_only")
    return ";".join(dict.fromkeys(reasons))


def analyze(scored_csv: str | Path, *, top_n: int = 200) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(scored_csv)
    extractor = FeatureExtractor()
    frame["_profile"] = frame["prompt"].map(lambda value: extractor._question_profile(str(value)))
    frame["_bad_alignment_reason"] = frame.apply(_bad_alignment_reason, axis=1)
    frame["_bad_alignment"] = frame["_bad_alignment_reason"].astype(bool)
    if "is_hallucination" in frame.columns:
        frame["_label"] = frame["is_hallucination"].astype(int)
        frame["_false_negative"] = (frame["_label"] == 1) & (frame["is_hallucination_proba"].astype(float) < 0.5)
        frame["_false_positive"] = (frame["_label"] == 0) & (frame["is_hallucination_proba"].astype(float) >= 0.5)
    else:
        frame["_false_negative"] = False
        frame["_false_positive"] = False

    overlaps = [_float(row, "evidence_top_evidence_overlap") for _, row in frame.iterrows()]
    compact_values = [_compact(row) for _, row in frame.iterrows()]
    top_title_counts: Counter[str] = Counter()
    weak_patterns: Counter[str] = Counter()
    for row, compact in zip((row for _, row in frame.iterrows()), compact_values):
        if not row["_bad_alignment"]:
            continue
        titles = _titles(compact.get("snippets") or [])
        if titles:
            top_title_counts[titles[0]] += 1
        subjects = sorted(subject_terms_from_prompt(_text(row, "prompt")))[:6]
        weak_patterns[" ".join(subjects) or "<empty>"] += 1

    bad = frame[frame["_bad_alignment"]].copy()
    summary = {
        "rows": int(len(frame)),
        "rows_with_evidence": int((_text(row, "evidence_status") == "ok" for _, row in frame.iterrows()).__sizeof__()) if False else int((frame.get("evidence_status", pd.Series(dtype=str)) == "ok").sum()),
        "rows_with_aligned_evidence": int((frame.get("evidence_aligned_hit_count", pd.Series(dtype=float)).fillna(0).astype(float) > 0).sum()),
        "average_top_evidence_overlap": float(statistics.fmean(overlaps)) if overlaps else 0.0,
        "p50_top_evidence_overlap": float(pd.Series(overlaps).quantile(0.50)) if overlaps else 0.0,
        "p90_top_evidence_overlap": float(pd.Series(overlaps).quantile(0.90)) if overlaps else 0.0,
        "bad_alignment_count": int(frame["_bad_alignment"].sum()),
        "bad_alignment_by_profile": bad["_profile"].value_counts().astype(int).to_dict(),
        "bad_alignment_by_score_path": bad.get("score_path", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "bad_alignment_by_audit_status": bad.get("audit_status", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "false_negatives_with_bad_alignment": int((frame["_bad_alignment"] & frame["_false_negative"]).sum()),
        "false_positives_with_bad_alignment": int((frame["_bad_alignment"] & frame["_false_positive"]).sum()),
        "top_unrelated_evidence_titles": dict(top_title_counts.most_common(30)),
        "top_weak_query_patterns": dict(weak_patterns.most_common(30)),
    }

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
        "evidence_aligned_hit_count",
        "evidence_aligned_title_entity_match",
        "_bad_alignment_reason",
        "evidence_compact_json",
    ]
    existing = [column for column in output_columns if column in bad.columns]
    return summary, bad[existing].head(top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v7 retrieval evidence alignment quality.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    summary, bad_rows = analyze(args.scored_csv, top_n=args.top_n)
    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    bad_rows.to_csv(csv_path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
