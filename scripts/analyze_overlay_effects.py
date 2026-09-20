from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.feature_extractor import FeatureExtractor


def _score_bucket(value: float) -> str:
    if value < 0.25:
        return "0.00-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def _reason_column(frame: pd.DataFrame) -> str:
    for column in ("drift_overlay_v6_reason", "entity_overlay_v5_reason", "refute_overlay_reason", "refute_overlay_v4", "refute_overlay_v3"):
        if column in frame.columns:
            return column
    return ""


def analyze_overlay_effects(
    base_csv: str | Path,
    overlay_csv: str | Path,
    csv_output: str | Path,
    *,
    false_positive_output: str | Path | None = None,
    true_positive_output: str | Path | None = None,
) -> dict[str, Any]:
    base = pd.read_csv(base_csv)
    overlay = pd.read_csv(overlay_csv)
    if len(base) != len(overlay):
        raise ValueError(f"Base and overlay row counts differ: {len(base)} != {len(overlay)}")
    extractor = FeatureExtractor()
    reason_col = _reason_column(overlay)
    base_score = base["is_hallucination_proba"].astype(float)
    overlay_score = overlay["is_hallucination_proba"].astype(float)
    changed = overlay.copy()
    changed["base_is_hallucination_proba"] = base_score
    changed["overlay_is_hallucination_proba"] = overlay_score
    changed["overlay_delta"] = overlay_score - base_score
    changed["_profile"] = changed["prompt"].map(lambda value: extractor._question_profile(str(value)))
    changed["_base_score_bucket"] = changed["base_is_hallucination_proba"].map(_score_bucket)
    if reason_col:
        changed["overlay_reason"] = changed[reason_col]
    else:
        changed["overlay_reason"] = "unknown"
    changed_rows = changed[changed["overlay_delta"] > 1e-12].copy()

    output_columns = [
        "prompt",
        "model_answer",
        "is_hallucination",
        "base_is_hallucination_proba",
        "overlay_is_hallucination_proba",
        "overlay_delta",
        "_profile",
        "_base_score_bucket",
        "overlay_reason",
        "evidence_retrieval_hit_count",
        "evidence_top_evidence_overlap",
        "evidence_core_supported",
        "evidence_core_refuted",
        "evidence_number_refuted_count",
        "evidence_year_refuted_count",
        "evidence_aligned_hit_count",
        "evidence_aligned_year_refuted_count",
        "evidence_aligned_number_refuted_count",
        "evidence_answer_year_in_prompt",
        "evidence_answer_number_in_prompt",
        "evidence_aligned_title_entity_match",
        "evidence_compact_json",
    ]
    existing = [column for column in output_columns if column in changed_rows.columns]
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    changed_rows[existing].to_csv(csv_path, index=False)

    false_positive_output = false_positive_output or csv_path.with_name(csv_path.stem + "_false_positives.csv")
    true_positive_output = true_positive_output or csv_path.with_name(csv_path.stem + "_true_positives.csv")
    false_positive_rows = changed_rows[changed_rows["is_hallucination"] == 0].sort_values("overlay_delta", ascending=False)
    true_positive_rows = changed_rows[changed_rows["is_hallucination"] == 1].sort_values("overlay_delta", ascending=False)
    false_positive_rows[existing].head(100).to_csv(false_positive_output, index=False)
    true_positive_rows[existing].head(100).to_csv(true_positive_output, index=False)

    report: dict[str, Any] = {
        "rows": int(len(changed)),
        "changed_rows": int(len(changed_rows)),
        "changed_positive_rows": int((changed_rows["is_hallucination"] == 1).sum()) if "is_hallucination" in changed_rows else 0,
        "changed_negative_rows": int((changed_rows["is_hallucination"] == 0).sum()) if "is_hallucination" in changed_rows else 0,
        "changed_precision": None,
        "false_positive_changed_rows": int((changed_rows["is_hallucination"] == 0).sum()) if "is_hallucination" in changed_rows else 0,
        "true_positive_changed_rows": int((changed_rows["is_hallucination"] == 1).sum()) if "is_hallucination" in changed_rows else 0,
        "changed_by_profile": changed_rows["_profile"].value_counts().astype(int).to_dict(),
        "changed_by_overlay_reason": changed_rows["overlay_reason"].value_counts().astype(int).to_dict(),
        "base_score_bucket_summary": changed_rows["_base_score_bucket"].value_counts().sort_index().astype(int).to_dict(),
        "csv_output": str(csv_path),
        "top_changed_false_positives_output": str(false_positive_output),
        "top_changed_true_positives_output": str(true_positive_output),
    }
    if len(changed_rows) and "is_hallucination" in changed_rows:
        report["changed_precision"] = float((changed_rows["is_hallucination"] == 1).mean())
    if "is_hallucination" in changed.columns:
        report["pr_auc_base"] = float(average_precision_score(changed["is_hallucination"], base_score))
        report["pr_auc_overlay"] = float(average_precision_score(changed["is_hallucination"], overlay_score))
        report["delta_pr_auc"] = float(report["pr_auc_overlay"] - report["pr_auc_base"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rows affected by a refute overlay candidate.")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--overlay-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--false-positive-output", default=None)
    parser.add_argument("--true-positive-output", default=None)
    args = parser.parse_args()

    report = analyze_overlay_effects(
        args.base_csv,
        args.overlay_csv,
        args.csv_output,
        false_positive_output=args.false_positive_output,
        true_positive_output=args.true_positive_output,
    )
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
