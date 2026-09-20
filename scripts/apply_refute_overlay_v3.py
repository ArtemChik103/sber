from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.feature_extractor import FeatureExtractor


PROMOTED_BASELINE = 0.579354


def _num(row: pd.Series, column: str) -> float:
    value = row.get(column, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def overlay_score(row: pd.Series, extractor: FeatureExtractor, *, numeric_year_only: bool = False) -> tuple[float, str]:
    base = _num(row, "is_hallucination_proba")
    profile = extractor._question_profile(str(row.get("prompt", "")))
    if _num(row, "evidence_retrieval_hit_count") < 2:
        return base, "unchanged_low_hits"
    if _num(row, "evidence_top_evidence_overlap") < 0.12:
        return base, "unchanged_low_overlap"

    core_refuted = _num(row, "evidence_core_refuted") > 0
    typed_numeric_refuted = (profile == "when" and _num(row, "evidence_year_refuted_count") > 0) or (
        profile == "count" and _num(row, "evidence_number_refuted_count") > 0
    )
    typed_entity_refuted = profile in {"who", "where", "by_whom", "title_name"} and _num(row, "evidence_entity_refuted_count") > 0

    if numeric_year_only:
        if typed_numeric_refuted:
            return max(base, 0.82), "numeric_or_year_refuted"
        return base, "unchanged_numeric_year_only"

    if core_refuted or typed_numeric_refuted or typed_entity_refuted:
        return max(base, 0.82), "core_or_typed_refuted"
    if _num(row, "evidence_claim_refuted_count") > 0:
        return max(base, 0.65), "tail_refuted"
    return base, "unchanged_weak_evidence"


def apply_overlay(scored_csv: str | Path, output_csv: str | Path, *, numeric_year_only: bool = False) -> dict[str, Any]:
    frame = pd.read_csv(scored_csv)
    extractor = FeatureExtractor()
    original = frame["is_hallucination_proba"].astype(float).copy()
    decisions = frame.apply(lambda row: overlay_score(row, extractor, numeric_year_only=numeric_year_only), axis=1)
    frame["base_is_hallucination_proba"] = original
    frame["is_hallucination_proba"] = [score for score, _ in decisions]
    frame["refute_overlay_v3"] = [reason for _, reason in decisions]
    frame["refute_overlay_delta"] = frame["is_hallucination_proba"] - frame["base_is_hallucination_proba"]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "output": str(output_path),
        "changed_rows": int((frame["refute_overlay_delta"] > 0).sum()),
        "decision_counts": frame["refute_overlay_v3"].value_counts().astype(int).to_dict(),
        "never_decreased": bool((frame["refute_overlay_delta"] >= -1e-12).all()),
        "numeric_year_only": bool(numeric_year_only),
    }
    if "is_hallucination" in frame.columns:
        report["base_pr_auc"] = float(average_precision_score(frame["is_hallucination"], frame["base_is_hallucination_proba"]))
        report["overlay_pr_auc"] = float(average_precision_score(frame["is_hallucination"], frame["is_hallucination_proba"]))
        report["delta_vs_base"] = float(report["overlay_pr_auc"] - report["base_pr_auc"])
        report["delta_vs_promoted_baseline"] = float(report["overlay_pr_auc"] - PROMOTED_BASELINE)
        false_positive_overlay = frame[
            (frame["is_hallucination"] == 0)
            & (frame["refute_overlay_delta"] > 0)
            & (frame["is_hallucination_proba"] >= 0.5)
        ].sort_values("refute_overlay_delta", ascending=False)
        fp_path = output_path.with_name(output_path.stem + "_overlay_false_positives.csv")
        false_positive_overlay.to_csv(fp_path, index=False)
        report["overlay_false_positives_output"] = str(fp_path)
        report["overlay_false_positives"] = int(len(false_positive_overlay))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply conservative high-confidence refute overlay v3.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--numeric-year-only", action="store_true")
    args = parser.parse_args()

    report = apply_overlay(args.scored_csv, args.output_csv, numeric_year_only=args.numeric_year_only)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
