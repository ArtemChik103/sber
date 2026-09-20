from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.drift_overlay import apply_drift_overlay_frame, taxonomy_label_v6
from guardian_of_truth.feature_extractor import FeatureExtractor


def analyze_drift_candidates(scored_csv: str | Path, *, csv_output: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(scored_csv)
    output = apply_drift_overlay_frame(frame, cap=0.60)
    output["drift_taxonomy_v6"] = output.apply(taxonomy_label_v6, axis=1)
    false_negatives = output[(output["is_hallucination"] == 1) & (output["base_is_hallucination_proba"] < 0.5)].copy()
    true_negatives = output[(output["is_hallucination"] == 0) & (output["base_is_hallucination_proba"] < 0.5)].copy()
    candidates = output[output["drift_overlay_v6_delta"] > 1e-12].copy()
    generic_long_fn = false_negatives[(false_negatives["_profile"] == "generic") & (false_negatives["answer_word_count"] >= 35)]
    generic_long_tn = true_negatives[(true_negatives["_profile"] == "generic") & (true_negatives["answer_word_count"] >= 35)]

    columns = [
        "prompt",
        "model_answer",
        "is_hallucination",
        "base_is_hallucination_proba",
        "is_hallucination_proba",
        "_profile",
        "score_path",
        "audit_status",
        "evidence_status",
        "evidence_retrieval_hit_count",
        "evidence_top_evidence_overlap",
        "evidence_core_supported",
        "evidence_core_refuted",
        "evidence_claim_refuted_count",
        "evidence_claim_unknown_count",
        "evidence_tail_unsupported_entity_count",
        "evidence_tail_unsupported_number_count",
        "evidence_answer_entity_not_in_evidence_ratio",
        "evidence_answer_number_not_in_evidence_ratio",
        "answer_word_count",
        "answer_sentence_count",
        "answer_tail_word_count",
        "answer_tail_ratio",
        "drift_low_overlap",
        "drift_weak_audit_path",
        "drift_long_answer_shape",
        "drift_tail_unsupported",
        "drift_evidence_gap",
        "drift_taxonomy_v6",
        "drift_overlay_v6_reason",
        "drift_overlay_v6_delta",
        "evidence_compact_json",
    ]
    existing = [column for column in columns if column in candidates.columns]
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.sort_values(["drift_overlay_v6_reason", "base_is_hallucination_proba"]).to_csv(csv_path, columns=existing, index=False)

    report: dict[str, Any] = {
        "rows": int(len(output)),
        "false_negatives": int(len(false_negatives)),
        "false_negatives_by_profile": false_negatives["_profile"].value_counts().astype(int).to_dict(),
        "false_negatives_by_score_path": false_negatives.get("score_path", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "false_negatives_by_audit_status": false_negatives.get("audit_status", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "long_answer_false_negatives": int((false_negatives["answer_word_count"] >= 35).sum()),
        "long_answer_true_negatives": int((true_negatives["answer_word_count"] >= 35).sum()),
        "generic_long_answer_false_negatives": int(len(generic_long_fn)),
        "generic_long_answer_true_negatives": int(len(generic_long_tn)),
        "taxonomy_counts_all": output["drift_taxonomy_v6"].value_counts().astype(int).to_dict(),
        "taxonomy_counts_false_negatives": false_negatives["drift_taxonomy_v6"].value_counts().astype(int).to_dict(),
        "candidate_trigger_counts": candidates["drift_overlay_v6_reason"].value_counts().astype(int).to_dict(),
        "candidate_rows": int(len(candidates)),
        "candidate_precision": float((candidates["is_hallucination"] == 1).mean()) if len(candidates) else None,
        "candidate_false_positives": int((candidates["is_hallucination"] == 0).sum()),
        "candidate_true_positives": int((candidates["is_hallucination"] == 1).sum()),
        "candidate_rows_by_reason": candidates["drift_overlay_v6_reason"].value_counts().astype(int).to_dict(),
        "top_changed_false_positive_risks": _top_risks(candidates),
        "csv_output": str(csv_path),
    }
    return report


def _top_risks(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    risks = candidates[candidates["is_hallucination"] == 0].sort_values("drift_overlay_v6_delta", ascending=False).head(20)
    return [
        {
            "prompt": str(row.get("prompt", ""))[:240],
            "model_answer": str(row.get("model_answer", ""))[:240],
            "base_score": float(row.get("base_is_hallucination_proba", 0.0)),
            "reason": str(row.get("drift_overlay_v6_reason", "")),
            "profile": str(row.get("_profile", "")),
        }
        for _, row in risks.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v6 unsupported-tail / answer-drift overlay candidates.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    summary = analyze_drift_candidates(args.scored_csv, csv_output=args.csv_output)
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
