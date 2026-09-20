from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.feature_extractor import FeatureExtractor


def analyze(scored_csv: str | Path, *, top_n: int = 100) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(scored_csv)
    extractor = FeatureExtractor()
    frame["_profile"] = frame["prompt"].map(lambda value: extractor._question_profile(str(value)))
    frame["_error_kind"] = "ok"
    frame.loc[(frame["is_hallucination"] == 0) & (frame["is_hallucination_proba"] >= 0.5), "_error_kind"] = "false_positive"
    frame.loc[(frame["is_hallucination"] == 1) & (frame["is_hallucination_proba"] < 0.5), "_error_kind"] = "false_negative"
    frame["_ranking_error"] = frame.apply(
        lambda row: float(row["is_hallucination_proba"]) if int(row["is_hallucination"]) == 0 else 1.0 - float(row["is_hallucination_proba"]),
        axis=1,
    )
    errors = frame[frame["_error_kind"] != "ok"].sort_values("_ranking_error", ascending=False).head(top_n)
    summary = {
        "rows": int(len(frame)),
        "pr_auc": float(average_precision_score(frame["is_hallucination"], frame["is_hallucination_proba"])),
        "top_n": int(top_n),
        "error_counts": frame["_error_kind"].value_counts().astype(int).to_dict(),
        "top_error_profiles": errors["_profile"].value_counts().astype(int).to_dict(),
        "top_error_evidence_status": errors.get("evidence_status", pd.Series(dtype=str)).value_counts(dropna=False).astype(int).to_dict(),
        "top_error_core_refuted": errors.get("evidence_core_refuted", pd.Series(dtype=float)).value_counts(dropna=False).astype(int).to_dict(),
        "top_error_hit_count": errors.get("evidence_retrieval_hit_count", pd.Series(dtype=float)).value_counts(dropna=False).head(20).astype(int).to_dict(),
    }
    columns = [
        "prompt",
        "model_answer",
        "is_hallucination",
        "is_hallucination_proba",
        "_error_kind",
        "_profile",
        "score_path",
        "audit_status",
        "evidence_status",
        "evidence_retrieval_hit_count",
        "evidence_top_evidence_overlap",
        "evidence_core_supported",
        "evidence_core_refuted",
        "evidence_claim_refuted_count",
        "evidence_compact_json",
    ]
    existing_columns = [column for column in columns if column in errors.columns]
    return summary, errors[existing_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top public ranking errors with evidence diagnostics.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", default="outputs/evidence_error_summary.json")
    parser.add_argument("--csv-output", default="outputs/evidence_top_errors.csv")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    summary, errors = analyze(args.scored_csv, top_n=args.top_n)
    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    errors.to_csv(csv_path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
