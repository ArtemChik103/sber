from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.evaluate import _question_profile
from guardian_of_truth.utils import sha256_hexdigest


SCORE = "is_hallucination_proba"
LABEL = "is_hallucination"


def _key_frame(path: str | Path, score_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["_row_key"] = frame.apply(lambda row: sha256_hexdigest(row.get("prompt"), row.get("model_answer")), axis=1)
    frame = frame.drop_duplicates("_row_key", keep="last")
    keep = ["_row_key", SCORE]
    return frame[keep].rename(columns={SCORE: score_name})


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rendered = frame.fillna("").copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(lambda value: str(value).replace("\n", " ").replace("|", "\\|")[:260])
    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rendered.to_numpy()]
    return "\n".join([header, separator, *body])


def diagnose_fallback_rows(
    current_csv: str | Path = "outputs/public_scored_promoted_min_ensemble_v1_from_audit_v5_clean.csv",
    fallback_candidate_csv: str | Path = "outputs/public_scored_candidate_fallback_v2_from_audit_v5_clean.csv",
    *,
    output_md: str | Path = "outputs/fallback_diagnostics_v5_clean.md",
    output_csv: str | Path = "outputs/fallback_diagnostics_v5_clean.csv",
    min_abs_delta: float = 0.10,
) -> dict[str, Any]:
    current = pd.read_csv(current_csv)
    current["_row_key"] = current.apply(lambda row: sha256_hexdigest(row.get("prompt"), row.get("model_answer")), axis=1)
    current = current.drop_duplicates("_row_key", keep="last")
    candidate_scores = _key_frame(fallback_candidate_csv, "fallback_v2_score")
    merged = current.merge(candidate_scores, on="_row_key", how="left")
    merged["current_score"] = pd.to_numeric(merged[SCORE], errors="coerce")
    merged["fallback_v2_score"] = pd.to_numeric(merged["fallback_v2_score"], errors="coerce")
    merged["fallback_delta"] = merged["fallback_v2_score"] - merged["current_score"]
    merged["profile"] = merged["prompt"].map(lambda value: _question_profile(str(value)))
    fallback_rows = merged[merged.get("score_path", "main").fillna("main").astype(str) == "fallback"].copy()
    changed = fallback_rows[fallback_rows["fallback_delta"].abs() >= min_abs_delta].copy()
    summary_by = []
    for cols in ([LABEL], ["profile"], ["audit_status"], [LABEL, "profile"], [LABEL, "audit_status"]):
        available = [col for col in cols if col in fallback_rows.columns]
        if not available:
            continue
        grouped = fallback_rows.groupby(available, dropna=False)["fallback_delta"].agg(["count", "mean", "median", "min", "max"]).reset_index()
        grouped["grouping"] = "+".join(available)
        summary_by.extend(grouped.round(6).to_dict(orient="records"))
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fallback_rows.to_csv(output, index=False)
    summary = {
        "current_csv": str(current_csv),
        "fallback_candidate_csv": str(fallback_candidate_csv),
        "fallback_rows": int(len(fallback_rows)),
        "changed_ge_threshold": int(len(changed)),
        "min_abs_delta": float(min_abs_delta),
        "summary_by_label_profile_status": summary_by,
    }
    lines = [
        "# Fallback Diagnostics V5 Clean",
        "",
        f"- current: `{current_csv}`",
        f"- fallback candidate: `{fallback_candidate_csv}`",
        f"- fallback rows: `{len(fallback_rows)}`",
        f"- rows with abs delta >= `{min_abs_delta}`: `{len(changed)}`",
        "",
        "## Summary",
        "",
        _markdown_table(pd.DataFrame(summary_by)),
        "",
        "## Changed Rows",
        "",
        _markdown_table(
            changed.sort_values("fallback_delta", key=lambda s: s.abs(), ascending=False)[
                ["prompt", "model_answer", LABEL, "profile", "audit_status", "current_score", "fallback_v2_score", "fallback_delta"]
            ].head(60)
        ),
        "",
    ]
    Path(output_md).write_text("\n".join(lines), encoding="utf-8")
    Path(output_md).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current fallback rows against fallback_v2.")
    parser.add_argument("--current-csv", default="outputs/public_scored_promoted_min_ensemble_v1_from_audit_v5_clean.csv")
    parser.add_argument("--fallback-candidate-csv", default="outputs/public_scored_candidate_fallback_v2_from_audit_v5_clean.csv")
    parser.add_argument("--output-md", default="outputs/fallback_diagnostics_v5_clean.md")
    parser.add_argument("--output-csv", default="outputs/fallback_diagnostics_v5_clean.csv")
    parser.add_argument("--min-abs-delta", type=float, default=0.10)
    args = parser.parse_args()
    print(
        json.dumps(
            diagnose_fallback_rows(
                args.current_csv,
                args.fallback_candidate_csv,
                output_md=args.output_md,
                output_csv=args.output_csv,
                min_abs_delta=args.min_abs_delta,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
