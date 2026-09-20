from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.evaluate import _question_profile
from guardian_of_truth.utils import sha256_hexdigest
from scripts.analysis_taxonomy import TAXONOMY_LABELS, answer_length_bucket, taxonomy_labels


SCORE = "is_hallucination_proba"
LABEL = "is_hallucination"
AUDIT_FIELDS = ("h", "n", "e", "u", "x", "we", "wn", "ue", "bt", "conf")


def row_key(prompt: Any, answer: Any) -> str:
    return sha256_hexdigest(prompt, answer)


def _read_scored(path: str | Path, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["_row_key"] = frame.apply(lambda row: row_key(row.get("prompt"), row.get("model_answer")), axis=1)
    frame["_duplicate_key_count"] = frame.groupby("_row_key")["_row_key"].transform("size")
    frame = frame.drop_duplicates("_row_key", keep="last").copy()
    frame["_source_name"] = source_name
    return frame


def _ap(labels: pd.Series, scores: pd.Series) -> float:
    labels = pd.to_numeric(labels, errors="coerce")
    scores = pd.to_numeric(scores, errors="coerce")
    valid = labels.notna() & scores.notna()
    if labels[valid].nunique() < 2:
        return 0.0
    return float(average_precision_score(labels[valid].astype(int), scores[valid].astype(float)))


def _ap_by(frame: pd.DataFrame, by: str, score_col: str) -> list[dict[str, Any]]:
    if by not in frame.columns:
        return []
    rows = []
    for value, chunk in frame.groupby(by, dropna=False, sort=True):
        rows.append(
            {
                by: str(value),
                "rows": int(len(chunk)),
                "ap": round(_ap(chunk[LABEL], chunk[score_col]), 6),
                "positive_rate": round(float(pd.to_numeric(chunk[LABEL], errors="coerce").mean()), 6),
                "score_mean": round(float(pd.to_numeric(chunk[score_col], errors="coerce").mean()), 6),
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, Any]] | pd.DataFrame) -> str:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "_No rows._"
    rendered = frame.fillna("").copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].map(lambda value: str(value).replace("\n", " ").replace("|", "\\|")[:260])
    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rendered.to_numpy()]
    return "\n".join([header, separator, *body])


def _score_delta_distribution(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.groupby(LABEL)["candidate_delta"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .round(6)
        .to_dict(orient="records")
    )


def _audit_field_deltas(frame: pd.DataFrame, source_col: str) -> list[dict[str, Any]]:
    rows = []
    for field in AUDIT_FIELDS:
        left = f"audit_{field}_baseline"
        right = f"audit_{field}_{source_col}"
        if left not in frame.columns or right not in frame.columns:
            continue
        delta = pd.to_numeric(frame[right], errors="coerce") - pd.to_numeric(frame[left], errors="coerce")
        rows.append(
            {
                "field": field,
                "count": int(delta.notna().sum()),
                "mean_delta": round(float(delta.mean()), 6) if delta.notna().any() else 0.0,
                "median_delta": round(float(delta.median()), 6) if delta.notna().any() else 0.0,
                "min_delta": round(float(delta.min()), 6) if delta.notna().any() else 0.0,
                "max_delta": round(float(delta.max()), 6) if delta.notna().any() else 0.0,
            }
        )
    return rows


def _top(frame: pd.DataFrame, sort_col: str, columns: list[str], n: int = 20) -> pd.DataFrame:
    cols = [col for col in columns if col in frame.columns]
    if sort_col not in frame.columns or not cols:
        return pd.DataFrame()
    return frame.sort_values(sort_col, ascending=False)[cols].head(n)


def compare_audit_sources(
    *,
    historical_csv: str | Path,
    v5_clean_csv: str | Path,
    output_md: str | Path = "outputs/audit_drift_v5_clean.md",
    output_csv: str | Path = "outputs/audit_drift_v5_clean.csv",
    output_summary: str | Path = "outputs/audit_drift_v5_clean.summary.json",
    prior_csvs: list[str | Path] | None = None,
    candidate_csvs: list[str | Path] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    baseline = _read_scored(historical_csv, "historical").add_suffix("_baseline")
    baseline = baseline.rename(columns={"_row_key_baseline": "_row_key"})
    v5 = _read_scored(v5_clean_csv, "v5_clean").add_suffix("_v5")
    v5 = v5.rename(columns={"_row_key_v5": "_row_key"})
    merged = baseline.merge(v5, on="_row_key", how="outer", indicator="baseline_v5_presence")
    for col in ("prompt", "model_answer", LABEL):
        merged[col] = merged.get(f"{col}_v5").combine_first(merged.get(f"{col}_baseline"))
    merged["baseline_score"] = pd.to_numeric(merged.get(f"{SCORE}_baseline"), errors="coerce")
    merged["v5_score"] = pd.to_numeric(merged.get(f"{SCORE}_v5"), errors="coerce")
    merged["candidate_score"] = merged["v5_score"]
    merged["candidate_delta"] = merged["v5_score"] - merged["baseline_score"]
    merged["profile"] = merged["prompt"].map(lambda value: _question_profile(str(value)))
    merged["answer_length_bucket"] = merged["model_answer"].map(answer_length_bucket)
    merged["score_path"] = merged.get("score_path_v5", merged.get("score_path_baseline", "unknown"))
    merged["audit_status"] = merged.get("audit_status_v5", merged.get("audit_status_baseline", "unknown"))
    merged["taxonomy_labels"] = merged.apply(lambda row: ",".join(taxonomy_labels(row)), axis=1)

    candidate_summaries = []
    candidate_frames = []
    for candidate_path in candidate_csvs or []:
        candidate_name = Path(candidate_path).stem
        score_column = f"{candidate_name}_candidate_score"
        candidate = _read_scored(candidate_path, candidate_name)[["_row_key", SCORE]].rename(columns={SCORE: score_column})
        merged = merged.merge(candidate, on="_row_key", how="left")
        delta_col = f"{candidate_name}_minus_baseline"
        merged[delta_col] = pd.to_numeric(merged[score_column], errors="coerce") - merged["baseline_score"]
        candidate_summaries.append(
            {
                "candidate": candidate_name,
                "matched_rows": int(merged[score_column].notna().sum()),
                "ap": round(_ap(merged[LABEL], merged[score_column]), 6),
                "delta_ap_vs_baseline": round(_ap(merged[LABEL], merged[score_column]) - _ap(merged[LABEL], merged["baseline_score"]), 6),
            }
        )
        bad = merged[((merged[LABEL] == 0) & (merged[delta_col] > 0)) | ((merged[LABEL] == 1) & (merged[delta_col] < 0))].copy()
        bad["candidate_name"] = candidate_name
        bad["ranking_worsening"] = bad[delta_col].abs()
        candidate_frames.append(bad)

    prior_summaries = []
    for prior_path in prior_csvs or []:
        prior = _read_scored(prior_path, Path(prior_path).stem)
        prior_summaries.append(
            {
                "source": Path(prior_path).stem,
                "rows": int(len(prior)),
                "unique_keys": int(prior["_row_key"].nunique()),
                "ap": round(_ap(prior[LABEL], prior[SCORE]), 6) if LABEL in prior.columns and SCORE in prior.columns else 0.0,
            }
        )

    merged_out = Path(output_csv)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_out, index=False)

    summary = {
        "historical_csv": str(historical_csv),
        "v5_clean_csv": str(v5_clean_csv),
        "rows": int(len(merged)),
        "matched_rows": int((merged["baseline_v5_presence"] == "both").sum()),
        "baseline_ap": round(_ap(merged[LABEL], merged["baseline_score"]), 6),
        "v5_clean_ap": round(_ap(merged[LABEL], merged["v5_score"]), 6),
        "duplicate_keys": {
            "historical": int((baseline.get("_duplicate_key_count_baseline", pd.Series(dtype=int)) > 1).sum()),
            "v5_clean": int((v5.get("_duplicate_key_count_v5", pd.Series(dtype=int)) > 1).sum()),
        },
        "missing_rows": merged["baseline_v5_presence"].value_counts().astype(int).to_dict(),
        "score_delta_by_label": _score_delta_distribution(merged.dropna(subset=["candidate_delta"])),
        "audit_field_deltas": _audit_field_deltas(merged, "v5"),
        "taxonomy_counts": pd.Series(",".join(merged["taxonomy_labels"].fillna("none")).split(",")).value_counts().astype(int).to_dict(),
        "prior_sources": prior_summaries,
        "candidates": candidate_summaries,
    }
    Path(output_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    top_v5_positive_down = _top(
        merged[merged[LABEL] == 1].assign(v5_positive_down=-(merged["candidate_delta"])),
        "v5_positive_down",
        ["prompt", "model_answer", LABEL, "baseline_score", "v5_score", "candidate_delta", "audit_status", "score_path", "taxonomy_labels"],
        top_n,
    )
    top_v5_negative_up = _top(
        merged[merged[LABEL] == 0].assign(v5_negative_up=merged["candidate_delta"]),
        "v5_negative_up",
        ["prompt", "model_answer", LABEL, "baseline_score", "v5_score", "candidate_delta", "audit_status", "score_path", "taxonomy_labels"],
        top_n,
    )
    worsened = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    lines = [
        "# Audit Drift V5 Clean",
        "",
        f"- historical: `{historical_csv}`",
        f"- v5 clean: `{v5_clean_csv}`",
        f"- matched rows: `{summary['matched_rows']}`",
        f"- historical AP: `{summary['baseline_ap']:.6f}`",
        f"- v5 clean AP: `{summary['v5_clean_ap']:.6f}`",
        "",
        "## AP By Audit Status",
        "",
        _markdown_table(_ap_by(merged, "audit_status", "v5_score")),
        "",
        "## AP By Score Path",
        "",
        _markdown_table(_ap_by(merged, "score_path", "v5_score")),
        "",
        "## AP By Profile",
        "",
        _markdown_table(_ap_by(merged, "profile", "v5_score")),
        "",
        "## AP By Answer Length Bucket",
        "",
        _markdown_table(_ap_by(merged, "answer_length_bucket", "v5_score")),
        "",
        "## Score Delta By Label",
        "",
        _markdown_table(summary["score_delta_by_label"]),
        "",
        "## Audit Field Deltas",
        "",
        _markdown_table(summary["audit_field_deltas"]),
        "",
        "## Taxonomy Counts",
        "",
        _markdown_table(pd.Series(summary["taxonomy_counts"]).rename_axis("taxonomy").reset_index(name="rows")),
        "",
        "## V5 Clean Moved Positives Down",
        "",
        _markdown_table(top_v5_positive_down),
        "",
        "## V5 Clean Moved Negatives Up",
        "",
        _markdown_table(top_v5_negative_up),
        "",
        "## Candidate Worsened Baseline Ranking",
        "",
        _markdown_table(_top(worsened, "ranking_worsening", ["candidate_name", "prompt", "model_answer", LABEL, "baseline_score", "ranking_worsening", "taxonomy_labels"], top_n)),
        "",
    ]
    Path(output_md).write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare historical, prior audit-source, v5-clean, and rejected candidate score drift.")
    parser.add_argument("--historical-csv", default="outputs/public_scored_promoted_min_ensemble_v1.csv")
    parser.add_argument("--v5-clean-csv", default="outputs/public_scored_promoted_min_ensemble_v1_from_audit_v5_clean.csv")
    parser.add_argument("--prior-csv", action="append", default=["outputs/public_scored_promoted_from_audit_v4.csv"])
    parser.add_argument("--candidate-csv", action="append", default=["outputs/public_scored_candidate_constrained_gate_v2_from_audit_v5_clean.csv", "outputs/public_scored_candidate_fallback_v2_from_audit_v5_clean.csv"])
    parser.add_argument("--output-md", default="outputs/audit_drift_v5_clean.md")
    parser.add_argument("--output-csv", default="outputs/audit_drift_v5_clean.csv")
    parser.add_argument("--output-summary", default="outputs/audit_drift_v5_clean.summary.json")
    args = parser.parse_args()
    summary = compare_audit_sources(
        historical_csv=args.historical_csv,
        v5_clean_csv=args.v5_clean_csv,
        prior_csvs=[path for path in args.prior_csv if Path(path).exists()],
        candidate_csvs=[path for path in args.candidate_csv if Path(path).exists()],
        output_md=args.output_md,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
