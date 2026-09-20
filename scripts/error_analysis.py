from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.evaluate import _question_profile
from guardian_of_truth.utils import sha256_hexdigest


SCORE_COLUMN = "is_hallucination_proba"
LABEL_COLUMN = "is_hallucination"


def _row_key(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: sha256_hexdigest(row.get("prompt"), row.get("model_answer"), row.get(LABEL_COLUMN)),
        axis=1,
    )


def _average_precision(labels: pd.Series, scores: pd.Series) -> float:
    if labels.nunique() < 2:
        return 0.0
    return float(average_precision_score(labels.astype(int), scores.astype(float)))


def _score_distribution(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    return (
        frame.groupby(LABEL_COLUMN)[score_column]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .round(6)
    )


def _ap_by_column(frame: pd.DataFrame, column: str, score_column: str) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame()
    rows = []
    for value, chunk in frame.groupby(column, dropna=False, sort=True):
        rows.append(
            {
                column: value,
                "rows": len(chunk),
                "ap": _average_precision(chunk[LABEL_COLUMN], chunk[score_column]),
                "positive_rate": float(chunk[LABEL_COLUMN].mean()),
                "score_mean": float(chunk[score_column].mean()),
            }
        )
    return pd.DataFrame(rows).round(6)


def _answer_length_summary(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["answer_len_chars"] = enriched["model_answer"].fillna("").astype(str).str.len()
    enriched["answer_len_words"] = enriched["model_answer"].fillna("").astype(str).str.split().map(len)
    rows = []
    for label, chunk in enriched.groupby(LABEL_COLUMN, sort=True):
        rows.append(
            {
                LABEL_COLUMN: label,
                "rows": len(chunk),
                "chars_p50": float(chunk["answer_len_chars"].quantile(0.50)),
                "chars_p90": float(chunk["answer_len_chars"].quantile(0.90)),
                "words_p50": float(chunk["answer_len_words"].quantile(0.50)),
                "words_p90": float(chunk["answer_len_words"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).round(3)


def _score_delta_summary(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["score_delta"] = enriched["candidate_score"] - enriched["baseline_score"]
    return (
        enriched.groupby(LABEL_COLUMN)["score_delta"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .round(6)
    )


def _top_errors(frame: pd.DataFrame, score_column: str, label: int, n: int) -> pd.DataFrame:
    subset = frame[frame[LABEL_COLUMN] == label].copy()
    if label == 0:
        subset = subset.sort_values(score_column, ascending=False)
    else:
        subset = subset.sort_values(score_column, ascending=True)
    columns = ["prompt", "model_answer", LABEL_COLUMN, "baseline_score", "candidate_score"]
    return subset[columns].head(n)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rendered = frame.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(lambda value: str(value).replace("\n", " ").replace("|", "\\|"))
    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rendered.to_numpy()]
    return "\n".join([header, separator, *body])


def _ranking_regressions(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    positives = frame[frame[LABEL_COLUMN] == 1][["row_key", "baseline_score", "candidate_score"]]
    negatives = frame[frame[LABEL_COLUMN] == 0][["row_key", "baseline_score", "candidate_score"]]
    merged = positives.merge(negatives, how="cross", suffixes=("_pos", "_neg"))
    broken = merged[
        (merged["baseline_score_pos"] > merged["baseline_score_neg"])
        & (merged["candidate_score_pos"] <= merged["candidate_score_neg"])
    ].copy()
    if broken.empty:
        return broken
    broken["regression_margin"] = (
        (broken["baseline_score_pos"] - broken["baseline_score_neg"])
        + (broken["candidate_score_neg"] - broken["candidate_score_pos"])
    )
    return broken.sort_values("regression_margin", ascending=False).head(n)


def build_report(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    baseline = pd.read_csv(baseline_path)
    candidate = pd.read_csv(candidate_path)
    baseline["row_key"] = _row_key(baseline)
    candidate["row_key"] = _row_key(candidate)

    merged = baseline[["row_key", SCORE_COLUMN]].rename(columns={SCORE_COLUMN: "baseline_score"}).merge(
        candidate.rename(columns={SCORE_COLUMN: "candidate_score"}),
        on="row_key",
        how="inner",
    )
    merged["profile"] = merged["prompt"].map(lambda prompt: _question_profile(str(prompt)))

    profile_rows = []
    for profile, chunk in merged.groupby("profile", sort=True):
        profile_rows.append(
            {
                "profile": profile,
                "rows": len(chunk),
                "baseline_ap": _average_precision(chunk[LABEL_COLUMN], chunk["baseline_score"]),
                "candidate_ap": _average_precision(chunk[LABEL_COLUMN], chunk["candidate_score"]),
                "delta_ap": _average_precision(chunk[LABEL_COLUMN], chunk["candidate_score"])
                - _average_precision(chunk[LABEL_COLUMN], chunk["baseline_score"]),
            }
        )
    profile_ap = pd.DataFrame(profile_rows).round(6)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranking_regressions = _ranking_regressions(merged, top_n)

    lines = [
        "# Candidate Error Analysis",
        "",
        f"- baseline: `{baseline_path}`",
        f"- candidate: `{candidate_path}`",
        f"- matched rows: `{len(merged)}`",
        f"- baseline AP: `{_average_precision(merged[LABEL_COLUMN], merged['baseline_score']):.6f}`",
        f"- candidate AP: `{_average_precision(merged[LABEL_COLUMN], merged['candidate_score']):.6f}`",
        "",
        "## AP By Question Profile",
        "",
        _markdown_table(profile_ap),
        "",
        "## Baseline Score Distribution By Label",
        "",
        _markdown_table(_score_distribution(merged, "baseline_score")),
        "",
        "## Candidate Score Distribution By Label",
        "",
        _markdown_table(_score_distribution(merged, "candidate_score")),
        "",
        "## Candidate AP By Audit Status",
        "",
        _markdown_table(_ap_by_column(merged, "audit_status", "candidate_score")),
        "",
        "## Candidate AP By Score Path",
        "",
        _markdown_table(_ap_by_column(merged, "score_path", "candidate_score")),
        "",
        "## Answer Length By Label",
        "",
        _markdown_table(_answer_length_summary(merged)),
        "",
        "## Candidate Minus Baseline Score Delta",
        "",
        _markdown_table(_score_delta_summary(merged)),
        "",
        "## Top Candidate False Positives",
        "",
        _markdown_table(_top_errors(merged, "candidate_score", label=0, n=top_n)),
        "",
        "## Top Candidate False Negatives",
        "",
        _markdown_table(_top_errors(merged, "candidate_score", label=1, n=top_n)),
        "",
        "## Ranking Regressions",
        "",
        _markdown_table(ranking_regressions) if not ranking_regressions.empty else "_No broken baseline-correct pairs found._",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare public benchmark scores for baseline and candidate outputs.")
    parser.add_argument("--baseline", default="outputs/public_scored_quality_upgrade_full.csv")
    parser.add_argument("--candidate", default="outputs/public_scored_candidate_v4_full.csv")
    parser.add_argument("--output", default="outputs/error_analysis_candidate_v4.md")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    build_report(args.baseline, args.candidate, output_path=args.output, top_n=args.top_n)


if __name__ == "__main__":
    main()
