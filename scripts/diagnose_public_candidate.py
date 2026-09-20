from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.feature_extractor import FeatureExtractor


def _ap(frame: pd.DataFrame) -> float:
    if "is_hallucination" not in frame.columns or frame["is_hallucination"].nunique() < 2:
        return 0.0
    return float(average_precision_score(frame["is_hallucination"].astype(int), frame["is_hallucination_proba"]))


def _profile(prompt: str) -> str:
    return FeatureExtractor()._question_profile(str(prompt))


def public_candidate_diagnostics(
    baseline_csv: str | Path,
    candidate_csv: str | Path,
    *,
    output_md: str | Path = "outputs/diagnostics_candidate_audit_robust_v1.md",
) -> dict:
    baseline = pd.read_csv(baseline_csv)
    candidate = pd.read_csv(candidate_csv)
    summary = {
        "baseline_ap": _ap(baseline),
        "candidate_ap": _ap(candidate),
        "rows": int(len(candidate)),
        "score_path_counts": candidate.get("score_path", pd.Series(dtype=str)).fillna("unknown").value_counts().astype(int).to_dict(),
        "audit_status_counts": candidate.get("audit_status", pd.Series(dtype=str)).fillna("unknown").value_counts().astype(int).to_dict(),
        "profile_ap": {},
    }
    candidate = candidate.copy()
    candidate["question_profile"] = candidate["prompt"].map(_profile)
    baseline = baseline.copy()
    baseline["question_profile"] = baseline["prompt"].map(_profile)
    for profile, chunk in candidate.groupby("question_profile"):
        base_chunk = baseline[baseline["question_profile"] == profile]
        summary["profile_ap"][profile] = {"baseline": _ap(base_chunk), "candidate": _ap(chunk), "rows": int(len(chunk))}
    lines = [
        "# Candidate audit robust v1 diagnostics",
        "",
        f"- Baseline PR-AUC: {summary['baseline_ap']:.6f}",
        f"- Candidate PR-AUC: {summary['candidate_ap']:.6f}",
        f"- Delta: {summary['candidate_ap'] - summary['baseline_ap']:.6f}",
        f"- Score path counts: `{json.dumps(summary['score_path_counts'], ensure_ascii=False)}`",
        f"- Audit status counts: `{json.dumps(summary['audit_status_counts'], ensure_ascii=False)}`",
        "",
        "## Profile AP",
    ]
    for profile, values in sorted(summary["profile_ap"].items()):
        lines.append(f"- {profile}: baseline {values['baseline']:.6f}, candidate {values['candidate']:.6f}, rows {values['rows']}")
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write public-style candidate diagnostics without fitting on public labels.")
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output-md", default="outputs/diagnostics_candidate_audit_robust_v1.md")
    args = parser.parse_args()
    print(json.dumps(public_candidate_diagnostics(args.baseline_csv, args.candidate_csv, output_md=args.output_md), ensure_ascii=False))


if __name__ == "__main__":
    main()
