from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from guardian_of_truth.feature_extractor import FeatureExtractor


PROMOTED_BASELINE = 0.579354


def _safe_ap(frame: pd.DataFrame) -> float | None:
    if len(frame) == 0 or frame["is_hallucination"].nunique() < 2:
        return None
    return float(average_precision_score(frame["is_hallucination"], frame["is_hallucination_proba"]))


def _safe_roc(frame: pd.DataFrame) -> float | None:
    if len(frame) == 0 or frame["is_hallucination"].nunique() < 2:
        return None
    return float(roc_auc_score(frame["is_hallucination"], frame["is_hallucination_proba"]))


def build_report(scored_csv: str | Path, *, baseline: float = PROMOTED_BASELINE) -> dict:
    frame = pd.read_csv(scored_csv)
    extractor = FeatureExtractor()
    frame["_profile"] = frame["prompt"].map(lambda value: extractor._question_profile(str(value)))
    report = {
        "row_count": int(len(frame)),
        "baseline_pr_auc": float(baseline),
        "pr_auc": _safe_ap(frame),
        "roc_auc": _safe_roc(frame),
        "ap_by_profile": {},
        "ap_by_score_path": {},
        "ap_by_audit_status": {},
    }
    report["delta_vs_baseline"] = None if report["pr_auc"] is None else float(report["pr_auc"] - baseline)
    for column, key in [("_profile", "ap_by_profile"), ("score_path", "ap_by_score_path"), ("audit_status", "ap_by_audit_status")]:
        if column not in frame.columns:
            continue
        for value, chunk in frame.groupby(column):
            ap = _safe_ap(chunk)
            report[key][str(value)] = {"rows": int(len(chunk)), "ap": ap}
    return report


def _markdown(report: dict) -> str:
    lines = [
        "# Candidate Metric Report",
        "",
        f"- Rows: {report['row_count']}",
        f"- PR-AUC: {report['pr_auc']}",
        f"- ROC-AUC: {report['roc_auc']}",
        f"- Promoted baseline PR-AUC: {report['baseline_pr_auc']}",
        f"- Delta vs baseline: {report['delta_vs_baseline']}",
        "",
        "## AP by profile",
    ]
    for profile, values in sorted(report["ap_by_profile"].items()):
        lines.append(f"- {profile}: rows={values['rows']} ap={values['ap']}")
    lines.append("")
    lines.append("## AP by score path")
    for path, values in sorted(report["ap_by_score_path"].items()):
        lines.append(f"- {path}: rows={values['rows']} ap={values['ap']}")
    lines.append("")
    lines.append("## AP by audit status")
    for status, values in sorted(report["ap_by_audit_status"].items()):
        lines.append(f"- {status}: rows={values['rows']} ap={values['ap']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report PR-AUC/ROC-AUC and slices for a scored benchmark CSV.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--json-output", default="outputs/metric_report.json")
    parser.add_argument("--md-output", default="outputs/metric_report.md")
    parser.add_argument("--baseline", type=float, default=PROMOTED_BASELINE)
    args = parser.parse_args()

    report = build_report(args.scored_csv, baseline=args.baseline)
    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
