from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.refute_overlay import REFUTE_OVERLAY_CAP, apply_overlay_frame


CAPS = (0.65, 0.72, REFUTE_OVERLAY_CAP)
PROMOTED_BASELINE = 0.579354


def apply_overlay_for_cap(frame: pd.DataFrame, cap: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = apply_overlay_frame(frame, cap=cap)
    changed = output[output["refute_overlay_delta"] > 1e-12]
    report: dict[str, Any] = {
        "cap": cap,
        "rows": int(len(output)),
        "changed_rows": int(len(changed)),
        "decision_counts": output["refute_overlay_v4"].value_counts().astype(int).to_dict(),
        "expected_kind_counts": output["refute_overlay_v4_expected_kind"].value_counts().astype(int).to_dict(),
        "never_decreased": bool((output["refute_overlay_delta"] >= -1e-12).all()),
    }
    if "is_hallucination" in output.columns:
        report["base_pr_auc"] = float(average_precision_score(output["is_hallucination"], output["base_is_hallucination_proba"]))
        report["overlay_pr_auc"] = float(average_precision_score(output["is_hallucination"], output["is_hallucination_proba"]))
        report["delta_vs_base"] = float(report["overlay_pr_auc"] - report["base_pr_auc"])
        report["delta_vs_promoted_baseline"] = float(report["overlay_pr_auc"] - PROMOTED_BASELINE)
        report["changed_precision"] = float((changed["is_hallucination"] == 1).mean()) if len(changed) else None
        report["overlay_false_positives"] = int(((changed["is_hallucination"] == 0) & (changed["is_hallucination_proba"] >= 0.5)).sum())
        report["overlay_true_positive_lifts"] = int((changed["is_hallucination"] == 1).sum())
    return output, report


def apply_overlay(scored_csv: str | Path, output_prefix: str | Path, *, caps: tuple[float, ...] = CAPS) -> dict[str, Any]:
    frame = pd.read_csv(scored_csv)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for cap in caps:
        output, report = apply_overlay_for_cap(frame, cap)
        suffix = str(cap).replace(".", "")
        csv_path = prefix.with_name(prefix.name + f"_cap{suffix}.csv")
        json_path = prefix.with_name(prefix.name + f"_cap{suffix}.json")
        fp_path = prefix.with_name(prefix.name + f"_cap{suffix}_changed_false_positives.csv")
        tp_path = prefix.with_name(prefix.name + f"_cap{suffix}_changed_true_positives.csv")
        output.to_csv(csv_path, index=False)
        changed = output[output["refute_overlay_delta"] > 1e-12]
        changed[changed.get("is_hallucination", pd.Series(dtype=int)) == 0].sort_values("refute_overlay_delta", ascending=False).head(100).to_csv(fp_path, index=False)
        changed[changed.get("is_hallucination", pd.Series(dtype=int)) == 1].sort_values("refute_overlay_delta", ascending=False).head(100).to_csv(tp_path, index=False)
        report["output_csv"] = str(csv_path)
        report["json_output"] = str(json_path)
        report["top_changed_false_positives_output"] = str(fp_path)
        report["top_changed_true_positives_output"] = str(tp_path)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)
    best = max(reports, key=lambda item: item.get("overlay_pr_auc", float("-inf")))
    summary = {"scored_csv": str(scored_csv), "output_prefix": str(output_prefix), "reports": reports, "best": best}
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary["summary_output"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply subject-aligned numeric/date/count refute overlay v4 for multiple caps.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--caps", default="0.65,0.72,0.82")
    args = parser.parse_args()

    caps = tuple(float(value.strip()) for value in args.caps.split(",") if value.strip())
    summary = apply_overlay(args.scored_csv, args.output_prefix, caps=caps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
