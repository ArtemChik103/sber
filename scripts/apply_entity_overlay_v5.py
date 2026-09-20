from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.entity_overlay import ENTITY_OVERLAY_CAPS, apply_entity_overlay_frame


PROMOTED_V4_CAP082 = 0.583058


def apply_overlay_for_cap(frame: pd.DataFrame, cap: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = apply_entity_overlay_frame(frame, cap=cap)
    changed = output[output["entity_overlay_v5_delta"] > 1e-12]
    report: dict[str, Any] = {
        "cap": cap,
        "rows": int(len(output)),
        "changed_rows": int(len(changed)),
        "changed_by_profile": changed["entity_overlay_v5_profile"].value_counts().astype(int).to_dict(),
        "decision_counts": output["entity_overlay_v5_reason"].value_counts().astype(int).to_dict(),
        "never_decreased": bool((output["entity_overlay_v5_delta"] >= -1e-12).all()),
        "generic_changed_rows": int((changed.get("_profile", pd.Series(dtype=str)) == "generic").sum()),
    }
    if "is_hallucination" in output.columns:
        report["base_pr_auc"] = float(average_precision_score(output["is_hallucination"], output["base_is_hallucination_proba"]))
        report["overlay_pr_auc"] = float(average_precision_score(output["is_hallucination"], output["is_hallucination_proba"]))
        report["delta_vs_base"] = float(report["overlay_pr_auc"] - report["base_pr_auc"])
        report["delta_vs_v4_cap082"] = float(report["overlay_pr_auc"] - PROMOTED_V4_CAP082)
        report["changed_precision"] = float((changed["is_hallucination"] == 1).mean()) if len(changed) else None
        report["changed_false_positives"] = int((changed["is_hallucination"] == 0).sum())
        report["changed_true_positives"] = int((changed["is_hallucination"] == 1).sum())
        report["minimum_gate_pass"] = bool(
            report["overlay_pr_auc"] >= 0.586
            and (report["changed_precision"] is not None and report["changed_precision"] >= 0.75)
            and report["changed_false_positives"] <= 5
            and 5 <= report["changed_rows"] <= 60
            and report["never_decreased"]
            and report["generic_changed_rows"] == 0
        )
    return output, report


def apply_overlay(scored_csv: str | Path, output_prefix: str | Path, *, caps: tuple[float, ...] = ENTITY_OVERLAY_CAPS) -> dict[str, Any]:
    frame = pd.read_csv(scored_csv)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for cap in caps:
        output, report = apply_overlay_for_cap(frame, cap)
        suffix = str(cap).replace(".", "")
        csv_path = prefix.with_name(prefix.name + f"_cap{suffix}.csv")
        json_path = prefix.with_name(prefix.name + f"_cap{suffix}.json")
        changed_path = prefix.with_name(prefix.name + f"_cap{suffix}_changed_rows.csv")
        output.to_csv(csv_path, index=False)
        output[output["entity_overlay_v5_delta"] > 1e-12].sort_values("entity_overlay_v5_delta", ascending=False).to_csv(changed_path, index=False)
        report["output_csv"] = str(csv_path)
        report["json_output"] = str(json_path)
        report["changed_rows_output"] = str(changed_path)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)
    best = max(reports, key=lambda item: item.get("overlay_pr_auc", float("-inf")))
    summary = {
        "scored_csv": str(scored_csv),
        "output_prefix": str(output_prefix),
        "v4_cap082_pr_auc": PROMOTED_V4_CAP082,
        "reports": reports,
        "best": best,
    }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary["summary_output"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ultra-strict entity mismatch overlay v5 for multiple caps.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--caps", default="0.65,0.72,0.82")
    args = parser.parse_args()

    caps = tuple(float(value.strip()) for value in args.caps.split(",") if value.strip())
    summary = apply_overlay(args.scored_csv, args.output_prefix, caps=caps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
