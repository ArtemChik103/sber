from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.drift_overlay import DRIFT_OVERLAY_CAPS, apply_drift_overlay_frame


V4_CAP082_PR_AUC = 0.583058
V5_BEST_FAILED_PR_AUC = 0.583571


def _safe_ap(frame: pd.DataFrame, scores: pd.Series | None = None) -> float | None:
    if "is_hallucination" not in frame.columns or len(frame) == 0 or frame["is_hallucination"].nunique() < 2:
        return None
    return float(average_precision_score(frame["is_hallucination"], scores if scores is not None else frame["is_hallucination_proba"]))


def apply_overlay_for_cap(frame: pd.DataFrame, cap: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = apply_drift_overlay_frame(frame, cap=cap)
    changed = output[output["drift_overlay_v6_delta"] > 1e-12]
    report: dict[str, Any] = {
        "cap": cap,
        "rows": int(len(output)),
        "changed_rows": int(len(changed)),
        "changed_by_profile": changed["drift_overlay_v6_profile"].value_counts().astype(int).to_dict(),
        "decision_counts": output["drift_overlay_v6_reason"].value_counts().astype(int).to_dict(),
        "never_decreased": bool((output["drift_overlay_v6_delta"] >= -1e-12).all()),
    }
    if "is_hallucination" in output.columns:
        base_ap = _safe_ap(output, output["base_is_hallucination_proba"])
        overlay_ap = _safe_ap(output, output["is_hallucination_proba"])
        report["base_pr_auc"] = base_ap
        report["overlay_pr_auc"] = overlay_ap
        report["delta_vs_base"] = None if base_ap is None or overlay_ap is None else float(overlay_ap - base_ap)
        report["delta_vs_v4_cap082"] = None if overlay_ap is None else float(overlay_ap - V4_CAP082_PR_AUC)
        report["delta_vs_v5_best_failed"] = None if overlay_ap is None else float(overlay_ap - V5_BEST_FAILED_PR_AUC)
        report["changed_precision"] = float((changed["is_hallucination"] == 1).mean()) if len(changed) else None
        report["changed_false_positives"] = int((changed["is_hallucination"] == 0).sum())
        report["changed_true_positives"] = int((changed["is_hallucination"] == 1).sum())
        report["generic_ap_base"] = _safe_ap(output[output["_profile"] == "generic"], output.loc[output["_profile"] == "generic", "base_is_hallucination_proba"])
        report["generic_ap_overlay"] = _safe_ap(output[output["_profile"] == "generic"])
        fallback_mask = output["score_path"].astype(str).eq("fallback") if "score_path" in output.columns else pd.Series(False, index=output.index)
        report["fallback_ap_base"] = _safe_ap(output[fallback_mask], output.loc[fallback_mask, "base_is_hallucination_proba"])
        report["fallback_ap_overlay"] = _safe_ap(output[fallback_mask])
        report["major_profile_regressions"] = _major_profile_regressions(output)
        report["minimum_gate_pass"] = bool(
            overlay_ap is not None
            and overlay_ap >= 0.586
            and report["changed_precision"] is not None
            and report["changed_precision"] >= 0.65
            and report["changed_false_positives"] <= 10
            and 10 <= report["changed_rows"] <= 80
            and report["never_decreased"]
            and report["generic_ap_overlay"] is not None
            and report["generic_ap_base"] is not None
            and report["generic_ap_overlay"] > report["generic_ap_base"]
            and not report["major_profile_regressions"]
        )
        report["strong_gate_pass"] = bool(
            overlay_ap is not None
            and overlay_ap >= 0.590
            and report["changed_precision"] is not None
            and report["changed_precision"] >= 0.70
            and report["changed_false_positives"] <= 8
            and report["fallback_ap_overlay"] is not None
            and report["fallback_ap_base"] is not None
            and report["fallback_ap_overlay"] > report["fallback_ap_base"]
            and report["generic_ap_overlay"] is not None
            and report["generic_ap_base"] is not None
            and report["generic_ap_overlay"] - report["generic_ap_base"] >= 0.01
        )
    return output, report


def apply_overlay(scored_csv: str | Path, output_prefix: str | Path, *, caps: tuple[float, ...] = DRIFT_OVERLAY_CAPS) -> dict[str, Any]:
    frame = pd.read_csv(scored_csv)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for cap in caps:
        output, report = apply_overlay_for_cap(frame, cap)
        suffix = f"{int(round(cap * 100)):03d}"
        csv_path = prefix.with_name(prefix.name + f"_cap{suffix}.csv")
        json_path = prefix.with_name(prefix.name + f"_cap{suffix}.json")
        changed_path = prefix.with_name(prefix.name + f"_cap{suffix}_changed_rows.csv")
        output.to_csv(csv_path, index=False)
        output[output["drift_overlay_v6_delta"] > 1e-12].sort_values("drift_overlay_v6_delta", ascending=False).to_csv(changed_path, index=False)
        report.update({"output_csv": str(csv_path), "json_output": str(json_path), "changed_rows_output": str(changed_path)})
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)
    best = max(reports, key=lambda item: item.get("overlay_pr_auc") or float("-inf"))
    summary = {
        "scored_csv": str(scored_csv),
        "output_prefix": str(output_prefix),
        "v4_cap082_pr_auc": V4_CAP082_PR_AUC,
        "v5_best_failed_pr_auc": V5_BEST_FAILED_PR_AUC,
        "reports": reports,
        "best": best,
    }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary["summary_output"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _major_profile_regressions(output: pd.DataFrame) -> dict[str, float]:
    regressions: dict[str, float] = {}
    for profile, chunk in output.groupby("_profile"):
        if len(chunk) < 100 or chunk["is_hallucination"].nunique() < 2:
            continue
        base = average_precision_score(chunk["is_hallucination"], chunk["base_is_hallucination_proba"])
        overlay = average_precision_score(chunk["is_hallucination"], chunk["is_hallucination_proba"])
        if overlay - base < -0.01:
            regressions[str(profile)] = float(overlay - base)
    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply unsupported-tail / answer-drift overlay v6 for multiple caps.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--caps", default="0.55,0.60,0.65")
    args = parser.parse_args()

    caps = tuple(float(value.strip()) for value in args.caps.split(",") if value.strip())
    summary = apply_overlay(args.scored_csv, args.output_prefix, caps=caps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
