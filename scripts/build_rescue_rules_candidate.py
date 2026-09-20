from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib


def _classifier_bundle(model_dir: Path) -> dict[str, Any]:
    detector = joblib.load(model_dir / "detector.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    return {
        "scaler": scaler,
        "model": detector["model"],
        "calibrator": detector.get("calibrator"),
        "calibration_kind": detector.get("calibration_kind", "none"),
        "score_transform": detector.get("score_transform", detector.get("calibration_kind", "predict_proba")),
        "feature_names": detector.get("feature_names", []),
        "feature_indices": detector.get("feature_indices", []),
    }


def _fallback_bundle(model_dir: Path) -> dict[str, Any]:
    return joblib.load(model_dir / "fallback.joblib")


def build_rescue_rules_candidate(
    primary_dir: str | Path,
    secondary_dir: str | Path,
    output_dir: str | Path,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = Path(primary_dir)
    secondary = Path(secondary_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["detector.joblib", "scaler.joblib", "fallback.joblib"]:
        shutil.copy2(primary / name, output / name)
    rescue_policy = {
        "max_delta_up_generic": 0.03,
        "max_delta_up_who_when_where": 0.05,
        "max_delta_up_count": 0.08,
        "max_delta_up_strong_audit_risk": 0.10,
        "wrong_fact_threshold": 0.80,
        "contradiction_threshold": 0.50,
        "unsupported_threshold": 0.82,
        "allow_unsupported_rescue": False,
        "low_confidence_ceiling": 0.30,
        "fallback_text_delta": 0.03,
        **(policy or {}),
    }
    joblib.dump(
        {
            "combiner": "rescue_rules",
            "classifiers": [_classifier_bundle(secondary)],
            "fallback_classifiers": [_fallback_bundle(secondary)],
            "gate_policy": rescue_policy,
            "selection_note": "Rule-based monotonic upward rescue from min(primary, secondary); no learned gate.",
        },
        output / "ensemble.joblib",
    )
    summary = {
        "model_family": "rescue_rules",
        "primary_model_dir": str(primary),
        "secondary_model_dir": str(secondary),
        "combiner": "rescue_rules",
        "policy": rescue_policy,
        "selection_constraints": {
            "base_score": "min(primary, secondary)",
            "never_decrease_base": True,
            "no_learned_model": True,
            "default_max_delta": 0.03,
            "who_where_when_max_delta": 0.05,
            "count_max_delta": 0.08,
            "strong_audit_max_delta": 0.10,
        },
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a production-loadable monotonic rescue-rules candidate.")
    parser.add_argument("--primary-dir", default="outputs/candidate_ablation_old_full")
    parser.add_argument("--secondary-dir", default="outputs/candidate_old_full_targeted_lite_v1")
    parser.add_argument("--output-dir", default="outputs/candidate_rescue_rules_v1")
    args = parser.parse_args()
    print(json.dumps(build_rescue_rules_candidate(args.primary_dir, args.secondary_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
