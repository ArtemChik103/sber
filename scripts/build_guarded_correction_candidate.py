from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_multi_min_ensemble import _classifier_bundle, _fallback_bundle


DEFAULT_POLICY = {
    "max_down_clean_audit": 0.015,
    "max_up_clean_audit": 0.015,
    "max_down_noisy_correct": 0.12,
    "max_up_neutral_wrong": 0.08,
    "high_disagreement": 0.22,
    "clean_h": 0.08,
    "clean_u": 0.08,
    "clean_wrong_field": 0.08,
    "noisy_h": 0.24,
    "noisy_u": 0.45,
    "noisy_wrong_field": 0.60,
    "neutral_wrong_base_ceiling": 0.55,
    "profile_text_risk": 0.5,
}


def build_guarded_correction_candidate(
    primary_dir: str | Path,
    robust_dir: str | Path,
    output_dir: str | Path,
    *,
    policy: dict | None = None,
) -> dict:
    primary = Path(primary_dir)
    robust = Path(robust_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["detector.joblib", "scaler.joblib", "fallback.joblib"]:
        shutil.copy2(primary / name, output / name)
    primary_classifiers: list[dict] = []
    primary_fallback_classifiers: list[dict] = []
    primary_ensemble = primary / "ensemble.joblib"
    primary_combiner = "single"
    if primary_ensemble.exists():
        bundle = joblib.load(primary_ensemble)
        primary_combiner = str(bundle.get("combiner", "single"))
        primary_classifiers = list(bundle.get("classifiers", []))
        primary_fallback_classifiers = list(bundle.get("fallback_classifiers", []))
    effective_policy = {**DEFAULT_POLICY, **(policy or {})}
    effective_policy.update(
        {
            "base_score_count": 1 + len(primary_classifiers),
            "base_fallback_score_count": 1 + len(primary_fallback_classifiers),
            "primary_combiner": primary_combiner,
        }
    )
    joblib.dump(
        {
            "combiner": "guarded_correction",
            "classifiers": primary_classifiers + [_classifier_bundle(robust)],
            "fallback_classifiers": primary_fallback_classifiers + [_fallback_bundle(robust)],
            "gate_policy": effective_policy,
        },
        output / "ensemble.joblib",
    )
    summary_path = primary / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "model_family": "guarded_correction",
            "primary_model_dir": str(primary),
            "robust_model_dir": str(robust),
            "combiner": "guarded_correction",
            "gate_policy": effective_policy,
            "selection_note": "Promoted baseline remains primary; robust model is only a statically capped correction layer.",
        }
    )
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a production-loadable guarded correction candidate.")
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--robust-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_guarded_correction_candidate(args.primary_dir, args.robust_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
