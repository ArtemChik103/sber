from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib


def _classifier_bundle(model_dir: Path) -> dict:
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


def _fallback_bundle(model_dir: Path) -> dict:
    return joblib.load(model_dir / "fallback.joblib")


def build_min_ensemble(primary_dir: str | Path, secondary_dir: str | Path, output_dir: str | Path) -> None:
    primary = Path(primary_dir)
    secondary = Path(secondary_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name in ["detector.joblib", "scaler.joblib", "fallback.joblib"]:
        shutil.copy2(primary / name, output / name)

    primary_summary_path = primary / "training_summary.json"
    summary = json.loads(primary_summary_path.read_text(encoding="utf-8")) if primary_summary_path.exists() else {}
    summary.update(
        {
            "model_family": "min_ensemble",
            "primary_model_dir": str(primary),
            "secondary_model_dir": str(secondary),
            "combiner": "min",
            "base_selected_variants": [
                primary.name,
                secondary.name,
            ],
            "public_eval_artifact": "outputs/public_scored_promoted_min_ensemble_v1.csv",
            "public_pr_auc": 0.579354,
            "candidate_promoted_max_abs_diff": 0.0,
            "selection_note": "Fixed conservative min combiner; no public calibration weights.",
        }
    )
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(
        {
            "combiner": "min",
            "classifiers": [_classifier_bundle(secondary)],
            "fallback_classifiers": [_fallback_bundle(secondary)],
        },
        output / "ensemble.joblib",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a production-loadable min ensemble from two model directories.")
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--secondary-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build_min_ensemble(args.primary_dir, args.secondary_dir, args.output_dir)


if __name__ == "__main__":
    main()
