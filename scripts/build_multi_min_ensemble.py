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


def build_multi_min_ensemble(
    primary_dir: str | Path,
    robust_dir: str | Path,
    output_dir: str | Path,
    *,
    extra_secondary_dir: str | Path | None = None,
) -> dict:
    primary = Path(primary_dir)
    robust = Path(robust_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["detector.joblib", "scaler.joblib", "fallback.joblib"]:
        shutil.copy2(primary / name, output / name)
    secondaries = [robust]
    if extra_secondary_dir is not None:
        secondaries.insert(0, Path(extra_secondary_dir))
    joblib.dump(
        {
            "combiner": "min",
            "classifiers": [_classifier_bundle(path) for path in secondaries],
            "fallback_classifiers": [_fallback_bundle(path) for path in secondaries],
        },
        output / "ensemble.joblib",
    )
    summary_path = primary / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update({
        "model_family": "multi_min_ensemble",
        "primary_model_dir": str(primary),
        "robust_model_dir": str(robust),
        "extra_secondary_dir": str(extra_secondary_dir) if extra_secondary_dir else None,
        "combiner": "min",
        "base_selected_variants": [primary.name] + [path.name for path in secondaries],
        "selection_note": "Conservative min combiner with one audit-robust secondary; no learned gates.",
    })
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a production-loadable min ensemble preserving the promoted base and adding a robust classifier.")
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--robust-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra-secondary-dir", default=None)
    args = parser.parse_args()
    print(json.dumps(build_multi_min_ensemble(args.primary_dir, args.robust_dir, args.output_dir, extra_secondary_dir=args.extra_secondary_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
