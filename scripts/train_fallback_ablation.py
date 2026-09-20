from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from guardian_of_truth.classifier import HallucinationClassifier, load_fallback_bundle, save_fallback_bundle
from guardian_of_truth.feature_extractor import FeatureExtractor


def _read_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["is_hallucination"].astype(int).to_numpy()


def _text_matrix(frame: pd.DataFrame) -> np.ndarray:
    extractor = FeatureExtractor()
    rows = [
        extractor.extract_text_only(str(row.prompt), str(row.model_answer))
        for row in frame.itertuples(index=False)
    ]
    return np.asarray(rows, dtype=np.float32)


def _fit_classifier(model: Any, X: np.ndarray, y: np.ndarray) -> HallucinationClassifier:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return HallucinationClassifier(
        feature_names=FeatureExtractor.text_feature_names,
        scaler=scaler,
        model=model,
        score_transform="predict_proba",
    )


def train_fallback_ablation(
    dataset_path: str | Path,
    *,
    promoted_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frame = _read_jsonl(dataset_path)
    train = frame[frame["split"].isin(["train", "gate_val"])].reset_index(drop=True)
    hard = frame[frame["split"] == "hard_val"].reset_index(drop=True)
    if hard.empty:
        raise ValueError("Dataset has no hard_val rows.")
    X_train = _text_matrix(train)
    y_train = _labels(train)
    X_hard = _text_matrix(hard)
    y_hard = _labels(hard)

    current = load_fallback_bundle(promoted_dir)
    current_scores = current.predict_proba(X_hard)
    metrics: dict[str, float] = {"current_fallback": float(average_precision_score(y_hard, current_scores))}
    candidates = {
        "logreg_text_only": LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=42),
        "histgb_text_only": HistGradientBoostingClassifier(learning_rate=0.04, max_iter=80, l2_regularization=0.1, random_state=42),
        "extratrees_text_only": ExtraTreesClassifier(n_estimators=160, min_samples_leaf=3, class_weight="balanced", random_state=42),
    }
    best_name = "current_fallback"
    best_classifier = current
    best_ap = metrics[best_name]
    for name, model in candidates.items():
        classifier = _fit_classifier(model, X_train, y_train)
        proba = classifier.predict_proba(X_hard)
        ap = float(average_precision_score(y_hard, proba))
        metrics[name] = ap
        if ap > best_ap:
            best_name = name
            best_ap = ap
            best_classifier = classifier

    promoted = Path(promoted_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["detector.joblib", "scaler.joblib", "ensemble.joblib"]:
        source = promoted / name
        if source.exists():
            shutil.copy2(source, output / name)
    save_fallback_bundle(best_classifier, output)
    summary = {
        "model_family": "fallback_ablation_v2",
        "promoted_model_dir": str(promoted_dir),
        "dataset_path": str(dataset_path),
        "selected_fallback": best_name,
        "hard_val_fallback_ap": metrics,
        "hard_val_rows": int(len(hard)),
        "main_path": "copied_from_promoted",
        "public_calibration": "none",
    }
    base_summary_path = promoted / "training_summary.json"
    if base_summary_path.exists():
        summary["base_training_summary"] = json.loads(base_summary_path.read_text(encoding="utf-8"))
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text-only fallback ablations while preserving the promoted main path.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--promoted-dir", default="model")
    parser.add_argument("--output-dir", default="outputs/candidate_fallback_v2")
    args = parser.parse_args()
    summary = train_fallback_ablation(args.dataset_path, promoted_dir=args.promoted_dir, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
