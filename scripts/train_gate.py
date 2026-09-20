from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.classifier import HallucinationClassifier
from guardian_of_truth.feature_extractor import FeatureExtractor
try:
    from scripts.build_min_ensemble import _classifier_bundle, _fallback_bundle
except ModuleNotFoundError:  # direct `python scripts/train_gate.py`
    from build_min_ensemble import _classifier_bundle, _fallback_bundle


GATE_FEATURE_NAMES = [
    "primary_score",
    "secondary_score",
    "score_min",
    "score_max",
    "score_mean",
    "score_abs_diff",
    "question_profile_who",
    "question_profile_when",
    "question_profile_where",
    "question_profile_count",
    "question_profile_generic",
    "audit_ok",
    "audit_bad",
    "score_path_main",
    "score_path_fallback",
    "typed_answer_len_bucket",
    "typed_exact_answer_shape",
    "tail_new_number_count",
    "tail_new_entity_count",
    "answer_overexplains_typed_question",
]

DEFAULT_GATE_POLICY = {
    "base_score": "min(primary_score, secondary_score)",
    "max_delta_up_generic": 0.08,
    "max_delta_up_count": 0.12,
    "max_delta_up_who_when_where": 0.05,
    "max_delta_down": 0.05,
    "high_score_threshold": 0.65,
    "max_delta_up_high_score": 0.12,
    "max_delta_up_strong_audit_risk": 0.15,
}


def _read_jsonl(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _labels(frame: pd.DataFrame) -> np.ndarray:
    if "is_hallucination" in frame.columns:
        return frame["is_hallucination"].astype(int).to_numpy()
    return frame["label"].astype(int).to_numpy()


def _gate_matrix(
    frame: pd.DataFrame,
    *,
    primary: HallucinationClassifier,
    secondary: HallucinationClassifier,
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    audit = AuditPayload.neutral(status="ok", mode="runtime", model_name="synthetic_gate", ok=True)
    rows: list[list[float]] = []
    min_scores: list[float] = []
    for row in frame.itertuples(index=False):
        prompt = str(getattr(row, "prompt"))
        answer = str(getattr(row, "model_answer", getattr(row, "answer", "")))
        features = extractor.extract(prompt, answer, audit)
        primary_score = float(primary.predict_proba(features)[0])
        secondary_score = float(secondary.predict_proba(features)[0])
        scores = [primary_score, secondary_score]
        min_scores.append(min(scores))
        profile = extractor._question_profile(prompt)
        text_values = extractor.extract_text_only(prompt, answer)
        text = dict(zip(FeatureExtractor.text_feature_names, text_values, strict=False))
        values = {
            "primary_score": primary_score,
            "secondary_score": secondary_score,
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": float(sum(scores) / len(scores)),
            "score_abs_diff": abs(primary_score - secondary_score),
            "audit_ok": 1.0,
            "audit_bad": 0.0,
            "score_path_main": 1.0,
            "score_path_fallback": 0.0,
            "typed_answer_len_bucket": float(text.get("typed_answer_len_bucket", 0.0)),
            "typed_exact_answer_shape": float(text.get("typed_exact_answer_shape", 0.0)),
            "tail_new_number_count": float(text.get("tail_new_number_count", 0.0)),
            "tail_new_entity_count": float(text.get("tail_new_entity_count", 0.0)),
            "answer_overexplains_typed_question": float(text.get("answer_overexplains_typed_question", 0.0)),
        }
        for name in ("who", "when", "where", "count", "generic"):
            values[f"question_profile_{name}"] = float(profile == name)
        rows.append([values[name] for name in GATE_FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float32), np.asarray(min_scores, dtype=np.float64)


def _cap_predictions(raw: np.ndarray, base: np.ndarray, X: np.ndarray) -> np.ndarray:
    profile_who = X[:, GATE_FEATURE_NAMES.index("question_profile_who")] > 0.5
    profile_when = X[:, GATE_FEATURE_NAMES.index("question_profile_when")] > 0.5
    profile_where = X[:, GATE_FEATURE_NAMES.index("question_profile_where")] > 0.5
    profile_count = X[:, GATE_FEATURE_NAMES.index("question_profile_count")] > 0.5
    primary = X[:, GATE_FEATURE_NAMES.index("primary_score")]
    secondary = X[:, GATE_FEATURE_NAMES.index("secondary_score")]
    max_up = np.full(len(base), DEFAULT_GATE_POLICY["max_delta_up_generic"], dtype=np.float64)
    max_up[profile_count] = DEFAULT_GATE_POLICY["max_delta_up_count"]
    max_up[profile_who | profile_when | profile_where] = DEFAULT_GATE_POLICY["max_delta_up_who_when_where"]
    high = (primary >= DEFAULT_GATE_POLICY["high_score_threshold"]) & (secondary >= DEFAULT_GATE_POLICY["high_score_threshold"])
    max_up[high] = np.maximum(max_up[high], DEFAULT_GATE_POLICY["max_delta_up_high_score"])
    delta = np.asarray(raw, dtype=np.float64) - base
    capped = base + np.minimum(max_up, np.maximum(-DEFAULT_GATE_POLICY["max_delta_down"], delta))
    return np.clip(capped, 0.0, 1.0)


def _ap_by(frame: pd.DataFrame, y: np.ndarray, proba: np.ndarray, column: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for value, indices in frame.groupby(column).groups.items():
        idx = list(indices)
        if len(set(y[idx])) < 2:
            continue
        out[str(value)] = float(average_precision_score(y[idx], proba[idx]))
    return out


def train_gate(
    dataset_path: str | Path,
    *,
    primary_dir: str | Path,
    secondary_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    frame = _read_jsonl(dataset_path)
    primary = HallucinationClassifier.load(primary_dir)
    secondary = HallucinationClassifier.load(secondary_dir)
    extractor = FeatureExtractor()
    train = frame[frame["split"].isin(["train", "gate_val"])].reset_index(drop=True)
    hard = frame[frame["split"] == "hard_val"].reset_index(drop=True)
    if len(hard) == 0:
        raise ValueError("Dataset has no hard_val rows.")
    X_train, min_train = _gate_matrix(train, primary=primary, secondary=secondary, extractor=extractor)
    y_train = _labels(train)
    X_hard, min_hard = _gate_matrix(hard, primary=primary, secondary=secondary, extractor=extractor)
    y_hard = _labels(hard)

    candidates: dict[str, Any] = {
        "capped_delta_logreg_v2": LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=42),
        "capped_delta_histgb_v2": HistGradientBoostingClassifier(learning_rate=0.04, max_iter=80, l2_regularization=0.1, random_state=42),
    }
    metrics = {"min_ensemble_v1": float(average_precision_score(y_hard, min_hard))}
    by_profile = {"min_ensemble_v1": _ap_by(hard, y_hard, min_hard, "question_profile")}
    by_variant = {"min_ensemble_v1": _ap_by(hard, y_hard, min_hard, "variant_type")}
    best_name = "min_ensemble_v1"
    best_model: Any | None = None
    best_ap = metrics[best_name]
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        raw = model.predict_proba(X_hard)[:, 1]
        proba = _cap_predictions(raw, min_hard, X_hard)
        ap = float(average_precision_score(y_hard, proba))
        metrics[name] = ap
        by_profile[name] = _ap_by(hard, y_hard, proba, "question_profile")
        by_variant[name] = _ap_by(hard, y_hard, proba, "variant_type")
        regressions = [
            profile
            for profile, base_ap in by_profile["min_ensemble_v1"].items()
            if by_profile[name].get(profile, base_ap) < base_ap - 0.01
        ]
        typed_regression = any(profile in {"who", "where", "when"} for profile in regressions)
        allowed = not typed_regression and not regressions
        if allowed and ap > best_ap:
            best_name = name
            best_model = model
            best_ap = ap

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary_path = Path(primary_dir)
    secondary_path = Path(secondary_dir)
    for name in ["detector.joblib", "scaler.joblib", "fallback.joblib"]:
        shutil.copy2(primary_path / name, output / name)
    ensemble: dict[str, Any] = {
        "combiner": "min" if best_model is None else best_name,
        "classifiers": [_classifier_bundle(secondary_path)],
        "fallback_classifiers": [_fallback_bundle(secondary_path)],
    }
    if best_model is not None:
        ensemble.update(
            {
                "combiner": "capped_delta_gate",
                "gate_model": best_model,
                "gate_feature_names": GATE_FEATURE_NAMES,
                "gate_policy": DEFAULT_GATE_POLICY,
            }
        )
    joblib.dump(ensemble, output / "ensemble.joblib")
    summary = {
        "model_family": "gated_ensemble",
        "primary_model_dir": str(primary_dir),
        "secondary_model_dir": str(secondary_dir),
        "combiner": ensemble["combiner"],
        "gate_feature_names": GATE_FEATURE_NAMES if best_model is not None else [],
        "selection_metric": "hard_val_average_precision",
        "selected_gate": best_name,
        "selected_combiner": ensemble["combiner"],
        "gate_policy": DEFAULT_GATE_POLICY if best_model is not None else {},
        "hard_val_ap": metrics,
        "hard_val_ap_by_profile": by_profile,
        "hard_val_ap_by_variant_type": by_variant,
        "hard_val_rows": int(len(hard)),
        "dataset_path": str(dataset_path),
        "public_calibration": "none",
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a production-loadable gate over two base models.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--secondary-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = train_gate(
        args.dataset_path,
        primary_dir=args.primary_dir,
        secondary_dir=args.secondary_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
