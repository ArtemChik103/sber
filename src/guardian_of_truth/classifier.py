from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guardian_of_truth.utils import CONFIG_DIR, MODEL_DIR, load_yaml


class HallucinationClassifier:
    def __init__(
        self,
        *,
        feature_names: list[str] | None = None,
        feature_indices: list[int] | None = None,
        scaler: StandardScaler | None = None,
        model: LogisticRegression | None = None,
        calibrator: Any | None = None,
        calibration_kind: str = "none",
    ) -> None:
        config = load_yaml(CONFIG_DIR / "model.yaml")
        detector_cfg = config["detector"]
        self.feature_names = feature_names or []
        self.feature_indices = feature_indices or []
        self.scaler = scaler or StandardScaler()
        self.model = model or LogisticRegression(
            C=detector_cfg["C"],
            class_weight=detector_cfg["class_weight"],
            max_iter=detector_cfg["max_iter"],
            random_state=detector_cfg["random_state"],
        )
        self.calibrator = calibrator
        self.calibration_kind = calibration_kind

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        calibration: str = "isotonic",
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> "HallucinationClassifier":
        X_train = self._select_features(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)
        self.calibration_kind = "none"
        self.calibrator = None

        if X_val is None or y_val is None or len(np.unique(y_val)) < 2:
            return self

        X_val = self._select_features(X_val)
        raw_scores = self.model.decision_function(self.scaler.transform(X_val))

        if calibration == "isotonic":
            try:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(raw_scores, y_val, sample_weight=sample_weight_val)
                self.calibrator = calibrator
                self.calibration_kind = "isotonic"
                return self
            except ValueError:
                calibration = "sigmoid"

        if calibration == "sigmoid":
            calibrator = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
            calibrator.fit(raw_scores.reshape(-1, 1), y_val, sample_weight=sample_weight_val)
            self.calibrator = calibrator
            self.calibration_kind = "sigmoid"

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._select_features(np.asarray(X, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        scaled = self.scaler.transform(X)
        base_proba = self.model.predict_proba(scaled)[:, 1]
        if self.calibrator is None or self.calibration_kind == "none":
            return base_proba

        raw_scores = self.model.decision_function(scaled)
        if self.calibration_kind == "isotonic":
            return np.asarray(self.calibrator.transform(raw_scores), dtype=np.float64)
        if self.calibration_kind == "sigmoid":
            return np.asarray(self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1], dtype=np.float64)
        return base_proba

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            if self.feature_indices and X.shape[0] != len(self.feature_indices):
                return X[self.feature_indices]
            return X
        if self.feature_indices and X.shape[1] != len(self.feature_indices):
            return X[:, self.feature_indices]
        return X

    def save(self, model_dir: str | Path = MODEL_DIR, *, prefix: str = "") -> None:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        detector_name = f"{prefix}detector.joblib" if prefix else "detector.joblib"
        scaler_name = f"{prefix}scaler.joblib" if prefix else "scaler.joblib"

        joblib.dump(
            {
                "model": self.model,
                "calibrator": self.calibrator,
                "calibration_kind": self.calibration_kind,
                "feature_names": self.feature_names,
                "feature_indices": self.feature_indices,
            },
            model_path / detector_name,
        )
        joblib.dump(self.scaler, model_path / scaler_name)

    @classmethod
    def load(cls, model_dir: str | Path = MODEL_DIR, *, prefix: str = "") -> "HallucinationClassifier":
        model_path = Path(model_dir)
        detector_name = f"{prefix}detector.joblib" if prefix else "detector.joblib"
        scaler_name = f"{prefix}scaler.joblib" if prefix else "scaler.joblib"
        detector_bundle = joblib.load(model_path / detector_name)
        scaler = joblib.load(model_path / scaler_name)
        return cls(
            feature_names=detector_bundle.get("feature_names", []),
            feature_indices=detector_bundle.get("feature_indices", []),
            scaler=scaler,
            model=detector_bundle["model"],
            calibrator=detector_bundle.get("calibrator"),
            calibration_kind=detector_bundle.get("calibration_kind", "none"),
        )


def save_fallback_bundle(classifier: HallucinationClassifier, model_dir: str | Path = MODEL_DIR) -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": classifier.scaler,
            "model": classifier.model,
            "calibrator": classifier.calibrator,
            "calibration_kind": classifier.calibration_kind,
            "feature_names": classifier.feature_names,
            "feature_indices": classifier.feature_indices,
        },
        model_path / "fallback.joblib",
    )


def load_fallback_bundle(model_dir: str | Path = MODEL_DIR) -> HallucinationClassifier:
    bundle = joblib.load(Path(model_dir) / "fallback.joblib")
    return HallucinationClassifier(
        feature_names=bundle.get("feature_names", []),
        feature_indices=bundle.get("feature_indices", []),
        scaler=bundle["scaler"],
        model=bundle["model"],
        calibrator=bundle.get("calibrator"),
        calibration_kind=bundle.get("calibration_kind", "none"),
    )


def save_training_summary(model_dir: str | Path, summary: dict[str, Any]) -> None:
    target = Path(model_dir) / "training_summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
