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
        score_transform: str = "predict_proba",
    ) -> None:
        config = load_yaml(CONFIG_DIR / "model.yaml")
        detector_cfg = config["detector"]
        self.feature_names = feature_names or []
        self.feature_indices = feature_indices or []
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.model = model if model is not None else LogisticRegression(
            C=detector_cfg["C"],
            class_weight=detector_cfg["class_weight"],
            max_iter=detector_cfg["max_iter"],
            random_state=detector_cfg["random_state"],
        )
        self.calibrator = calibrator
        self.calibration_kind = calibration_kind
        self.score_transform = score_transform

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        calibration: str = "isotonic",
        score_transform: str | None = None,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> "HallucinationClassifier":
        X_train = self._select_features(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)
        self.calibration_kind = "none"
        self.calibrator = None
        self.score_transform = score_transform or calibration or "predict_proba"

        if X_val is None or y_val is None or len(np.unique(y_val)) < 2:
            return self

        X_val = self._select_features(X_val)
        raw_scores = self._raw_scores(self.scaler.transform(X_val))

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
        if self.score_transform == "predict_proba":
            return base_proba

        raw_scores = self._raw_scores(scaled)
        if self.score_transform == "raw_margin_sigmoid":
            return self._sigmoid(raw_scores)
        if self.calibrator is None or self.calibration_kind == "none":
            return base_proba
        if self.score_transform == "isotonic" and self.calibration_kind == "isotonic":
            return np.asarray(self.calibrator.transform(raw_scores), dtype=np.float64)
        if self.score_transform == "sigmoid" and self.calibration_kind == "sigmoid":
            return np.asarray(self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1], dtype=np.float64)
        return base_proba

    def _raw_scores(self, scaled: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(scaled), dtype=np.float64)
        proba = np.clip(self.model.predict_proba(scaled)[:, 1], 1e-6, 1.0 - 1e-6)
        return np.log(proba / (1.0 - proba))

    @staticmethod
    def _sigmoid(scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))

    def _select_features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            name_indices = self._indices_from_feature_names(X.shape[0])
            if name_indices is not None:
                return X[name_indices]
            if self.feature_indices and X.shape[0] != len(self.feature_indices):
                return X[self.feature_indices]
            return X
        name_indices = self._indices_from_feature_names(X.shape[1])
        if name_indices is not None:
            return X[:, name_indices]
        if self.feature_indices and X.shape[1] != len(self.feature_indices):
            return X[:, self.feature_indices]
        return X

    def _indices_from_feature_names(self, input_width: int) -> list[int] | None:
        if not self.feature_names or input_width == len(self.feature_names):
            return None
        try:
            from guardian_of_truth.feature_extractor import FeatureExtractor
        except ImportError:  # pragma: no cover
            return None
        base_names = FeatureExtractor.api_feature_names + FeatureExtractor.text_feature_names
        evidence_names = base_names + FeatureExtractor.evidence_feature_names
        current_names = evidence_names if input_width == len(evidence_names) else base_names
        if input_width != len(current_names):
            return None
        name_to_index = {name: idx for idx, name in enumerate(current_names)}
        if not all(name in name_to_index for name in self.feature_names):
            return None
        return [name_to_index[name] for name in self.feature_names]

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
                "score_transform": self.score_transform,
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
            score_transform=detector_bundle.get("score_transform", detector_bundle.get("calibration_kind", "predict_proba")),
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
            "score_transform": classifier.score_transform,
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
        score_transform=bundle.get("score_transform", bundle.get("calibration_kind", "predict_proba")),
    )


def save_training_summary(model_dir: str | Path, summary: dict[str, Any]) -> None:
    target = Path(model_dir) / "training_summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
