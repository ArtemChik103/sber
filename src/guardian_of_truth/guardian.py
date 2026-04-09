from __future__ import annotations

import time
import threading
from queue import Queue
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from guardian_of_truth.api_client import GroqVerifier
from guardian_of_truth.classifier import HallucinationClassifier, load_fallback_bundle
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.utils import MODEL_DIR


@dataclass
class ScoringResult:
    is_hallucination: bool
    is_hallucination_proba: float
    t_model_sec: float = 0.0
    t_overhead_sec: float = 0.0
    t_total_sec: float = 0.0


class HeuristicFallbackClassifier:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Word count, digit density, year count, capitalized ratio, hedging ratio, punct density, lexical diversity.
        score = (
            0.15 * np.clip(X[:, 0] / 32.0, 0.0, 1.0)
            + 0.35 * X[:, 4]
            + 0.20 * X[:, 1]
            + 0.15 * np.clip(X[:, 2] / 3.0, 0.0, 1.0)
            + 0.15 * np.clip(1.0 - X[:, 6], 0.0, 1.0)
        )
        proba = 1.0 / (1.0 + np.exp(-4.0 * (score - 0.35)))
        return np.clip(proba, 0.0, 1.0)


class GuardianOfTruth:
    def __init__(
        self,
        *,
        verifier: GroqVerifier | None = None,
        extractor: FeatureExtractor | None = None,
        classifier: HallucinationClassifier | None = None,
        fallback_classifier: HallucinationClassifier | HeuristicFallbackClassifier | None = None,
        model_dir: str | Path = MODEL_DIR,
    ) -> None:
        self.verifier = verifier or GroqVerifier()
        self.extractor = extractor or FeatureExtractor()
        self.classifier = classifier or self._load_main_classifier(model_dir)
        self.fallback_classifier = fallback_classifier or self._load_fallback_classifier(model_dir)
        self.model_dir = Path(model_dir)

    def score(self, prompt: str, answer: str) -> ScoringResult:
        t0 = time.perf_counter()
        audit, t_model = self._verify_with_runtime_budget(prompt, answer)

        overhead_started = time.perf_counter()
        use_fallback = (not audit.ok) or (t_model > self.verifier.settings.total_timeout_sec)

        if use_fallback:
            proba = float(self.fallback_classifier.predict_proba(self.extractor.extract_text_only(answer))[0])
        else:
            features = self.extractor.extract(prompt, answer, audit)
            proba = float(self.classifier.predict_proba(features)[0])

        proba = float(min(1.0, max(0.0, proba)))
        t_overhead = time.perf_counter() - overhead_started
        t_total = time.perf_counter() - t0

        return ScoringResult(
            is_hallucination=bool(proba >= 0.5),
            is_hallucination_proba=proba,
            t_model_sec=t_model,
            t_overhead_sec=t_overhead,
            t_total_sec=t_total,
        )

    def _verify_with_runtime_budget(self, prompt: str, answer: str) -> tuple[object, float]:
        budget_sec = max(0.0, self.verifier.settings.total_timeout_sec - 0.02)
        result_queue: Queue[tuple[object, float] | BaseException] = Queue(maxsize=1)

        def worker() -> None:
            started = time.perf_counter()
            try:
                audit = self.verifier.verify(prompt, answer, mode="runtime")
                result_queue.put((audit, time.perf_counter() - started))
            except BaseException as exc:  # pragma: no cover
                result_queue.put(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=budget_sec)

        if thread.is_alive():
            return self._runtime_budget_exceeded_audit(), budget_sec

        item = result_queue.get()
        if isinstance(item, BaseException):  # pragma: no cover
            return self._runtime_budget_exceeded_audit(), budget_sec
        audit, elapsed = item
        return audit, elapsed

    def _runtime_budget_exceeded_audit(self):
        from guardian_of_truth.api_client import AuditPayload

        return AuditPayload.neutral(
            status="runtime_budget_exceeded",
            mode="runtime",
            model_name=getattr(self.verifier.settings, "runtime_model", None),
            ok=False,
        )

    @staticmethod
    def _load_main_classifier(model_dir: str | Path) -> HallucinationClassifier:
        try:
            return HallucinationClassifier.load(model_dir)
        except FileNotFoundError:
            # Neutral uncalibrated classifier placeholder if training artifacts are missing.
            dummy = HallucinationClassifier(feature_names=FeatureExtractor.api_feature_names + FeatureExtractor.text_feature_names)
            dummy.scaler.fit(np.zeros((2, 14), dtype=np.float32))
            dummy.model.fit(np.zeros((2, 14), dtype=np.float32), np.array([0, 1], dtype=np.int32))
            return dummy

    @staticmethod
    def _load_fallback_classifier(model_dir: str | Path) -> HallucinationClassifier | HeuristicFallbackClassifier:
        try:
            return load_fallback_bundle(model_dir)
        except FileNotFoundError:
            return HeuristicFallbackClassifier()
