from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from guardian_of_truth.api_client import GroqVerifier
from guardian_of_truth.classifier import HallucinationClassifier, load_fallback_bundle
from guardian_of_truth.evidence import EvidenceRetriever
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.refute_overlay import (
    BAYESIAN_BLIND_RESCUE_POLICY,
    DISABLED_REFUTE_OVERLAY_POLICY,
    SUPPORTED_REFUTE_OVERLAY_POLICIES,
    apply_policy_to_score,
)
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
        # Word count, prompt overlap, entity coverage, number coverage, new-content ratio, new-number ratio, type mismatch.
        score = (
            0.12 * np.clip(X[:, 0] / 32.0, 0.0, 1.0)
            + 0.24 * np.clip(1.0 - X[:, 1], 0.0, 1.0)
            + 0.14 * np.clip(1.0 - X[:, 2], 0.0, 1.0)
            + 0.12 * np.clip(1.0 - X[:, 3], 0.0, 1.0)
            + 0.14 * X[:, 4]
            + 0.08 * X[:, 5]
            + 0.16 * X[:, 6]
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
        evidence_retriever: EvidenceRetriever | None = None,
        evidence_cache_path: str | Path | None = None,
        use_evidence: bool = True,
        refute_overlay_policy: str = BAYESIAN_BLIND_RESCUE_POLICY,
    ) -> None:
        if refute_overlay_policy not in SUPPORTED_REFUTE_OVERLAY_POLICIES:
            raise ValueError(f"Unsupported refute overlay policy: {refute_overlay_policy}")
        self.verifier = verifier or GroqVerifier()
        self.extractor = extractor or FeatureExtractor()
        self.classifier = classifier or self._load_main_classifier(model_dir)
        self.fallback_classifier = fallback_classifier or self._load_fallback_classifier(model_dir)
        self.model_dir = Path(model_dir)
        default_kb_path = self.model_dir / "evidence.db"
        if evidence_retriever is not None:
            self.evidence_retriever = evidence_retriever
        elif use_evidence and default_kb_path.exists():
            self.evidence_retriever = EvidenceRetriever(default_kb_path)
        else:
            self.evidence_retriever = None
        self.evidence_cache_path = Path(evidence_cache_path) if evidence_cache_path else None
        self.use_evidence = use_evidence
        self.refute_overlay_policy = refute_overlay_policy
        (
            self.ensemble_classifiers,
            self.ensemble_fallback_classifiers,
            self.ensemble_combiner,
            self.ensemble_gate_model,
            self.ensemble_gate_feature_names,
            self.ensemble_gate_policy,
        ) = self._load_ensemble(model_dir)
        self.last_score_path = "unknown"
        self.last_audit_latency_sec: float | None = None
        self.last_audit_would_timeout: bool | None = None
        self.last_evidence_audit = None
        self.last_base_is_hallucination_proba: float | None = None
        self.last_refute_overlay_policy = refute_overlay_policy
        self.last_refute_overlay_reason = "disabled"
        self.last_refute_overlay_expected_kind = "none"
        self.last_refute_overlay_delta = 0.0

    def score(self, prompt: str, answer: str) -> ScoringResult:
        t0 = time.perf_counter()
        t_model = 0.0

        api_started = time.perf_counter()
        audit = self.verifier.verify(prompt, answer, mode="runtime")
        audit_latency_sec = getattr(self.verifier, "last_audit_latency_sec", None)
        if audit_latency_sec is None:
            audit_latency_sec = time.perf_counter() - api_started
        if not audit.cached and audit.status not in {"missing_api_key", "local_rate_limited"}:
            t_model = time.perf_counter() - api_started

        overhead_started = time.perf_counter()
        evidence_audit = None
        if self.use_evidence and self.evidence_retriever is not None and self.evidence_retriever.available:
            evidence_audit = self.evidence_retriever.audit(prompt, answer)
        self.last_evidence_audit = evidence_audit
        source_would_timeout = getattr(self.verifier, "last_source_audit_would_timeout", None)
        audit_would_timeout = (
            bool(source_would_timeout)
            if source_would_timeout is not None
            else bool(t_model > self.verifier.settings.total_timeout_sec)
        )
        use_fallback = (not audit.ok) or audit_would_timeout
        self.last_audit_latency_sec = float(audit_latency_sec)
        self.last_audit_would_timeout = audit_would_timeout
        self.last_score_path = "fallback" if use_fallback else "main"

        if use_fallback:
            text_features = self.extractor.extract_text_only(prompt, answer)
            scores = [float(self.fallback_classifier.predict_proba(text_features)[0])]
            scores.extend(float(classifier.predict_proba(text_features)[0]) for classifier in self.ensemble_fallback_classifiers)
            proba = self._combine_scores(scores, prompt=prompt, answer=answer, audit=audit, score_path="fallback")
        else:
            features = self.extractor.extract(prompt, answer, audit, evidence_audit=evidence_audit)
            scores = [float(self.classifier.predict_proba(features)[0])]
            scores.extend(float(classifier.predict_proba(features)[0]) for classifier in self.ensemble_classifiers)
            proba = self._combine_scores(scores, prompt=prompt, answer=answer, audit=audit, score_path="main")

        base_proba = float(min(1.0, max(0.0, proba)))
        proba, overlay_reason, overlay_kind = apply_policy_to_score(
            prompt,
            answer,
            base_proba,
            evidence_audit,
            policy=self.refute_overlay_policy,
            cache=getattr(self.verifier, "cache", None),
        )
        proba = float(min(1.0, max(0.0, proba)))
        self.last_base_is_hallucination_proba = base_proba
        self.last_refute_overlay_policy = self.refute_overlay_policy
        self.last_refute_overlay_reason = overlay_reason
        self.last_refute_overlay_expected_kind = overlay_kind
        self.last_refute_overlay_delta = float(proba - base_proba)
        t_overhead = time.perf_counter() - overhead_started
        t_total = time.perf_counter() - t0

        return ScoringResult(
            is_hallucination=bool(proba >= 0.5),
            is_hallucination_proba=proba,
            t_model_sec=t_model,
            t_overhead_sec=t_overhead,
            t_total_sec=t_total,
        )

    def _combine_scores(
        self,
        scores: list[float],
        *,
        prompt: str | None = None,
        answer: str | None = None,
        audit: AuditPayload | None = None,
        score_path: str = "unknown",
    ) -> float:
        if not scores:
            return 0.0
        if self.ensemble_combiner == "min":
            return float(min(scores))
        if self.ensemble_combiner == "max":
            return float(max(scores))
        if self.ensemble_combiner == "mean":
            return float(sum(scores) / len(scores))
        if self.ensemble_combiner in {"gate_logreg", "gate_histgb", "piecewise_gate"} and self.ensemble_gate_model is not None:
            gate_features = self._gate_features(
                scores,
                prompt=prompt or "",
                answer=answer or "",
                audit=audit,
                score_path=score_path,
            )
            if hasattr(self.ensemble_gate_model, "predict_proba"):
                return float(self.ensemble_gate_model.predict_proba(gate_features)[0, 1])
            if hasattr(self.ensemble_gate_model, "predict"):
                return float(np.asarray(self.ensemble_gate_model.predict(gate_features)).reshape(-1)[0])
        if self.ensemble_combiner == "capped_delta_gate":
            return self._capped_delta_score(
                scores,
                prompt=prompt or "",
                answer=answer or "",
                audit=audit,
                score_path=score_path,
            )
        if self.ensemble_combiner == "rescue_rules":
            return self._rescue_rules_score(
                scores,
                prompt=prompt or "",
                answer=answer or "",
                audit=audit,
                score_path=score_path,
            )
        if self.ensemble_combiner == "guarded_correction":
            return self._guarded_correction_score(
                scores,
                prompt=prompt or "",
                answer=answer or "",
                audit=audit,
                score_path=score_path,
            )
        return float(scores[0])

    def _guarded_correction_score(
        self,
        scores: list[float],
        *,
        prompt: str,
        answer: str,
        audit: AuditPayload | None,
        score_path: str,
    ) -> float:
        policy = {
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
            **(self.ensemble_gate_policy or {}),
        }
        base_count_key = "base_fallback_score_count" if score_path == "fallback" else "base_score_count"
        base_count = int(policy.get(base_count_key, 1))
        base_scores = scores[: max(1, min(base_count, len(scores)))]
        if str(policy.get("primary_combiner", "single")) == "min":
            base = float(min(base_scores))
        elif str(policy.get("primary_combiner", "single")) == "max":
            base = float(max(base_scores))
        elif str(policy.get("primary_combiner", "single")) == "mean":
            base = float(sum(base_scores) / len(base_scores))
        else:
            base = float(base_scores[0])
        robust = float(scores[-1]) if len(scores) > base_count else base
        if audit is None or score_path == "fallback":
            return base

        wrong_field = max(float(audit.we), float(audit.wn), float(audit.ue), float(audit.bt))
        is_clean = (
            bool(audit.ok)
            and float(audit.h) <= float(policy["clean_h"])
            and float(audit.u) <= float(policy["clean_u"])
            and wrong_field <= float(policy["clean_wrong_field"])
            and float(audit.x) <= 0.0
        )
        is_noisy_correct_shape = (
            bool(audit.ok)
            and float(audit.h) >= float(policy["noisy_h"])
            and (float(audit.u) >= float(policy["noisy_u"]) or wrong_field >= float(policy["noisy_wrong_field"]))
            and float(audit.x) <= 0.0
        )
        is_neutral_wrong_shape = is_clean and base <= float(policy["neutral_wrong_base_ceiling"])

        text_values = self.extractor.extract_text_only(prompt, answer)
        text_name_to_value = dict(zip(FeatureExtractor.text_feature_names, text_values, strict=False))
        profile_risk = max(
            float(text_name_to_value.get("question_type_mismatch", 0.0)),
            float(text_name_to_value.get("answer_overexplains_typed_question", 0.0)),
        )
        disagreement = abs(robust - base)
        raw_delta = robust - base

        if is_clean:
            if disagreement >= float(policy["high_disagreement"]) and profile_risk < float(policy["profile_text_risk"]):
                return base
            cap_up = float(policy["max_up_clean_audit"])
            cap_down = float(policy["max_down_clean_audit"])
            if is_neutral_wrong_shape and raw_delta > 0.0 and profile_risk >= float(policy["profile_text_risk"]):
                cap_up = max(cap_up, float(policy["max_up_neutral_wrong"]))
            return float(np.clip(base + min(cap_up, max(-cap_down, raw_delta)), 0.0, 1.0))

        if is_noisy_correct_shape and raw_delta < 0.0:
            return float(np.clip(base + max(-float(policy["max_down_noisy_correct"]), raw_delta), 0.0, 1.0))

        if raw_delta > 0.0 and (float(audit.x) >= 0.5 or wrong_field >= 0.75):
            return float(np.clip(base + min(float(policy["max_up_neutral_wrong"]), raw_delta), 0.0, 1.0))

        return base

    def _rescue_rules_score(
        self,
        scores: list[float],
        *,
        prompt: str,
        answer: str,
        audit: AuditPayload | None,
        score_path: str,
    ) -> float:
        base = float(min(scores))
        policy = {
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
            **(self.ensemble_gate_policy or {}),
        }
        profile = self.extractor._question_profile(prompt)
        if profile == "count":
            max_delta = float(policy["max_delta_up_count"])
        elif profile in {"who", "when", "where"}:
            max_delta = float(policy["max_delta_up_who_when_where"])
        else:
            max_delta = float(policy["max_delta_up_generic"])

        delta = 0.0
        if audit is not None and audit.ok:
            wrong_fact_risk = max(float(audit.we), float(audit.wn), float(audit.ue), float(audit.bt))
            strong_contradiction = float(audit.x) >= float(policy["contradiction_threshold"])
            strong_wrong_fact = wrong_fact_risk >= float(policy["wrong_fact_threshold"])
            strong_unsupported = max(float(audit.u), float(audit.h)) >= float(policy["unsupported_threshold"])
            if strong_contradiction or strong_wrong_fact:
                max_delta = max(max_delta, float(policy["max_delta_up_strong_audit_risk"]))
                delta = max(delta, max_delta)
            elif (
                bool(policy["allow_unsupported_rescue"])
                and strong_unsupported
                and float(audit.conf) >= float(policy["low_confidence_ceiling"])
            ):
                delta = max(delta, min(max_delta, 0.04))

        text_values = self.extractor.extract_text_only(prompt, answer)
        text_name_to_value = dict(zip(FeatureExtractor.text_feature_names, text_values, strict=False))
        typed_shape = float(text_name_to_value.get("typed_exact_answer_shape", 0.0))
        type_mismatch = float(text_name_to_value.get("question_type_mismatch", 0.0))
        new_number_count = float(text_name_to_value.get("tail_new_number_count", 0.0))
        if score_path == "fallback" and profile in {"who", "when", "where", "count"}:
            if typed_shape >= 0.5 and (type_mismatch >= 0.5 or new_number_count > 0):
                delta = max(delta, min(max_delta, float(policy["fallback_text_delta"])))

        return float(np.clip(base + min(max_delta, max(0.0, delta)), 0.0, 1.0))

    def _capped_delta_score(
        self,
        scores: list[float],
        *,
        prompt: str,
        answer: str,
        audit: AuditPayload | None,
        score_path: str,
    ) -> float:
        base = float(min(scores))
        target = base
        if self.ensemble_gate_model is not None:
            gate_features = self._gate_features(scores, prompt=prompt, answer=answer, audit=audit, score_path=score_path)
            if hasattr(self.ensemble_gate_model, "predict_proba"):
                target = float(self.ensemble_gate_model.predict_proba(gate_features)[0, 1])
            elif hasattr(self.ensemble_gate_model, "predict"):
                target = float(np.asarray(self.ensemble_gate_model.predict(gate_features)).reshape(-1)[0])
        policy = {
            "max_delta_up_generic": 0.08,
            "max_delta_up_count": 0.12,
            "max_delta_up_who_when_where": 0.05,
            "max_delta_down": 0.05,
            "high_score_threshold": 0.65,
            "max_delta_up_high_score": 0.12,
            "max_delta_up_strong_audit_risk": 0.15,
            **(self.ensemble_gate_policy or {}),
        }
        profile = self.extractor._question_profile(prompt)
        if profile == "count":
            max_up = float(policy["max_delta_up_count"])
        elif profile in {"who", "when", "where"}:
            max_up = float(policy["max_delta_up_who_when_where"])
        else:
            max_up = float(policy["max_delta_up_generic"])
        if len(scores) >= 2 and scores[0] >= float(policy["high_score_threshold"]) and scores[1] >= float(policy["high_score_threshold"]):
            max_up = max(max_up, float(policy["max_delta_up_high_score"]))
        if audit is not None and (audit.x >= 0.5 or audit.we >= 0.75 or audit.wn >= 0.75 or audit.u >= 0.85 or audit.h >= 0.85):
            max_up = max(max_up, float(policy["max_delta_up_strong_audit_risk"]))
        delta = target - base
        capped = base + min(max_up, max(-float(policy["max_delta_down"]), delta))
        return float(np.clip(capped, 0.0, 1.0))

    def _gate_features(
        self,
        scores: list[float],
        *,
        prompt: str,
        answer: str,
        audit: AuditPayload | None,
        score_path: str,
    ) -> np.ndarray:
        primary = scores[0]
        secondary = scores[1] if len(scores) > 1 else primary
        score_values = {
            "primary_score": primary,
            "secondary_score": secondary,
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": float(sum(scores) / len(scores)),
            "score_abs_diff": abs(primary - secondary),
            "audit_ok": float(bool(audit and audit.ok)),
            "audit_bad": float(not bool(audit and audit.ok)),
            "score_path_main": float(score_path == "main"),
            "score_path_fallback": float(score_path == "fallback"),
        }
        profile = self.extractor._question_profile(prompt)
        for name in (
            "who",
            "when",
            "where",
            "count",
            "generic",
            "which_list",
            "what_property",
            "by_whom",
            "title_name",
            "definition",
        ):
            score_values[f"question_profile_{name}"] = float(profile == name)
        text_values = self.extractor.extract_text_only(prompt, answer)
        text_name_to_value = dict(zip(FeatureExtractor.text_feature_names, text_values, strict=False))
        for name in (
            "typed_answer_len_bucket",
            "typed_exact_answer_shape",
            "tail_new_number_count",
            "tail_new_entity_count",
            "answer_overexplains_typed_question",
        ):
            score_values[name] = float(text_name_to_value.get(name, 0.0))
        names = self.ensemble_gate_feature_names or list(score_values)
        return np.array([[score_values.get(name, 0.0) for name in names]], dtype=np.float32)

    @staticmethod
    def _load_main_classifier(model_dir: str | Path) -> HallucinationClassifier:
        try:
            return HallucinationClassifier.load(model_dir)
        except FileNotFoundError:
            # Neutral uncalibrated classifier placeholder if training artifacts are missing.
            dummy = HallucinationClassifier(feature_names=FeatureExtractor.api_feature_names + FeatureExtractor.text_feature_names)
            feature_dim = len(FeatureExtractor.api_feature_names) + len(FeatureExtractor.text_feature_names)
            dummy.scaler.fit(np.zeros((2, feature_dim), dtype=np.float32))
            dummy.model.fit(np.zeros((2, feature_dim), dtype=np.float32), np.array([0, 1], dtype=np.int32))
            return dummy

    @staticmethod
    def _load_fallback_classifier(model_dir: str | Path) -> HallucinationClassifier | HeuristicFallbackClassifier:
        try:
            return load_fallback_bundle(model_dir)
        except FileNotFoundError:
            return HeuristicFallbackClassifier()

    @staticmethod
    def _classifier_from_bundle(bundle: dict) -> HallucinationClassifier:
        return HallucinationClassifier(
            feature_names=bundle.get("feature_names", []),
            feature_indices=bundle.get("feature_indices", []),
            scaler=bundle["scaler"],
            model=bundle["model"],
            calibrator=bundle.get("calibrator"),
            calibration_kind=bundle.get("calibration_kind", "none"),
            score_transform=bundle.get("score_transform", bundle.get("calibration_kind", "predict_proba")),
        )

    @classmethod
    def _load_ensemble(
        cls,
        model_dir: str | Path,
    ) -> tuple[list[HallucinationClassifier], list[HallucinationClassifier], str, Any | None, list[str], dict[str, Any]]:
        ensemble_path = Path(model_dir) / "ensemble.joblib"
        if not ensemble_path.exists():
            return [], [], "single", None, [], {}
        bundle = joblib.load(ensemble_path)
        classifiers = [cls._classifier_from_bundle(item) for item in bundle.get("classifiers", [])]
        fallback_classifiers = [cls._classifier_from_bundle(item) for item in bundle.get("fallback_classifiers", [])]
        return (
            classifiers,
            fallback_classifiers,
            str(bundle.get("combiner", "min")),
            bundle.get("gate_model"),
            list(bundle.get("gate_feature_names", [])),
            dict(bundle.get("gate_policy", {})),
        )
