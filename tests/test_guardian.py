import time

import joblib
import numpy as np

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.evidence import EvidenceAudit
from guardian_of_truth.guardian import GuardianOfTruth, HeuristicFallbackClassifier


class TimeoutVerifier:
    def __init__(self) -> None:
        self.settings = type("Settings", (), {"total_timeout_sec": 0.45})()

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        return AuditPayload.neutral(status="timeout", mode="runtime", model_name="llama-3.1-8b-instant", ok=False)


def test_guardian_fallback_handles_timeout() -> None:
    guardian = GuardianOfTruth(verifier=TimeoutVerifier(), fallback_classifier=HeuristicFallbackClassifier())
    result = guardian.score("Кто был первым президентом США?", "Первым президентом США был Джордж Вашингтон.")

    assert 0.0 <= result.is_hallucination_proba <= 1.0
    assert result.t_total_sec >= 0.0


def test_guardian_fallback_handles_429() -> None:
    class RateLimitVerifier(TimeoutVerifier):
        def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
            return AuditPayload.neutral(status="http_429", mode="runtime", model_name="llama-3.1-8b-instant", ok=False)

    guardian = GuardianOfTruth(verifier=RateLimitVerifier(), fallback_classifier=HeuristicFallbackClassifier())
    result = guardian.score("Сколько градусов содержит прямой угол?", "Прямой угол содержит 90 градусов.")

    assert 0.0 <= result.is_hallucination_proba <= 1.0


class ConstantClassifier:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        rows = 1 if X.ndim == 1 else len(X)
        return np.full(rows, self.value, dtype=np.float64)


class ConstantSklearnModel(ConstantClassifier):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = super().predict_proba(X)
        return np.column_stack([1.0 - proba, proba])


class IdentityScaler:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


class OkVerifier(TimeoutVerifier):
    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        return AuditPayload(h=0.1, r=0.9, sem=0.9, status="ok", ok=True, model_name="test-model", mode="runtime")


def _bundle(value: float) -> dict:
    return {
        "scaler": IdentityScaler(),
        "model": ConstantSklearnModel(value),
        "calibrator": None,
        "calibration_kind": "none",
        "score_transform": "predict_proba",
        "feature_names": [],
        "feature_indices": [],
    }


def test_guardian_without_ensemble_keeps_single_model_behavior(tmp_path) -> None:
    guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        classifier=ConstantClassifier(0.7),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
    )

    result = guardian.score("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.")

    assert result.is_hallucination_proba == 0.7


class StaticEvidenceRetriever:
    available = True

    def __init__(self, audit: EvidenceAudit) -> None:
        self.audit_value = audit

    def audit(self, prompt: str, answer: str) -> EvidenceAudit:
        return self.audit_value


def _overlay_evidence_audit(**overrides) -> EvidenceAudit:
    values = {
        "status": "ok",
        "retrieval_latency_sec": 0.0,
        "retrieval_hit_count": 1,
        "top_bm25_score": 1.0,
        "top_evidence_overlap": 0.3,
        "core_supported": 0.0,
        "core_refuted": 0.0,
        "entity_supported_ratio": 0.0,
        "entity_refuted_count": 0.0,
        "number_supported_ratio": 0.0,
        "number_refuted_count": 0.0,
        "year_supported_ratio": 0.0,
        "year_refuted_count": 0.0,
        "claim_supported_count": 0.0,
        "claim_refuted_count": 0.0,
        "claim_unknown_count": 0.0,
        "tail_unsupported_entity_count": 0.0,
        "tail_unsupported_number_count": 0.0,
        "evidence_missing_for_typed_question": 0.0,
        "answer_entity_not_in_evidence_ratio": 0.0,
        "answer_number_not_in_evidence_ratio": 0.0,
        "aligned_hit_count": 1.0,
        "aligned_year_refuted_count": 1.0,
        "aligned_number_refuted_count": 0.0,
        "answer_year_in_prompt": 0.0,
        "answer_number_in_prompt": 0.0,
        "aligned_title_entity_match": 1.0,
        "compact_json": "{}",
    }
    values.update(overrides)
    return EvidenceAudit(**values)


def test_guardian_refute_overlay_disabled_keeps_prediction_identical(tmp_path) -> None:
    audit = _overlay_evidence_audit()
    base_guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        classifier=ConstantClassifier(0.3),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
        evidence_retriever=StaticEvidenceRetriever(audit),
        refute_overlay_policy="disabled",
    )

    result = base_guardian.score("В каком году был основан Санкт-Петербург?", "Санкт-Петербург был основан в 1492 году.")

    assert result.is_hallucination_proba == 0.3
    assert base_guardian.last_refute_overlay_delta == 0.0


def test_guardian_refute_overlay_policy_lifts_only_valid_aligned_refute(tmp_path) -> None:
    audit = _overlay_evidence_audit()
    guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        classifier=ConstantClassifier(0.3),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
        evidence_retriever=StaticEvidenceRetriever(audit),
        refute_overlay_policy="v4_cap082",
    )

    result = guardian.score("В каком году был основан Санкт-Петербург?", "Санкт-Петербург был основан в 1492 году.")

    assert result.is_hallucination_proba == 0.82
    assert result.is_hallucination
    assert guardian.last_base_is_hallucination_proba == 0.3
    assert guardian.last_refute_overlay_reason == "aligned_year_refuted"


def test_guardian_refute_overlay_policy_skips_non_year_count_prompt(tmp_path) -> None:
    audit = _overlay_evidence_audit()
    guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        classifier=ConstantClassifier(0.3),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
        evidence_retriever=StaticEvidenceRetriever(audit),
        refute_overlay_policy="v4_cap082",
    )

    result = guardian.score("В каком журнале в 1853 году был опубликован рассказ?", "В журнале Современник.")

    assert result.is_hallucination_proba == 0.3
    assert guardian.last_refute_overlay_reason == "unchanged_expected_none"


def test_guardian_min_ensemble_combines_main_scores(tmp_path) -> None:
    joblib.dump({"combiner": "min", "classifiers": [_bundle(0.25)], "fallback_classifiers": []}, tmp_path / "ensemble.joblib")
    guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        classifier=ConstantClassifier(0.8),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
    )

    result = guardian.score("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.")

    assert result.is_hallucination_proba == 0.25


def test_guardian_min_ensemble_combines_fallback_scores(tmp_path) -> None:
    joblib.dump({"combiner": "min", "classifiers": [], "fallback_classifiers": [_bundle(0.15)]}, tmp_path / "ensemble.joblib")
    guardian = GuardianOfTruth(
        verifier=TimeoutVerifier(),
        classifier=ConstantClassifier(0.8),
        fallback_classifier=ConstantClassifier(0.6),
        model_dir=tmp_path,
    )

    result = guardian.score("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.")

    assert result.is_hallucination_proba == 0.15


def test_guardian_gate_artifact_loads_and_scores(tmp_path) -> None:
    from sklearn.dummy import DummyClassifier

    gate = DummyClassifier(strategy="constant", constant=1)
    gate.fit(np.array([[0.0], [1.0]], dtype=np.float32), np.array([0, 1], dtype=np.int32))
    joblib.dump(
        {
            "combiner": "gate_logreg",
            "classifiers": [_bundle(0.2)],
            "fallback_classifiers": [],
            "gate_model": gate,
            "gate_feature_names": ["primary_score"],
        },
        tmp_path / "ensemble.joblib",
    )
    guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        classifier=ConstantClassifier(0.8),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
    )

    result = guardian.score("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.")

    assert 0.0 <= result.is_hallucination_proba <= 1.0


def test_guardian_capped_delta_gate_cannot_exceed_caps(tmp_path) -> None:
    from sklearn.dummy import DummyClassifier

    gate = DummyClassifier(strategy="constant", constant=1)
    gate.fit(np.array([[0.0], [1.0]], dtype=np.float32), np.array([0, 1], dtype=np.int32))
    joblib.dump(
        {
            "combiner": "capped_delta_gate",
            "classifiers": [_bundle(0.2)],
            "fallback_classifiers": [],
            "gate_model": gate,
            "gate_feature_names": ["primary_score"],
            "gate_policy": {
                "max_delta_up_generic": 0.08,
                "max_delta_up_count": 0.12,
                "max_delta_up_who_when_where": 0.05,
                "max_delta_down": 0.05,
            },
        },
        tmp_path / "ensemble.joblib",
    )
    guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        classifier=ConstantClassifier(0.3),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
    )

    result = guardian.score("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.")

    assert result.is_hallucination_proba <= 0.25
    assert 0.0 <= result.is_hallucination_proba <= 1.0


def test_guardian_rescue_rules_never_decreases_base_and_obeys_caps(tmp_path) -> None:
    class RiskVerifier(TimeoutVerifier):
        def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
            return AuditPayload(
                h=0.9,
                u=0.9,
                x=1.0,
                we=0.9,
                wn=0.0,
                ue=0.0,
                bt=0.0,
                conf=0.9,
                status="ok",
                ok=True,
                model_name="test-model",
                mode="runtime",
            )

    joblib.dump(
        {
            "combiner": "rescue_rules",
            "classifiers": [_bundle(0.2)],
            "fallback_classifiers": [],
            "gate_policy": {
                "max_delta_up_generic": 0.03,
                "max_delta_up_who_when_where": 0.05,
                "max_delta_up_count": 0.08,
                "max_delta_up_strong_audit_risk": 0.10,
            },
        },
        tmp_path / "ensemble.joblib",
    )
    guardian = GuardianOfTruth(
        verifier=RiskVerifier(),
        classifier=ConstantClassifier(0.3),
        fallback_classifier=ConstantClassifier(0.2),
        model_dir=tmp_path,
    )

    result = guardian.score("Who wrote Hamlet?", "Christopher Marlowe.")

    assert result.is_hallucination_proba >= 0.2
    assert result.is_hallucination_proba <= 0.3000001
    assert 0.0 <= result.is_hallucination_proba <= 1.0
