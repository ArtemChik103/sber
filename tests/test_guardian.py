import time

from guardian_of_truth.api_client import AuditPayload
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


def test_guardian_runtime_cutoff_returns_before_slow_api() -> None:
    class SlowVerifier(TimeoutVerifier):
        def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
            time.sleep(1.0)
            return AuditPayload(h=0.9, n=0.2, e=0.1, r=0.9, u=0.1, c=1, x=0, mode="runtime", model_name="llama-3.1-8b-instant")

    guardian = GuardianOfTruth(verifier=SlowVerifier(), fallback_classifier=HeuristicFallbackClassifier())
    started = time.perf_counter()
    result = guardian.score("Кто был первым президентом США?", "Первым президентом США был Джордж Вашингтон.")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert result.t_total_sec < 0.5
    assert 0.0 <= result.is_hallucination_proba <= 1.0
