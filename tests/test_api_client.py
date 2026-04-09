from pathlib import Path

from guardian_of_truth.api_client import ApiSettings, AuditPayload, GroqVerifier
from guardian_of_truth.cache import SQLiteCache


def test_invalid_json_normalizes_to_neutral_failure() -> None:
    audit = AuditPayload.from_response_text(
        "not-json",
        mode="runtime",
        model_name="llama-3.1-8b-instant",
    )

    assert not audit.ok
    assert audit.status == "invalid_json"
    assert audit.r == 0.5


def test_partial_json_is_filled_with_defaults() -> None:
    audit = AuditPayload.from_response_text(
        '{"h": 0.9, "x": 1}',
        mode="runtime",
        model_name="llama-3.1-8b-instant",
    )

    assert audit.ok
    assert audit.status == "partial_json"
    assert audit.h == 0.9
    assert audit.r == 0.5
    assert audit.x == 1.0


def test_cache_prevents_duplicate_network_call(tmp_path: Path) -> None:
    settings = ApiSettings.from_yaml()
    cache = SQLiteCache(tmp_path / "groq_cache.sqlite")
    verifier = GroqVerifier(api_key="test-key", settings=settings, cache=cache)
    calls = {"count": 0}

    async def fake_verify_async(*, prompt: str, answer: str, mode: str, model_name: str) -> AuditPayload:
        calls["count"] += 1
        return AuditPayload(h=0.2, n=0.1, e=0.0, r=0.9, u=0.1, c=1, x=0, mode=mode, model_name=model_name)

    verifier._verify_async = fake_verify_async  # type: ignore[method-assign]

    first = verifier.verify("prompt", "answer", mode="runtime")
    second = verifier.verify("prompt", "answer", mode="runtime")

    assert first.cached is False
    assert second.cached is True
    assert calls["count"] == 1
