import os

import pytest

from guardian_of_truth.api_client import GroqVerifier


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set")
def test_groq_verifier_returns_valid_payload() -> None:
    verifier = GroqVerifier()
    audit = verifier.verify(
        "В каком году Юрий Гагарин совершил первый полёт человека в космос?",
        "Юрий Гагарин совершил первый полёт человека в космос в 1961 году.",
        mode="runtime",
    )

    assert audit.mode == "runtime"
    assert 0.0 <= audit.h <= 1.0
    assert 0.0 <= audit.r <= 1.0
