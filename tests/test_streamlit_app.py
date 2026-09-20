from __future__ import annotations

from unittest.mock import MagicMock, patch

from guardian_of_truth.guardian import ScoringResult
from guardian_of_truth.streamlit_app import (
    EXAMPLES,
    format_scoring_payload,
    resolve_groq_api_key,
)
from run_project import ROOT_DIR, SRC_DIR


def test_format_scoring_payload_smoke() -> None:
    fake_engine = MagicMock()
    fake_engine.last_score_path = "main"
    fake_engine.last_base_is_hallucination_proba = 0.85
    fake_engine.last_refute_overlay_policy = "bayesian_blind_rescue_policy"
    fake_engine.last_refute_overlay_reason = "refute_active"
    fake_engine.last_refute_overlay_expected_kind = "year"
    fake_engine.last_refute_overlay_delta = 0.05

    result = ScoringResult(
        is_hallucination=True,
        is_hallucination_proba=0.90,
        t_model_sec=0.15,
        t_overhead_sec=0.02,
        t_total_sec=0.17,
    )

    payload = format_scoring_payload(fake_engine, result)

    assert payload["verdict"] == "Hallucination"
    assert payload["is_hallucination"] is True
    assert payload["is_hallucination_proba"] == 0.90
    assert payload["timing"]["t_total_ms"] == 170.0
    assert payload["timing"]["t_model_ms"] == 150.0
    assert payload["pipeline"]["score_path"] == "main"
    assert payload["pipeline"]["overlay_reason"] == "refute_active"
    assert payload["pipeline"]["overlay_delta"] == 0.05


def test_format_scoring_payload_factual() -> None:
    fake_engine = MagicMock()
    fake_engine.last_score_path = "main"
    fake_engine.last_base_is_hallucination_proba = 0.12
    fake_engine.last_refute_overlay_policy = "bayesian_blind_rescue_policy"
    fake_engine.last_refute_overlay_reason = "none"
    fake_engine.last_refute_overlay_expected_kind = "none"
    fake_engine.last_refute_overlay_delta = 0.0

    result = ScoringResult(
        is_hallucination=False,
        is_hallucination_proba=0.12,
        t_model_sec=0.08,
        t_overhead_sec=0.01,
        t_total_sec=0.09,
    )

    payload = format_scoring_payload(fake_engine, result)

    assert payload["verdict"] == "Likely factual"
    assert payload["is_hallucination"] is False
    assert payload["is_hallucination_proba"] == 0.12


def test_examples_structure() -> None:
    assert len(EXAMPLES) >= 3
    for ex in EXAMPLES:
        assert "title" in ex
        assert "prompt" in ex
        assert "answer" in ex
        assert len(ex["prompt"]) > 5
        assert len(ex["answer"]) > 5


def test_resolve_groq_api_key_from_env() -> None:
    with patch.dict("os.environ", {"GROQ_API_KEY": "test_gsk_12345"}):
        key = resolve_groq_api_key()
        assert key == "test_gsk_12345"


def test_run_project_paths_are_repo_relative() -> None:
    assert ROOT_DIR.name == "sber"
    assert SRC_DIR == ROOT_DIR / "src"
