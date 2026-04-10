from guardian_of_truth.gradio_app import build_demo, format_result, run_score
from guardian_of_truth.guardian import ScoringResult


class StubGuardian:
    def score(self, prompt: str, answer: str) -> ScoringResult:
        assert prompt == "Prompt"
        assert answer == "Answer"
        return ScoringResult(
            is_hallucination=True,
            is_hallucination_proba=0.875,
            t_model_sec=0.12,
            t_overhead_sec=0.01,
            t_total_sec=0.13,
        )


def test_format_result_smoke() -> None:
    verdict, payload = format_result(
        ScoringResult(
            is_hallucination=False,
            is_hallucination_proba=0.125,
            t_model_sec=0.2,
            t_overhead_sec=0.03,
            t_total_sec=0.23,
        )
    )

    assert verdict == "Likely factual"
    assert payload["predict_proba"] == 0.125
    assert payload["t_total_sec"] == 0.23


def test_run_score_uses_guardian() -> None:
    verdict, payload = run_score("Prompt", "Answer", guardian=StubGuardian())

    assert verdict == "Hallucination"
    assert payload["predict_proba"] == 0.875


def test_run_score_rejects_blank_input() -> None:
    verdict, payload = run_score("", "Answer", guardian=StubGuardian())

    assert verdict == "Input required"
    assert "error" in payload


def test_build_demo_smoke() -> None:
    demo = build_demo()

    assert demo is not None
