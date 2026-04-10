from guardian_of_truth.generation import (
    _balanced_candidate_records,
    is_high_quality_supported_positive,
    is_high_quality_targeted_negative,
)


def test_is_high_quality_supported_positive_requires_longer_supported_answer() -> None:
    reference = "Санкт-Петербург был основан в 1703 году."
    supported = "Санкт-Петербург был основан в 1703 году и быстро стал важным портовым городом России."
    unchanged = "Санкт-Петербург был основан в 1703 году."

    assert is_high_quality_supported_positive(reference, supported)
    assert not is_high_quality_supported_positive(reference, unchanged)


def test_balanced_candidate_records_rotates_question_profiles() -> None:
    records = [
        {"prompt": "Кто написал роман?", "answer": "Автором был Иванов.", "label": 0, "variant_type": "positive", "source": "test"},
        {"prompt": "В каком году основан город?", "answer": "Город основан в 1703 году.", "label": 0, "variant_type": "positive", "source": "test"},
        {"prompt": "Где находится башня?", "answer": "Башня находится в Париже.", "label": 0, "variant_type": "positive", "source": "test"},
        {"prompt": "Сколько континентов?", "answer": "Обычно выделяют семь континентов.", "label": 0, "variant_type": "positive", "source": "test"},
        {"prompt": "Как называется столица Франции?", "answer": "Столица Франции — Париж.", "label": 0, "variant_type": "positive", "source": "test"},
    ]

    selected = _balanced_candidate_records(records, limit=5)
    prompts = [record["prompt"] for record in selected]

    assert len(selected) == 5
    assert any(prompt.startswith("Кто") for prompt in prompts)
    assert any(prompt.startswith("В каком году") for prompt in prompts)


def test_is_high_quality_targeted_negative_allows_longer_drift_answer() -> None:
    reference = "Эйфелева башня находится в Париже."
    drift = "Эйфелева башня находится в Лондоне, столице Великобритании, известной своей богатой историей и культурным наследием."

    assert is_high_quality_targeted_negative(reference, drift)
