from guardian_of_truth.external_data import _choose_primary_answer, _infer_answer_type, _parse_possible_answers


def test_parse_possible_answers() -> None:
    answers = _parse_possible_answers('["journalist", "journo"]')
    assert answers == ["journalist", "journo"]


def test_choose_primary_answer_prefers_short_reasonable_value() -> None:
    answer = _choose_primary_answer(["this is a fairly short answer", "journalist"])
    assert answer == "this is a fairly short answer"


def test_infer_answer_type() -> None:
    assert _infer_answer_type("1984", "date of birth") == "year"
    assert _infer_answer_type("3.14", "height") == "number"
    assert _infer_answer_type("Paris", "capital") == "place"
    assert _infer_answer_type("journalist", "occupation") == "entity"
