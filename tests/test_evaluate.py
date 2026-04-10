import pandas as pd

from guardian_of_truth.evaluate import _prepare_frame


def test_prepare_frame_stable_dev_slice_is_deterministic_and_stratified() -> None:
    frame = pd.DataFrame(
        [
            {"prompt": f"q{i}", "model_answer": f"a{i}", "is_hallucination": i % 2}
            for i in range(10)
        ]
    )

    slice_one = _prepare_frame(frame, dev_slice_size=4)
    slice_two = _prepare_frame(frame, dev_slice_size=4)

    assert slice_one.equals(slice_two)
    assert len(slice_one) == 4
    assert slice_one["is_hallucination"].value_counts().to_dict() == {0: 2, 1: 2}


def test_prepare_frame_typed_slice_is_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {"prompt": "Кто написал роман?", "model_answer": "Фёдор Достоевский", "is_hallucination": 0},
            {"prompt": "В каком году основан город?", "model_answer": "1703", "is_hallucination": 0},
            {"prompt": "Где находится башня?", "model_answer": "В Париже", "is_hallucination": 0},
            {"prompt": "Сколько континентов?", "model_answer": "7", "is_hallucination": 1},
            {"prompt": "Объясни смысл романа", "model_answer": "Это роман о...", "is_hallucination": 1},
            {"prompt": "Что такое фотосинтез?", "model_answer": "Это процесс...", "is_hallucination": 0},
        ]
    )

    slice_one = _prepare_frame(frame, dev_slice_size=4, slice_name="typed")
    slice_two = _prepare_frame(frame, dev_slice_size=4, slice_name="typed")

    assert slice_one.equals(slice_two)
    assert len(slice_one) == 4
    assert slice_one["prompt"].str.lower().str.contains("кто|в каком году|где|сколько").sum() >= 3
