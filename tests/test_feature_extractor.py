import numpy as np

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.feature_extractor import FeatureExtractor


def test_feature_extractor_returns_fixed_shape() -> None:
    extractor = FeatureExtractor()
    audit = AuditPayload(h=0.9, n=0.2, e=0.1, r=0.8, u=0.4, c=2, x=1)
    vector = extractor.extract(
        prompt="В каком году был основан Санкт-Петербург?",
        answer="Санкт-Петербург был основан в 1703 году.",
        audit=audit,
    )

    expected_shape = len(FeatureExtractor.api_feature_names) + len(FeatureExtractor.text_feature_names)
    assert vector.shape == (expected_shape,)
    assert vector.dtype == np.float32


def test_feature_extractor_marks_question_type_mismatch() -> None:
    extractor = FeatureExtractor()
    mismatch = extractor.extract_text_only(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан Петром I.",
    )
    aligned = extractor.extract_text_only(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1703 году.",
    )

    mismatch_idx = FeatureExtractor.text_feature_names.index("question_type_mismatch")
    assert mismatch[mismatch_idx] > aligned[mismatch_idx]


def test_feature_extractor_marks_answer_drift_from_prompt() -> None:
    extractor = FeatureExtractor()
    concise = extractor.extract_text_only(
        "Кто написал роман «Преступление и наказание»?",
        "Роман «Преступление и наказание» написал Фёдор Достоевский.",
    )
    confabulated = extractor.extract_text_only(
        "Кто написал роман «Преступление и наказание»?",
        "Роман «Преступление и наказание» написал Фёдор Достоевский, который родился в 1821 году и умер в Париже.",
    )

    drift_idx = FeatureExtractor.text_feature_names.index("answer_new_content_ratio")
    assert confabulated[drift_idx] > concise[drift_idx]
