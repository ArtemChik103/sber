import numpy as np

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.classifier import HallucinationClassifier
from guardian_of_truth.feature_extractor import FeatureExtractor


def test_classifier_supports_score_transform() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    classifier = HallucinationClassifier(feature_names=["a", "b"], score_transform="raw_margin_sigmoid")

    classifier.fit(X, y, X, y, calibration="none", score_transform="raw_margin_sigmoid")
    proba = classifier.predict_proba(X)

    assert proba.shape == (4,)
    assert np.all((0.0 <= proba) & (proba <= 1.0))


def test_classifier_selects_features_by_name_for_old_bundles() -> None:
    extractor = FeatureExtractor()
    full_vector = extractor.extract(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1703 году.",
        AuditPayload(h=0.1, r=0.9, sem=0.8),
    )
    old_feature_names = [
        "h",
        "n",
        "e",
        "inv_r",
        "u",
        "c",
        "x",
        "inv_q",
        "inv_s",
        "m",
        "answer_len_words",
        "prompt_overlap_ratio",
    ]
    classifier = HallucinationClassifier(feature_names=old_feature_names, feature_indices=list(range(len(old_feature_names))))

    selected = classifier._select_features(full_vector)

    assert selected.shape == (len(old_feature_names),)
    assert selected[-2] == full_vector[FeatureExtractor.api_feature_names.__len__()]
