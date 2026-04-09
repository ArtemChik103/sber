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

    assert vector.shape == (14,)
    assert vector.dtype == np.float32
