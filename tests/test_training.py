import numpy as np
import pandas as pd

from guardian_of_truth.training import build_variant_stratify_labels, compute_variant_weights
from train import _feature_set_indices


def test_build_variant_stratify_labels_falls_back_for_rare_groups() -> None:
    meta = pd.DataFrame({"variant_type": ["positive", "positive", "groq_negative", "rare_variant"]})
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    labels = build_variant_stratify_labels(y, meta)

    assert labels[-1] == "1"
    assert labels[0] == "positive__0"


def test_compute_variant_weights_normalizes_mean_to_one() -> None:
    meta = pd.DataFrame({"variant_type": ["groq_drift_negative", "positive", "groq_supported_positive"]})

    weights = compute_variant_weights(meta)

    assert weights.shape == (3,)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] > weights[1]
    assert weights[2] > 0.0


def test_feature_set_ablation_indices_are_named() -> None:
    feature_sets = _feature_set_indices()

    assert "old_full_14" in feature_sets
    assert len(feature_sets["old_full_14"]) == 14
    assert feature_sets["text_only"]
    assert feature_sets["hybrid_old_new"] == feature_sets["full"]
