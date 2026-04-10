from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.preprocess import build_feature_matrix, build_text_only_matrix
from guardian_of_truth.utils import write_jsonl


def test_limit_uses_stable_stratified_subset(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    records = [
        {"prompt": f"p{i}", "answer": f"a{i}", "label": i % 2, "variant_type": variant, "source": "test"}
        for i, variant in enumerate(
            [
                "positive",
                "rule_negative",
                "groq_negative",
                "popqa_positive",
                "popqa_negative",
                "fever_refutes",
            ]
        )
    ]
    write_jsonl(dataset_path, records)
    extractor = FeatureExtractor()

    X, y, meta = build_feature_matrix(dataset_path, verifier=None, extractor=extractor, use_api=False, limit=4)
    X_text, y_text = build_text_only_matrix(dataset_path, extractor, limit=4)
    _, _, meta_repeat = build_feature_matrix(dataset_path, verifier=None, extractor=extractor, use_api=False, limit=4)

    assert X.shape == (4, 14)
    assert X_text.shape == (4, 7)
    assert len(set(meta["variant_type"])) == 4
    assert meta["variant_type"].tolist() == meta_repeat["variant_type"].tolist()
    assert y.tolist() == y_text.tolist()


def test_preprocess_filters_low_quality_groq_negative(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    write_jsonl(
        dataset_path,
        [
            {"prompt": "Как называется столица Албания?", "answer": "Столица Албания — Тирана.", "label": 0, "variant_type": "positive", "source": "test"},
            {"prompt": "Как называется столица Албания?", "answer": "Столица Албании — Тирана.", "label": 1, "variant_type": "groq_negative", "source": "test"},
            {"prompt": "Как называется столица Франции?", "answer": "Столица Франции — Париж.", "label": 0, "variant_type": "positive", "source": "test"},
            {"prompt": "Как называется столица Франции?", "answer": "Столица Франции — Руан.", "label": 1, "variant_type": "groq_negative", "source": "test"},
        ],
    )

    extractor = FeatureExtractor()
    _, y, meta = build_feature_matrix(dataset_path, verifier=None, extractor=extractor, use_api=False)

    assert len(meta) == 3
    assert meta["answer"].tolist().count("Столица Албании — Тирана.") == 0
    assert y.tolist() == [0, 0, 1]
