import json

import pandas as pd
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.classifier import HallucinationClassifier, save_fallback_bundle
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.guardian import GuardianOfTruth
from guardian_of_truth.utils import sha256_hexdigest
from scripts.build_synthetic_validation import build_synthetic_validation
from scripts.build_synthetic_validation_v2 import VARIANTS, build_synthetic_validation_v2
from scripts.build_synthetic_validation_v3 import VARIANTS as V3_VARIANTS, build_synthetic_validation_v3
from scripts.build_synthetic_validation_v4 import AUDIT_FIXTURE_FAMILIES, AUDIT_KEYS, build_synthetic_validation_v4
from scripts.build_synthetic_validation_v5 import (
    AUDIT_FIXTURE_FAMILIES as V5_AUDIT_FIXTURE_FAMILIES,
    NEW_PROFILES as V5_NEW_PROFILES,
    OLD_PROFILES as V5_OLD_PROFILES,
    build_synthetic_validation_v5,
)
from scripts.analysis_taxonomy import primary_taxonomy_label, taxonomy_labels
from scripts.compare_audit_sources import compare_audit_sources, row_key
from scripts.diagnose_fallback_rows import diagnose_fallback_rows
from scripts.diagnose_public_candidate import public_candidate_diagnostics
from scripts.build_rescue_rules_candidate import build_rescue_rules_candidate
from scripts.build_multi_min_ensemble import build_multi_min_ensemble
from scripts.select_audit_robust_candidate import _passes
from scripts.select_audit_robust_candidate_v2 import _passes as _passes_v2
from scripts.train_audit_robust_candidate import build_augmented_rows, split_original_rows, train_audit_robust_candidates
from scripts.train_audit_robust_candidate_v2 import (
    augmentation_plan as augmentation_plan_v2,
    build_augmented_rows as build_augmented_rows_v2,
    split_original_rows as split_original_rows_v2,
    train_audit_robust_candidates_v2,
)
from scripts.validate_candidate_on_hard_v4 import build_fixture_map, hard_v4_key, validate_candidate_on_hard_v4
from scripts.validate_candidate_on_hard_v5 import build_fixture_map as build_fixture_map_v5, hard_v5_key, validate_candidate_on_hard_v5
from scripts.build_guarded_correction_candidate import build_guarded_correction_candidate
from scripts.train_gate import train_gate
from scripts.verify_promotion import verify_promotion
from guardian_of_truth.evaluate import run_evaluation


class OkVerifier:
    def __init__(self) -> None:
        self.settings = type("Settings", (), {"total_timeout_sec": 0.45})()

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        return AuditPayload(h=0.1, r=0.9, sem=0.9, status="ok", ok=True, model_name="test-model", mode="runtime")


def test_synthetic_validation_split_is_deterministic(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"prompt": f"When was city {idx} founded?", "answer": f"{1700 + idx}", "label": idx % 2}, ensure_ascii=False)
            for idx in range(20)
        ),
        encoding="utf-8",
    )
    out_one = tmp_path / "one.jsonl"
    out_two = tmp_path / "two.jsonl"

    first = build_synthetic_validation([source], out_one, min_hard_val=5)
    second = build_synthetic_validation([source], out_two, min_hard_val=5)

    assert first == second
    assert sum(row["split"] == "hard_val" for row in first) >= 5
    assert {"reference_answer", "variant_type", "question_profile", "split", "generation_rule"}.issubset(first[0])


def test_verify_promotion_detects_matching_and_non_matching_scores(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    promoted = tmp_path / "promoted.csv"
    pd.DataFrame({"is_hallucination": [0, 1], "is_hallucination_proba": [0.1, 0.9]}).to_csv(candidate, index=False)
    pd.DataFrame({"is_hallucination": [0, 1], "is_hallucination_proba": [0.1, 0.9]}).to_csv(promoted, index=False)

    summary = verify_promotion(candidate, promoted)

    assert summary["matches_exactly"]
    pd.DataFrame({"is_hallucination": [0, 1], "is_hallucination_proba": [0.1, 0.8]}).to_csv(promoted, index=False)
    summary = verify_promotion(candidate, promoted)
    assert not summary["matches_exactly"]
    assert summary["max_abs_diff"] == pytest.approx(0.1)


def _write_tiny_model(path, *, offset: float) -> None:
    feature_count = len(FeatureExtractor.api_feature_names) + len(FeatureExtractor.text_feature_names)
    X = np.vstack([np.zeros(feature_count), np.ones(feature_count) + offset]).astype(np.float32)
    y = np.array([0, 1], dtype=np.int32)
    classifier = HallucinationClassifier(
        feature_names=FeatureExtractor.api_feature_names + FeatureExtractor.text_feature_names,
        scaler=StandardScaler().fit(X),
        model=LogisticRegression().fit(StandardScaler().fit_transform(X), y),
        score_transform="predict_proba",
    )
    classifier.save(path)
    text_count = len(FeatureExtractor.text_feature_names)
    X_text = np.vstack([np.zeros(text_count), np.ones(text_count) + offset]).astype(np.float32)
    fallback = HallucinationClassifier(
        feature_names=FeatureExtractor.text_feature_names,
        scaler=StandardScaler().fit(X_text),
        model=LogisticRegression().fit(StandardScaler().fit_transform(X_text), y),
        score_transform="predict_proba",
    )
    save_fallback_bundle(fallback, path)


def test_train_tiny_gate_on_synthetic_mini_data(tmp_path) -> None:
    dataset = tmp_path / "gate.jsonl"
    rows = []
    for idx in range(30):
        rows.append(
            {
                "prompt": f"When was city {idx} founded?",
                "model_answer": str(1700 + idx),
                "is_hallucination": idx % 2,
                "variant_type": "mini",
                "question_profile": "when",
                "split": "hard_val" if idx >= 20 else "gate_val",
                "generation_rule": "test",
                "reference_answer": str(1700 + idx),
            }
        )
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_tiny_model(primary, offset=0.0)
    _write_tiny_model(secondary, offset=0.2)

    summary = train_gate(dataset, primary_dir=primary, secondary_dir=secondary, output_dir=tmp_path / "gate_model")

    assert summary["hard_val_rows"] == 10
    assert (tmp_path / "gate_model" / "ensemble.joblib").exists()


def test_synthetic_validation_v2_contains_required_fields_and_profiles(tmp_path) -> None:
    output = tmp_path / "hard_v2.jsonl"

    rows = build_synthetic_validation_v2(output, public_csv=None, copies_per_fact=8)

    required = {
        "prompt",
        "model_answer",
        "is_hallucination",
        "reference_answer",
        "variant_type",
        "question_profile",
        "split",
        "generation_rule",
        "source",
    }
    hard = [row for row in rows if row["split"] == "hard_val"]
    assert required.issubset(rows[0])
    assert {"who", "where", "when", "count", "generic"}.issubset({row["question_profile"] for row in rows})
    assert set(VARIANTS).issubset({row["variant_type"] for row in rows})
    assert hard


def test_taxonomy_labels_are_deterministic() -> None:
    row = {
        "prompt": "Who wrote Hamlet?",
        "model_answer": "Marlowe",
        "is_hallucination": 1,
        "baseline_score": 0.6,
        "candidate_score": 0.3,
        "score_path": "main",
        "audit_h": 0.9,
    }

    assert taxonomy_labels(row) == taxonomy_labels(dict(row))
    assert primary_taxonomy_label(row) == "gate_underraises_wrong"


def test_synthetic_validation_v3_required_fields_profiles_and_public_keys(tmp_path) -> None:
    public = tmp_path / "public.csv"
    pd.DataFrame([{"prompt": "Who discovered radium?", "model_answer": "Marie Curie and Pierre Curie"}]).to_csv(public, index=False)
    output = tmp_path / "hard_v3.jsonl"

    rows = build_synthetic_validation_v3(output, public_csv=public, copies_per_fact=8)
    second = build_synthetic_validation_v3(tmp_path / "hard_v3_second.jsonl", public_csv=public, copies_per_fact=8)

    required = {
        "prompt",
        "model_answer",
        "is_hallucination",
        "reference_answer",
        "variant_type",
        "taxonomy_family",
        "question_profile",
        "split",
        "generation_rule",
        "source",
    }
    public_keys = {row_key("Who discovered radium?", "Marie Curie and Pierre Curie")}
    hard = [row for row in rows if row["split"] == "hard_val"]
    assert rows == second
    assert required.issubset(rows[0])
    assert {"who", "where", "when", "count", "generic"}.issubset({row["question_profile"] for row in rows})
    assert set(V3_VARIANTS).issubset({row["taxonomy_family"] for row in hard})
    assert public_keys.isdisjoint({row_key(row["prompt"], row["model_answer"]) for row in rows})


def test_audit_drift_keying_missing_duplicates_and_tiny_report(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    v5 = tmp_path / "v5.csv"
    pd.DataFrame(
        [
            {"prompt": "q1", "model_answer": "a1", "is_hallucination": 0, "is_hallucination_proba": 0.1, "audit_status": "ok", "audit_h": 0.1},
            {"prompt": "q1", "model_answer": "a1", "is_hallucination": 0, "is_hallucination_proba": 0.2, "audit_status": "ok", "audit_h": 0.2},
            {"prompt": "q2", "model_answer": "a2", "is_hallucination": 1, "is_hallucination_proba": 0.8, "audit_status": "ok", "audit_h": 0.8},
        ]
    ).to_csv(baseline, index=False)
    pd.DataFrame(
        [
            {"prompt": "q1", "model_answer": "a1", "is_hallucination": 0, "is_hallucination_proba": 0.3, "audit_status": "ok", "audit_h": 0.1},
            {"prompt": "q3", "model_answer": "a3", "is_hallucination": 1, "is_hallucination_proba": 0.4, "audit_status": "partial_json", "audit_h": 0.9},
        ]
    ).to_csv(v5, index=False)
    pd.DataFrame(
        [{"prompt": "q1", "model_answer": "a1", "is_hallucination": 0, "is_hallucination_proba": 0.5}]
    ).to_csv(candidate, index=False)

    summary = compare_audit_sources(
        historical_csv=baseline,
        v5_clean_csv=v5,
        candidate_csvs=[candidate],
        output_md=tmp_path / "drift.md",
        output_csv=tmp_path / "drift.csv",
        output_summary=tmp_path / "drift.summary.json",
    )

    assert row_key("q1", "a1") == sha256_hexdigest("q1", "a1")
    assert summary["matched_rows"] == 1
    assert summary["duplicate_keys"]["historical"] == 1
    assert (tmp_path / "drift.md").exists()


def test_fallback_diagnostics_tiny_rows(tmp_path) -> None:
    current = tmp_path / "current.csv"
    fallback = tmp_path / "fallback.csv"
    pd.DataFrame(
        [
            {"prompt": "Who wrote Hamlet?", "model_answer": "Marlowe", "is_hallucination": 1, "is_hallucination_proba": 0.2, "score_path": "fallback", "audit_status": "timeout"},
            {"prompt": "What is water?", "model_answer": "H2O", "is_hallucination": 0, "is_hallucination_proba": 0.1, "score_path": "main", "audit_status": "ok"},
        ]
    ).to_csv(current, index=False)
    pd.DataFrame(
        [
            {"prompt": "Who wrote Hamlet?", "model_answer": "Marlowe", "is_hallucination": 1, "is_hallucination_proba": 0.35},
            {"prompt": "What is water?", "model_answer": "H2O", "is_hallucination": 0, "is_hallucination_proba": 0.1},
        ]
    ).to_csv(fallback, index=False)

    summary = diagnose_fallback_rows(current, fallback, output_md=tmp_path / "fallback.md", output_csv=tmp_path / "fallback_out.csv")

    assert summary["fallback_rows"] == 1
    assert summary["changed_ge_threshold"] == 1


def test_build_tiny_rescue_rule_candidate_loads(tmp_path) -> None:
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_tiny_model(primary, offset=0.0)
    _write_tiny_model(secondary, offset=0.2)

    summary = build_rescue_rules_candidate(primary, secondary, tmp_path / "rescue")
    guardian = GuardianOfTruth(
        verifier=OkVerifier(),
        model_dir=tmp_path / "rescue",
    )
    result = guardian.score("Who wrote Hamlet?", "William Shakespeare.")

    assert summary["combiner"] == "rescue_rules"
    assert 0.0 <= result.is_hallucination_proba <= 1.0


def test_synthetic_validation_v4_required_fields_coverage_and_public_keys(tmp_path) -> None:
    public = tmp_path / "public.csv"
    pd.DataFrame([{"prompt": "Who painted The Starry Night? Answerьте точной формулировкой для краткой справки.", "model_answer": "Vincent van Gogh"}]).to_csv(public, index=False)

    rows = build_synthetic_validation_v4(tmp_path / "hard_v4.jsonl", public_csv=public, copies_per_fact=20)
    second = build_synthetic_validation_v4(tmp_path / "hard_v4_second.jsonl", public_csv=public, copies_per_fact=20)
    hard = [row for row in rows if row["split"] == "hard_val"]
    required = {
        "prompt", "answer", "model_answer", "label", "is_hallucination", "reference_answer", "variant_type",
        "taxonomy_family", "audit_fixture_family", "audit_fixture", "question_profile", "split", "generation_rule",
        "source", "synthetic_id",
    }

    assert rows == second
    assert len(hard) >= 3000
    assert required.issubset(rows[0])
    assert rows[0]["answer"] == rows[0]["model_answer"]
    assert rows[0]["label"] == rows[0]["is_hallucination"]
    assert set(AUDIT_KEYS) == set(rows[0]["audit_fixture"])
    AuditPayload.model_validate(rows[0]["audit_fixture"])
    frame = pd.DataFrame(hard)
    assert frame["question_profile"].value_counts().min() >= 500
    assert frame["audit_fixture_family"].value_counts().min() >= 200
    assert set(AUDIT_FIXTURE_FAMILIES).issubset(set(frame["audit_fixture_family"]))
    public_keys = {row_key("Who painted The Starry Night? Answerьте точной формулировкой для краткой справки.", "Vincent van Gogh")}
    assert public_keys.isdisjoint({row_key(row["prompt"], row["model_answer"]) for row in rows})
    assert not any("[copy_idx]" in row["prompt"] for row in rows)


def test_hard_v4_validator_keying_missing_duplicates_and_tiny_candidate(tmp_path) -> None:
    model_dir = tmp_path / "model"
    _write_tiny_model(model_dir, offset=0.0)
    rows = build_synthetic_validation_v4(tmp_path / "hard_v4.jsonl", public_csv=None, copies_per_fact=1)[:8]
    dataset = tmp_path / "tiny_hard_v4.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    summary = validate_candidate_on_hard_v4(dataset, candidate_dir=model_dir, output_summary=tmp_path / "summary.json")

    assert hard_v4_key(rows[0]["prompt"], rows[0]["model_answer"]) == sha256_hexdigest(rows[0]["prompt"], rows[0]["model_answer"])
    assert summary["rows"] == 8
    assert 0.0 <= summary["score_min"] <= summary["score_max"] <= 1.0
    fixtures, fixture_summary = build_fixture_map(pd.DataFrame(rows + [rows[0]]))
    assert fixtures
    assert fixture_summary["duplicate_keys"] == 1
    broken = dict(rows[0])
    broken.pop("audit_fixture")
    _, missing_summary = build_fixture_map(pd.DataFrame([broken]))
    assert missing_summary["missing_fixtures"] == 1


def test_rejected_rescue_shape_fails_hard_v4_acceptance_rules() -> None:
    baseline = {
        "overall_ap": 0.70,
        "ap_by_profile": {"who": 0.70, "where": 0.70, "when": 0.70},
        "ap_by_audit_fixture_family": {"fallback_bad_status_text_only": 0.70},
        "noisy_correct_mean_score": 0.20,
        "neutral_wrong_mean_score": 0.60,
    }
    candidate = {
        "overall_ap": 0.705,
        "ap_by_profile": {"who": 0.699, "where": 0.70, "when": 0.70},
        "ap_by_audit_fixture_family": {"fallback_bad_status_text_only": 0.70},
        "noisy_correct_mean_score": 0.21,
        "neutral_wrong_mean_score": 0.60,
    }

    accepted, reasons = _passes(candidate, baseline)

    assert not accepted
    assert "overall_ap_delta" in reasons
    assert "typed_regression:who" in reasons


def test_audit_augmentation_is_deterministic_same_split_and_weighted() -> None:
    records = [
        {"prompt": "Who wrote Hamlet?", "answer": "William Shakespeare", "label": 0},
        {"prompt": "Who wrote Hamlet?", "answer": "Christopher Marlowe", "label": 1},
    ]
    audits = [AuditPayload(status="ok", ok=True, model_name="test", mode="dataset") for _ in records]

    first = build_augmented_rows(records, audits)
    second = build_augmented_rows(records, audits)

    assert first[["prompt", "answer", "label", "split", "augmentation", "sample_weight"]].to_dict("records") == second[["prompt", "answer", "label", "split", "augmentation", "sample_weight"]].to_dict("records")
    assert all(group["split"].nunique() == 1 for _, group in first.groupby("original_key"))
    weights = dict(zip(first["augmentation"], first["sample_weight"], strict=False))
    assert weights["original"] == pytest.approx(1.0)
    assert weights["neutral_dropout"] == pytest.approx(0.30)
    assert weights["correct_high_noisy_x0"] == pytest.approx(0.75)
    assert weights["wrong_neutral"] == pytest.approx(0.75)
    assert weights["wrong_strong_contradiction"] == pytest.approx(0.40)
    assert split_original_rows(records) == split_original_rows(records)


def test_tiny_audit_robust_train_multi_min_evaluate_and_diagnostics(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    rows = []
    for idx in range(24):
        rows.append({"prompt": f"Who founded test city {idx}?", "answer": "Alice" if idx % 2 == 0 else "Bob", "label": idx % 2, "variant_type": "tiny"})
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    train_summary = train_audit_robust_candidates(dataset, output_root=tmp_path, disable_api=True)
    robust = tmp_path / "candidate_audit_robust_hybrid_v1"
    primary = tmp_path / "primary"
    _write_tiny_model(primary, offset=0.0)
    build_multi_min_ensemble(primary, robust, tmp_path / "multi")
    guardian = GuardianOfTruth(verifier=OkVerifier(), model_dir=tmp_path / "multi")
    score = guardian.score("Who founded test city 0?", "Alice").is_hallucination_proba

    assert train_summary["augmented_rows"] > train_summary["original_rows"]
    assert 0.0 <= score <= 1.0

    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    pd.DataFrame([
        {"prompt": "Who founded test city 0?", "model_answer": "Alice", "is_hallucination": 0, "is_hallucination_proba": 0.2, "score_path": "main", "audit_status": "ok"},
        {"prompt": "Who founded test city 1?", "model_answer": "Bob", "is_hallucination": 1, "is_hallucination_proba": 0.6, "score_path": "main", "audit_status": "ok"},
    ]).to_csv(baseline_csv, index=False)
    pd.DataFrame([
        {"prompt": "Who founded test city 0?", "model_answer": "Alice", "is_hallucination": 0, "is_hallucination_proba": 0.1, "score_path": "main", "audit_status": "ok"},
        {"prompt": "Who founded test city 1?", "model_answer": "Bob", "is_hallucination": 1, "is_hallucination_proba": 0.8, "score_path": "main", "audit_status": "ok"},
    ]).to_csv(candidate_csv, index=False)
    tiny_public = tmp_path / "tiny_public.csv"
    audit_source = tmp_path / "audit_source.csv"
    pd.DataFrame([
        {"prompt": "Who founded test city 0?", "model_answer": "Alice", "is_hallucination": 0},
        {"prompt": "Who founded test city 1?", "model_answer": "Bob", "is_hallucination": 1},
    ]).to_csv(tiny_public, index=False)
    pd.DataFrame([
        {"prompt": "Who founded test city 0?", "model_answer": "Alice", "audit_status": "ok", "audit_ok": True, "audit_h": 0.1},
        {"prompt": "Who founded test city 1?", "model_answer": "Bob", "audit_status": "ok", "audit_ok": True, "audit_h": 0.8},
    ]).to_csv(audit_source, index=False)
    evaluated = run_evaluation(tiny_public, output_path=tmp_path / "evaluated.csv", model_dir=tmp_path / "multi", audit_source_csv=audit_source, cache_miss_policy="neutral")
    diag = public_candidate_diagnostics(baseline_csv, candidate_csv, output_md=tmp_path / "diag.md")
    assert len(evaluated) == 2
    assert diag["candidate_ap"] >= diag["baseline_ap"]
    assert (tmp_path / "diag.md").exists()


def test_question_profile_classifier_detects_v5_profiles() -> None:
    extractor = FeatureExtractor()

    assert extractor._question_profile("Какие страны входят в Бенилюкс?") == "which_list"
    assert extractor._question_profile("What is the frequency of mains electricity in Europe?") == "what_property"
    assert extractor._question_profile("Кем была написана пьеса?") == "by_whom"
    assert extractor._question_profile("Какое название носит столица Австралии?") == "title_name"
    assert extractor._question_profile("Что такое инерция?") == "definition"


def test_synthetic_validation_v5_required_fields_coverage_and_public_keys(tmp_path) -> None:
    public = tmp_path / "public.csv"
    pd.DataFrame([{"prompt": "q", "model_answer": "a"}]).to_csv(public, index=False)

    rows = build_synthetic_validation_v5(tmp_path / "hard_v5.jsonl", public_csv=public, copies_per_fact=20)
    second = build_synthetic_validation_v5(tmp_path / "hard_v5_second.jsonl", public_csv=public, copies_per_fact=20)
    hard = [row for row in rows if row["split"] == "hard_val"]
    required = {
        "prompt", "answer", "model_answer", "label", "is_hallucination", "reference_answer", "variant_type",
        "taxonomy_family", "audit_fixture_family", "audit_fixture", "question_profile", "split", "generation_rule",
        "source", "synthetic_id",
    }

    assert rows == second
    assert len(hard) >= 6000
    assert required.issubset(rows[0])
    assert rows[0]["answer"] == rows[0]["model_answer"]
    assert rows[0]["label"] == rows[0]["is_hallucination"]
    assert set(AUDIT_KEYS) == set(rows[0]["audit_fixture"])
    AuditPayload.model_validate(rows[0]["audit_fixture"])
    frame = pd.DataFrame(hard)
    profile_counts = frame["question_profile"].value_counts()
    for profile in V5_OLD_PROFILES:
        assert profile_counts[profile] >= 700
    for profile in V5_NEW_PROFILES:
        assert profile_counts[profile] >= 400
    assert frame["audit_fixture_family"].value_counts().min() >= 250
    assert set(V5_AUDIT_FIXTURE_FAMILIES).issubset(set(frame["audit_fixture_family"]))
    public_keys = {row_key("q", "a")}
    assert public_keys.isdisjoint({row_key(row["prompt"], row["model_answer"]) for row in rows})
    assert not any("[copy_idx]" in row["prompt"] for row in rows)


def test_hard_v5_validator_keying_missing_duplicates_and_tiny_candidate(tmp_path) -> None:
    model_dir = tmp_path / "model"
    _write_tiny_model(model_dir, offset=0.0)
    rows = build_synthetic_validation_v5(tmp_path / "hard_v5.jsonl", public_csv=None, copies_per_fact=1)[:12]
    dataset = tmp_path / "tiny_hard_v5.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    summary = validate_candidate_on_hard_v5(dataset, candidate_dir=model_dir, output_summary=tmp_path / "summary.json")

    assert hard_v5_key(rows[0]["prompt"], rows[0]["model_answer"]) == sha256_hexdigest(rows[0]["prompt"], rows[0]["model_answer"])
    assert summary["rows"] == 12
    assert 0.0 <= summary["score_min"] <= summary["score_max"] <= 1.0
    fixtures, fixture_summary = build_fixture_map_v5(pd.DataFrame(rows + [rows[0]]))
    assert fixtures
    assert fixture_summary["duplicate_keys"] == 1
    broken = dict(rows[0])
    broken.pop("audit_fixture")
    _, missing_summary = build_fixture_map_v5(pd.DataFrame([broken]))
    assert missing_summary["missing_fixtures"] == 1


def test_rejected_shapes_fail_hard_v5_acceptance_rules() -> None:
    baseline = {
        "overall_ap": 0.70,
        "ap_by_profile": {profile: 0.70 for profile in set(V5_OLD_PROFILES) | set(V5_NEW_PROFILES)},
        "ap_by_audit_fixture_family": {"fallback_bad_status_text_only": 0.70, "public_replay_long_clean_audit": 0.70},
        "clean_audit_correct_mean_score": 0.20,
        "clean_audit_wrong_mean_score": 0.60,
        "noisy_correct_mean_score": 0.20,
        "neutral_wrong_mean_score": 0.60,
    }
    candidate = {
        "overall_ap": 0.705,
        "ap_by_profile": {**baseline["ap_by_profile"], "who": 0.699, "which_list": 0.690},
        "ap_by_audit_fixture_family": {"fallback_bad_status_text_only": 0.70, "public_replay_long_clean_audit": 0.697},
        "clean_audit_correct_mean_score": 0.206,
        "clean_audit_wrong_mean_score": 0.594,
        "noisy_correct_mean_score": 0.21,
        "neutral_wrong_mean_score": 0.59,
    }

    accepted, reasons = _passes_v2(candidate, baseline)

    assert not accepted
    assert "overall_ap_delta" in reasons
    assert "typed_regression:who" in reasons
    assert "new_profile_regression:which_list" in reasons
    assert "public_replay_long_clean_audit_regression" in reasons


def test_audit_augmentation_v2_is_deterministic_same_split_and_weighted() -> None:
    records = [
        {"prompt": "Who wrote Hamlet?", "answer": "William Shakespeare", "label": 0},
        {"prompt": "Who wrote Hamlet?", "answer": "Christopher Marlowe", "label": 1},
    ]
    audits = [AuditPayload(status="ok", ok=True, model_name="test", mode="dataset") for _ in records]

    first = build_augmented_rows_v2(records, audits)
    second = build_augmented_rows_v2(records, audits)

    assert first[["prompt", "answer", "label", "split", "augmentation", "sample_weight"]].to_dict("records") == second[["prompt", "answer", "label", "split", "augmentation", "sample_weight"]].to_dict("records")
    assert all(group["split"].nunique() == 1 for _, group in first.groupby("original_key"))
    weights = {name: weight for label in (0, 1) for name, weight in augmentation_plan_v2(label)}
    assert weights["original"] == pytest.approx(1.50)
    assert weights["neutral_dropout"] == pytest.approx(0.20)
    assert weights["correct_high_noisy_x0"] == pytest.approx(0.45)
    assert weights["wrong_neutral"] == pytest.approx(0.55)
    assert weights["wrong_strong_contradiction"] == pytest.approx(0.25)
    assert weights["clean_audit_long"] == pytest.approx(0.60)
    assert split_original_rows_v2(records) == split_original_rows_v2(records)


def test_tiny_audit_robust_v2_guarded_train_evaluate_and_diagnostics(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    rows = []
    for idx in range(24):
        rows.append({"prompt": f"Who founded test city {idx}?", "answer": "Alice" if idx % 2 == 0 else "Bob", "label": idx % 2, "variant_type": "tiny"})
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    train_summary = train_audit_robust_candidates_v2(dataset, output_root=tmp_path, disable_api=True)
    robust = tmp_path / "candidate_audit_robust_hybrid_v2"
    primary = tmp_path / "primary"
    _write_tiny_model(primary, offset=0.0)
    guarded_summary = build_guarded_correction_candidate(primary, robust, tmp_path / "guarded")
    guardian = GuardianOfTruth(verifier=OkVerifier(), model_dir=tmp_path / "guarded")
    score = guardian.score("Who founded test city 0?", "Alice").is_hallucination_proba

    assert train_summary["augmented_rows"] > train_summary["original_rows"]
    assert guarded_summary["combiner"] == "guarded_correction"
    assert 0.0 <= score <= 1.0

    tiny_public = tmp_path / "tiny_public.csv"
    audit_source = tmp_path / "audit_source.csv"
    pd.DataFrame([
        {"prompt": "Who founded test city 0?", "model_answer": "Alice", "is_hallucination": 0},
        {"prompt": "Who founded test city 1?", "model_answer": "Bob", "is_hallucination": 1},
    ]).to_csv(tiny_public, index=False)
    pd.DataFrame([
        {"prompt": "Who founded test city 0?", "model_answer": "Alice", "audit_status": "ok", "audit_ok": True, "audit_h": 0.1},
        {"prompt": "Who founded test city 1?", "model_answer": "Bob", "audit_status": "ok", "audit_ok": True, "audit_h": 0.8},
    ]).to_csv(audit_source, index=False)
    evaluated = run_evaluation(tiny_public, output_path=tmp_path / "evaluated.csv", model_dir=tmp_path / "guarded", audit_source_csv=audit_source, cache_miss_policy="neutral")
    assert len(evaluated) == 2
