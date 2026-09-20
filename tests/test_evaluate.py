import pandas as pd
import pytest

from guardian_of_truth.evaluate import CacheOnlyRuntimeVerifier, SourceCsvRuntimeVerifier, _evidence_columns, _prepare_frame, run_evaluation
from guardian_of_truth.evidence import neutral_evidence_audit


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


def test_cache_only_runtime_verifier_marks_cache_miss(tmp_path) -> None:
    from guardian_of_truth.api_client import ApiSettings
    from guardian_of_truth.cache import SQLiteCache

    verifier = CacheOnlyRuntimeVerifier(
        api_key="test-key",
        settings=ApiSettings.from_yaml(),
        cache=SQLiteCache(tmp_path / "cache.sqlite"),
    )

    audit = verifier.verify("question", "answer")

    assert audit.status == "cache_miss"
    assert verifier.last_audit_status == "cache_miss"
    assert not audit.ok


def test_cache_only_runtime_verifier_can_use_neutral_cache_miss(tmp_path) -> None:
    from guardian_of_truth.api_client import ApiSettings
    from guardian_of_truth.cache import SQLiteCache

    verifier = CacheOnlyRuntimeVerifier(
        api_key="test-key",
        settings=ApiSettings.from_yaml(),
        cache=SQLiteCache(tmp_path / "cache.sqlite"),
        cache_miss_policy="neutral",
    )

    audit = verifier.verify("question", "answer")

    assert audit.status == "cache_miss"
    assert audit.ok


def test_run_evaluation_writes_audit_columns_from_source_csv(tmp_path) -> None:
    source = tmp_path / "source.csv"
    audit_source = tmp_path / "audit_source.csv"
    output = tmp_path / "scored.csv"
    pd.DataFrame(
        [
            {
                "prompt": "When was the city founded?",
                "model_answer": "1703",
                "is_hallucination": 0,
            }
        ]
    ).to_csv(source, index=False)
    pd.DataFrame(
        [
            {
                "prompt": "When was the city founded?",
                "model_answer": "1703",
                "audit_h": 0.1,
                "audit_n": 0.0,
                "audit_e": 0.0,
                "audit_r": 0.9,
                "audit_u": 0.0,
                "audit_c": 1,
                "audit_x": 0,
                "audit_q": 0.8,
                "audit_s": 0.8,
                "audit_m": 0.0,
                "audit_sem": 0.9,
                "audit_we": 0.0,
                "audit_wn": 0.0,
                "audit_ue": 0.0,
                "audit_bt": 0.0,
                "audit_conf": 0.9,
                "audit_status": "ok",
                "audit_ok": True,
                "audit_cached": False,
                "audit_model_name": "test-model",
                "audit_mode": "runtime",
                "audit_source": "live",
            }
        ]
    ).to_csv(audit_source, index=False)

    frame = run_evaluation(source, output_path=output, audit_source_csv=audit_source)

    scored = pd.read_csv(output)
    assert frame.loc[0, "audit_h"] == pytest.approx(0.1)
    assert scored.loc[0, "audit_status"] == "ok"
    assert scored.loc[0, "audit_source"] == "source_csv"
    assert scored.loc[0, "audit_model_name"] == "test-model"


def test_audit_source_csv_verifier_restores_payload_without_live_call(tmp_path) -> None:
    from guardian_of_truth.api_client import ApiSettings

    audit_source = tmp_path / "audit_source.csv"
    pd.DataFrame(
        [
            {
                "prompt": "question",
                "model_answer": "answer",
                "audit_h": 0.7,
                "audit_n": 0.2,
                "audit_e": 0.1,
                "audit_r": 0.3,
                "audit_u": 0.4,
                "audit_c": 2,
                "audit_x": 1,
                "audit_q": 0.2,
                "audit_s": 0.3,
                "audit_m": 0.4,
                "audit_sem": 0.5,
                "audit_we": 0.6,
                "audit_wn": 0.7,
                "audit_ue": 0.8,
                "audit_bt": 0.9,
                "audit_conf": 0.1,
                "audit_status": "partial_json",
                "audit_ok": True,
                "audit_cached": False,
                "audit_model_name": "csv-model",
                "audit_mode": "runtime",
            }
        ]
    ).to_csv(audit_source, index=False)
    verifier = SourceCsvRuntimeVerifier(
        audit_source_csv=audit_source,
        api_key=None,
        settings=ApiSettings.from_yaml(),
    )

    audit = verifier.verify("question", "answer")

    assert audit.h == pytest.approx(0.7)
    assert audit.status == "partial_json"
    assert audit.model_name == "csv-model"
    assert verifier.last_audit_source == "source_csv"


def test_audit_source_csv_missing_row_marks_cache_miss(tmp_path) -> None:
    from guardian_of_truth.api_client import ApiSettings

    audit_source = tmp_path / "audit_source.csv"
    pd.DataFrame(
        [
            {
                "prompt": "other question",
                "model_answer": "other answer",
                "audit_status": "ok",
            }
        ]
    ).to_csv(audit_source, index=False)
    verifier = SourceCsvRuntimeVerifier(
        audit_source_csv=audit_source,
        api_key=None,
        settings=ApiSettings.from_yaml(),
    )

    audit = verifier.verify("question", "answer")

    assert audit.status == "cache_miss"
    assert not audit.ok
    assert verifier.last_audit_source == "cache_miss"


def test_guardian_score_public_contract_is_unchanged() -> None:
    import inspect
    import typing

    from guardian_of_truth.guardian import GuardianOfTruth, ScoringResult

    signature = inspect.signature(GuardianOfTruth.score)

    assert list(signature.parameters) == ["self", "prompt", "answer"]
    assert typing.get_type_hints(GuardianOfTruth.score)["return"] is ScoringResult


def test_audit_source_csv_restores_prefill_metadata(tmp_path) -> None:
    from guardian_of_truth.api_client import ApiSettings

    audit_source = tmp_path / "audit_source.csv"
    pd.DataFrame(
        [
            {
                "prompt": "question",
                "model_answer": "answer",
                "audit_status": "ok",
                "audit_ok": True,
                "audit_latency_sec": 1.25,
                "audit_would_timeout": False,
                "audit_attempt_count": 2,
                "audit_rate_profile": "slow-safe",
            }
        ]
    ).to_csv(audit_source, index=False)
    verifier = SourceCsvRuntimeVerifier(
        audit_source_csv=audit_source,
        api_key=None,
        settings=ApiSettings.from_yaml(),
    )

    audit = verifier.verify("question", "answer")

    assert audit.status == "ok"
    assert verifier.last_audit_latency_sec == pytest.approx(1.25)
    assert verifier.last_source_audit_would_timeout is False


def test_runtime_prompt_version_override_changes_verifier_namespace(tmp_path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "scored.csv"
    pd.DataFrame([{"prompt": "question", "model_answer": "answer", "is_hallucination": 0}]).to_csv(source, index=False)

    frame = run_evaluation(
        source,
        output_path=output,
        cache_only_api=True,
        cache_miss_policy="neutral",
        runtime_prompt_version="groq-verifier-v5-exact-tail",
    )

    assert frame.loc[0, "audit_status"] == "cache_miss"


def test_run_evaluation_disabled_overlay_writes_zero_delta(tmp_path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "scored.csv"
    pd.DataFrame([{"prompt": "question", "model_answer": "answer", "is_hallucination": 0}]).to_csv(source, index=False)

    frame = run_evaluation(source, output_path=output, cache_only_api=True, cache_miss_policy="neutral")

    assert frame.loc[0, "refute_overlay_policy"] == "disabled"
    assert frame.loc[0, "refute_overlay_reason"] == "disabled"
    assert frame.loc[0, "refute_overlay_delta"] == pytest.approx(0.0)
    assert frame.loc[0, "base_is_hallucination_proba"] == pytest.approx(frame.loc[0, "is_hallucination_proba"])


def test_run_evaluation_enabled_overlay_writes_columns(tmp_path) -> None:
    from guardian_of_truth.evidence import build_kb

    source = tmp_path / "source.csv"
    output = tmp_path / "scored.csv"
    kb_source = tmp_path / "kb.jsonl"
    db = tmp_path / "evidence.sqlite"
    pd.DataFrame(
        [
            {
                "prompt": "В каком году был основан Санкт-Петербург?",
                "model_answer": "Санкт-Петербург был основан в 1492 году.",
                "is_hallucination": 1,
            }
        ]
    ).to_csv(source, index=False)
    kb_source.write_text(
        '{"title":"Санкт-Петербург","text":"Санкт-Петербург был основан Петром I в 1703 году. Санкт-Петербург является городом федерального значения.","source":"wiki"}\n',
        encoding="utf-8",
    )
    build_kb([kb_source], db, min_chars=80)

    frame = run_evaluation(
        source,
        output_path=output,
        cache_only_api=True,
        cache_miss_policy="neutral",
        evidence_db_path=db,
        refute_overlay_policy="v4_cap082",
    )

    assert frame.loc[0, "refute_overlay_policy"] == "v4_cap082"
    assert "refute_overlay_reason" in frame.columns
    assert "refute_overlay_expected_kind" in frame.columns
    assert frame.loc[0, "refute_overlay_delta"] >= 0.0


def test_evidence_columns_emit_v6_tail_gap_fields() -> None:
    audit = neutral_evidence_audit("ok")
    values = _evidence_columns(audit)
    missing = _evidence_columns(None)

    assert values["evidence_tail_unsupported_entity_count"] == 0.0
    assert values["evidence_tail_unsupported_number_count"] == 0.0
    assert values["evidence_answer_entity_not_in_evidence_ratio"] == 0.0
    assert values["evidence_answer_number_not_in_evidence_ratio"] == 0.0
    assert missing["evidence_status"] == "missing"
    assert missing["evidence_tail_unsupported_entity_count"] is None
