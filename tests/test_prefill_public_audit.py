import pandas as pd

from scripts.prefill_public_audit import DEFAULT_BAD_STATUSES, _good_statuses, _merge_base_rows, _merge_existing_output_rows, _row_key


def test_prefill_retry_selects_only_bad_statuses() -> None:
    frame = pd.DataFrame(
        [
            {"prompt": "q1", "model_answer": "a1", "audit_status": "ok"},
            {"prompt": "q2", "model_answer": "a2", "audit_status": "partial_json"},
            {"prompt": "q3", "model_answer": "a3", "audit_status": "http_429"},
            {"prompt": "q4", "model_answer": "a4", "audit_status": "timeout"},
        ]
    )

    good = _good_statuses(frame, set(DEFAULT_BAD_STATUSES))

    assert good.tolist() == [True, True, False, False]


def test_prefill_output_keys_are_unique() -> None:
    frame = pd.DataFrame(
        [
            {"prompt": "q1", "model_answer": "a1"},
            {"prompt": "q2", "model_answer": "a2"},
        ]
    )
    keys = frame.apply(_row_key, axis=1)

    assert keys.nunique() == len(frame)


def test_prefill_resume_prefers_existing_output_rows() -> None:
    base = pd.DataFrame([{"prompt": "q1", "model_answer": "a1", "audit_status": "ok", "audit_h": 0.1}])
    existing = pd.DataFrame([{"prompt": "q1", "model_answer": "a1", "audit_status": "ok", "audit_h": 0.8}])

    rows = _merge_base_rows(pd.DataFrame(), base, set(DEFAULT_BAD_STATUSES))
    rows.update(_merge_existing_output_rows(existing))

    assert next(iter(rows.values()))["audit_h"] == 0.8
