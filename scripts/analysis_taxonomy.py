from __future__ import annotations

from typing import Any

import pandas as pd

from guardian_of_truth.evaluate import _question_profile


TAXONOMY_LABELS = [
    "gate_overraises_correct",
    "gate_underraises_wrong",
    "fallback_overraises_correct",
    "fallback_misses_wrong",
    "long_supported_correct_penalized",
    "typed_short_wrong_missed",
    "where_who_when_regression",
    "audit_strong_risk_but_low_score",
    "audit_low_risk_but_high_score",
]


def answer_length_bucket(answer: Any) -> str:
    words = len(str(answer or "").split())
    if words <= 5:
        return "short"
    if words <= 20:
        return "medium"
    if words <= 60:
        return "long"
    return "very_long"


def taxonomy_labels(row: pd.Series | dict[str, Any]) -> list[str]:
    get = row.get if isinstance(row, dict) else row.get
    label = int(float(get("is_hallucination", 0) or 0))
    profile = str(get("profile") or get("question_profile") or _question_profile(str(get("prompt", ""))))
    score_path = str(get("score_path", "main"))
    baseline = float(get("baseline_score", get("base_score", get("score_min", get("is_hallucination_proba", 0.0)))) or 0.0)
    candidate = float(get("candidate_score", get("is_hallucination_proba", baseline)) or 0.0)
    delta = candidate - baseline
    bucket = str(get("answer_length_bucket") or answer_length_bucket(get("model_answer", "")))
    audit_risk = max(
        float(get("audit_h", 0.0) or 0.0),
        float(get("audit_u", 0.0) or 0.0),
        float(get("audit_x", 0.0) or 0.0),
        float(get("audit_we", 0.0) or 0.0),
        float(get("audit_wn", 0.0) or 0.0),
        float(get("audit_ue", 0.0) or 0.0),
        float(get("audit_bt", 0.0) or 0.0),
    )
    labels: list[str] = []
    if label == 0 and score_path == "main" and delta >= 0.03:
        labels.append("gate_overraises_correct")
    if label == 1 and score_path == "main" and delta <= -0.03:
        labels.append("gate_underraises_wrong")
    if label == 0 and score_path == "fallback" and delta >= 0.03:
        labels.append("fallback_overraises_correct")
    if label == 1 and score_path == "fallback" and candidate < 0.45:
        labels.append("fallback_misses_wrong")
    if label == 0 and bucket in {"long", "very_long"} and candidate >= 0.50:
        labels.append("long_supported_correct_penalized")
    if label == 1 and profile in {"who", "where", "when", "count"} and bucket == "short" and candidate < 0.45:
        labels.append("typed_short_wrong_missed")
    if profile in {"who", "where", "when"} and ((label == 0 and delta >= 0.03) or (label == 1 and delta <= -0.03)):
        labels.append("where_who_when_regression")
    if audit_risk >= 0.75 and candidate < 0.45:
        labels.append("audit_strong_risk_but_low_score")
    if audit_risk <= 0.25 and candidate >= 0.65:
        labels.append("audit_low_risk_but_high_score")
    return labels or ["none"]


def primary_taxonomy_label(row: pd.Series | dict[str, Any]) -> str:
    return taxonomy_labels(row)[0]
