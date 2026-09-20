from __future__ import annotations

import re
from typing import Any

import pandas as pd

from guardian_of_truth.claims import content_tokens, split_sentences, strip_markup
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.refute_overlay import expected_answer_kind


DRIFT_OVERLAY_CAPS = (0.55, 0.60, 0.65)
ALLOWED_DRIFT_PROFILES = {"generic", "what_property"}
WEAK_AUDIT_STATUSES = {"http_429", "timeout"}
MIN_RETRIEVAL_HITS = 3

DRIFT_DIAGNOSTIC_COLUMNS = [
    "answer_word_count",
    "answer_sentence_count",
    "answer_tail_word_count",
    "answer_tail_ratio",
    "drift_low_overlap",
    "drift_weak_audit_path",
    "drift_long_answer_shape",
    "drift_tail_unsupported",
    "drift_evidence_gap",
]


def answer_shape(prompt: str, answer: str) -> dict[str, float]:
    clean = strip_markup(answer)
    sentences = split_sentences(clean)
    words = re.findall(r"\b[\w\-]+\b", clean, flags=re.UNICODE)
    tail = " ".join(sentences[1:]) if len(sentences) > 1 else ""
    tail_words = re.findall(r"\b[\w\-]+\b", tail, flags=re.UNICODE)
    return {
        "answer_word_count": float(len(words)),
        "answer_sentence_count": float(len(sentences)),
        "answer_tail_word_count": float(len(tail_words)),
        "answer_tail_ratio": float(len(tail_words) / max(1, len(words))),
    }


def drift_diagnostics(row: Any) -> dict[str, Any]:
    shape = answer_shape(str(row.get("prompt", "")), str(row.get("model_answer", "")))
    tail_entities = _num(row, "evidence_tail_unsupported_entity_count")
    tail_numbers = _num(row, "evidence_tail_unsupported_number_count")
    entity_gap = _num(row, "evidence_answer_entity_not_in_evidence_ratio")
    number_gap = _num(row, "evidence_answer_number_not_in_evidence_ratio")
    if not _has_key(row, "evidence_answer_entity_not_in_evidence_ratio"):
        entity_gap = _fallback_entity_gap(row)
    if not _has_key(row, "evidence_answer_number_not_in_evidence_ratio"):
        number_gap = _fallback_number_gap(row)
    return {
        **shape,
        "drift_low_overlap": bool(_num(row, "evidence_top_evidence_overlap") < 0.12),
        "drift_weak_audit_path": bool(str(row.get("score_path", "")) == "fallback" or str(row.get("audit_status", "")) in WEAK_AUDIT_STATUSES),
        "drift_long_answer_shape": bool(shape["answer_word_count"] >= 45 or shape["answer_sentence_count"] >= 3),
        "drift_tail_unsupported": bool(tail_entities >= 2 or tail_numbers >= 1),
        "drift_evidence_gap": bool(entity_gap >= 0.60 or number_gap >= 0.60),
    }


def overlay_decision_v6(row: Any, *, cap: float = 0.60) -> tuple[float, str, str, float]:
    base = _num(row, "is_hallucination_proba")
    profile = str(row.get("_profile") or FeatureExtractor()._question_profile(str(row.get("prompt", ""))))
    kind = expected_answer_kind(str(row.get("prompt", "")))
    if profile not in ALLOWED_DRIFT_PROFILES:
        return base, "unchanged_profile_not_allowed", profile, 0.0
    if kind in {"year", "date_or_month", "count"}:
        return base, "unchanged_expected_numeric", profile, 0.0
    if _num(row, "answer_word_count") < 35:
        return base, "unchanged_short_answer", profile, 0.0
    if base >= 0.65:
        return base, "unchanged_already_high_score", profile, 0.0
    if str(row.get("evidence_status", "missing")) != "ok":
        return base, "unchanged_evidence_not_ok", profile, 0.0
    if _num(row, "evidence_retrieval_hit_count") < MIN_RETRIEVAL_HITS:
        return base, "unchanged_low_retrieval_hits", profile, 0.0
    if _num(row, "refute_overlay_delta") > 1e-12:
        return base, "unchanged_v4_already_changed", profile, 0.0

    if _supported_core_unsupported_tail(row):
        score = max(base, cap)
        return score, "supported_core_unsupported_tail", profile, score - base
    if _fallback_long_low_overlap(row):
        score = max(base, cap)
        return score, "fallback_long_low_overlap", profile, score - base
    if _low_overlap_many_new_entities(row, profile):
        score = max(base, cap)
        return score, "low_overlap_many_new_entities", profile, score - base
    return base, "unchanged_no_drift_trigger", profile, 0.0


def apply_drift_overlay_frame(frame: pd.DataFrame, *, cap: float = 0.60) -> pd.DataFrame:
    output = frame.copy()
    extractor = FeatureExtractor()
    if "_profile" not in output.columns:
        output["_profile"] = output["prompt"].map(lambda value: extractor._question_profile(str(value)))
    _ensure_diagnostics(output)
    base = output["is_hallucination_proba"].astype(float).copy()
    decisions = output.apply(lambda row: overlay_decision_v6(row, cap=cap), axis=1)
    output["base_is_hallucination_proba"] = base
    output["is_hallucination_proba"] = [score for score, _, _, _ in decisions]
    output["drift_overlay_v6_reason"] = [reason for _, reason, _, _ in decisions]
    output["drift_overlay_v6_cap"] = cap
    output["drift_overlay_v6_profile"] = [profile for _, _, profile, _ in decisions]
    output["drift_overlay_v6_delta"] = output["is_hallucination_proba"] - output["base_is_hallucination_proba"]
    return output


def taxonomy_label_v6(row: Any) -> str:
    profile = str(row.get("_profile") or FeatureExtractor()._question_profile(str(row.get("prompt", ""))))
    if _supported_core_unsupported_tail(row):
        return "supported_core_unsupported_tail"
    if _fallback_long_low_overlap(row):
        return "fallback_long_low_overlap"
    if _low_overlap_many_new_entities(row, profile):
        return "low_overlap_many_new_entities"
    if _num(row, "evidence_top_evidence_overlap") < 0.12 or _num(row, "evidence_aligned_hit_count") < 1:
        return "bad_retrieval_alignment"
    if profile == "generic" and _num(row, "answer_word_count") >= 45:
        return "unsupported_long_generic"
    if _num(row, "evidence_claim_unknown_count") >= 2:
        return "low_overlap_many_unknown_claims"
    return "not_v6_candidate"


def _ensure_diagnostics(frame: pd.DataFrame) -> None:
    for column in ("evidence_tail_unsupported_entity_count", "evidence_tail_unsupported_number_count"):
        if column not in frame.columns:
            frame[column] = 0.0
    if "evidence_answer_entity_not_in_evidence_ratio" not in frame.columns:
        frame["evidence_answer_entity_not_in_evidence_ratio"] = frame.apply(_fallback_entity_gap, axis=1)
    if "evidence_answer_number_not_in_evidence_ratio" not in frame.columns:
        frame["evidence_answer_number_not_in_evidence_ratio"] = frame.apply(_fallback_number_gap, axis=1)
    missing = [column for column in DRIFT_DIAGNOSTIC_COLUMNS if column not in frame.columns]
    if not missing:
        return
    diag = pd.DataFrame([drift_diagnostics(row) for _, row in frame.iterrows()], index=frame.index)
    for column in DRIFT_DIAGNOSTIC_COLUMNS:
        frame[column] = diag[column]


def _supported_core_unsupported_tail(row: Any) -> bool:
    return bool(
        _num(row, "answer_word_count") >= 45
        and _num(row, "answer_sentence_count") >= 2
        and _num(row, "evidence_core_supported") == 1
        and _num(row, "evidence_core_refuted") == 0
        and _num(row, "evidence_claim_refuted_count") == 0
        and _num(row, "evidence_claim_unknown_count") >= 1
        and (_num(row, "evidence_tail_unsupported_entity_count") >= 2 or _num(row, "evidence_tail_unsupported_number_count") >= 1)
        and _num(row, "evidence_answer_entity_not_in_evidence_ratio") >= 0.50
    )


def _fallback_long_low_overlap(row: Any) -> bool:
    return bool(
        (str(row.get("score_path", "")) == "fallback" or str(row.get("audit_status", "")) in WEAK_AUDIT_STATUSES)
        and _num(row, "answer_word_count") >= 55
        and _num(row, "evidence_top_evidence_overlap") < 0.12
        and _num(row, "evidence_claim_unknown_count") >= 1
        and _num(row, "evidence_answer_entity_not_in_evidence_ratio") >= 0.60
        and _num(row, "base_is_hallucination_proba", fallback_column="is_hallucination_proba") < 0.50
    )


def _low_overlap_many_new_entities(row: Any, profile: str) -> bool:
    return bool(
        profile == "generic"
        and _num(row, "answer_word_count") >= 45
        and _num(row, "evidence_top_evidence_overlap") < 0.10
        and _num(row, "evidence_answer_core_entity_missing_from_aligned_evidence_count") >= 1
        and _num(row, "evidence_aligned_alternative_entity_count") >= 1
        and _num(row, "evidence_answer_entity_not_in_evidence_ratio") >= 0.75
        and not (_num(row, "evidence_core_supported") == 1 and _num(row, "evidence_claim_unknown_count") == 0)
    )


def _fallback_entity_gap(row: Any) -> float:
    count = _num(row, "evidence_entity_refuted_count")
    answer = str(row.get("model_answer", ""))
    prompt = str(row.get("prompt", ""))
    entity_like = len(re.findall(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:[\s\-][А-ЯЁA-Z][а-яёa-z]+){0,4}\b", strip_markup(answer)))
    prompt_overlap = len(content_tokens(answer) & content_tokens(prompt)) / max(1, len(content_tokens(answer)))
    if entity_like <= 0:
        return 0.0
    if count > 0:
        return min(1.0, count / max(1.0, float(entity_like)))
    return 1.0 - min(1.0, prompt_overlap)


def _fallback_number_gap(row: Any) -> float:
    count = _num(row, "evidence_number_refuted_count") + _num(row, "evidence_year_refuted_count")
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", str(row.get("model_answer", ""))))
    return 0.0 if numbers == 0 else min(1.0, count / max(1.0, float(numbers)))


def _num(row: Any, column: str, *, fallback_column: str | None = None) -> float:
    value = row.get(column, None)
    if value is None and fallback_column is not None:
        value = row.get(fallback_column, 0.0)
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_key(row: Any, column: str) -> bool:
    if isinstance(row, dict):
        return column in row
    index = getattr(row, "index", None)
    return bool(index is not None and column in index)
