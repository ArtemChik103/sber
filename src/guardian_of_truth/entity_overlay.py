from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from guardian_of_truth.claims import content_tokens, extract_claims, strip_markup
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.refute_overlay import expected_answer_kind


ENTITY_OVERLAY_POLICY = "v5_entity_overlay"
ENTITY_OVERLAY_CAPS = (0.65, 0.72, 0.82)
ALLOWED_ENTITY_PROFILES = {"who", "by_whom", "title_name", "where"}
MIN_ENTITY_TOP_OVERLAP = 0.28
TRUSTED_SOURCE_PREFIXES = ("wikipedia", "wikidata", "seed", "trusted")

GENERIC_ENTITY_WORDS = {
    "answer",
    "the",
    "this",
    "that",
    "who",
    "where",
    "when",
    "what",
    "which",
    "ответ",
    "кто",
    "где",
    "когда",
    "что",
    "какой",
    "какая",
    "какое",
    "какие",
    "название",
    "страна",
    "город",
    "фильм",
    "книга",
    "роман",
    "песня",
}

LOCATION_HINTS = {
    "city",
    "country",
    "state",
    "province",
    "located",
    "location",
    "город",
    "страна",
    "область",
    "регион",
    "столица",
    "находится",
    "расположен",
    "расположена",
}


@dataclass(frozen=True)
class EntityDiagnostics:
    answer_core_entities: tuple[str, ...]
    aligned_evidence_entities: tuple[str, ...]
    answer_core_entity_count: float
    answer_core_entity_in_aligned_evidence_count: float
    answer_core_entity_missing_from_aligned_evidence_count: float
    aligned_evidence_entity_count: float
    aligned_alternative_entity_count: float
    aligned_entity_mismatch_candidate: float
    aligned_entity_confidence: float


ENTITY_DIAGNOSTIC_COLUMNS = [
    "evidence_answer_core_entities",
    "evidence_aligned_evidence_entities",
    "evidence_answer_core_entity_count",
    "evidence_answer_core_entity_in_aligned_evidence_count",
    "evidence_answer_core_entity_missing_from_aligned_evidence_count",
    "evidence_aligned_evidence_entity_count",
    "evidence_aligned_alternative_entity_count",
    "evidence_aligned_entity_mismatch_candidate",
    "evidence_aligned_entity_confidence",
]


def normalize_entity(value: str) -> str:
    clean = strip_markup(value).replace("ё", "е").replace("Ё", "Е")
    clean = re.sub(r"[\[\](){},.;:!?\"'«»“”‘’`*_#<>|/\\]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip().lower()
    return clean


def extract_core_entities(text: str, *, prompt: str = "", allow_prompt_entities: bool = False) -> tuple[str, ...]:
    core = extract_claims(text).core_answer or str(text or "")
    prompt_entities = set(extract_text_entities(prompt)) if not allow_prompt_entities else set()
    entities: list[str] = []
    for raw in re.findall(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:[\s\-][А-ЯЁA-Z][а-яёa-z]+){0,4}\b", core):
        normalized = normalize_entity(raw)
        if _keep_entity(raw, normalized, prompt_entities=prompt_entities):
            entities.append(normalized)
    if not entities:
        tokens = [token for token in content_tokens(core) if token not in GENERIC_ENTITY_WORDS]
        if 1 <= len(tokens) <= 4 and len(core.split()) <= 6:
            entities.append(normalize_entity(" ".join(tokens)))
    return tuple(dict.fromkeys(entity for entity in entities if entity))


def extract_text_entities(text: str) -> tuple[str, ...]:
    entities: list[str] = []
    for raw in re.findall(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:[\s\-][А-ЯЁA-Z][а-яёa-z]+){0,4}\b", strip_markup(text)):
        normalized = normalize_entity(raw)
        if _keep_entity(raw, normalized, prompt_entities=set()):
            entities.append(normalized)
    return tuple(dict.fromkeys(entities))


def entity_diagnostics(
    prompt: str,
    answer: str,
    aligned_texts: list[str],
    *,
    aligned_title_entity_match: bool,
    top_overlap: float,
) -> EntityDiagnostics:
    profile = FeatureExtractor()._question_profile(prompt)
    answer_entities = extract_core_entities(prompt=prompt, text=answer, allow_prompt_entities=profile in {"where", "title_name"})
    aligned_entities = tuple(dict.fromkeys(entity for text in aligned_texts for entity in extract_text_entities(text)))
    aligned_norm_text = normalize_entity(" ".join(aligned_texts))
    in_evidence = tuple(entity for entity in answer_entities if entity and entity in aligned_norm_text)
    missing = tuple(entity for entity in answer_entities if entity and entity not in aligned_norm_text)
    prompt_entities = set(extract_text_entities(prompt))
    alternatives = tuple(entity for entity in aligned_entities if entity not in set(answer_entities) and entity not in prompt_entities)
    confidence = 0.0
    if answer_entities:
        confidence += 0.35
    if missing:
        confidence += 0.25
    if alternatives:
        confidence += 0.25
    if aligned_title_entity_match:
        confidence += 0.10
    if top_overlap >= MIN_ENTITY_TOP_OVERLAP:
        confidence += 0.05
    mismatch = bool(answer_entities and missing and alternatives and aligned_title_entity_match and top_overlap >= MIN_ENTITY_TOP_OVERLAP)
    return EntityDiagnostics(
        answer_core_entities=answer_entities,
        aligned_evidence_entities=aligned_entities,
        answer_core_entity_count=float(len(answer_entities)),
        answer_core_entity_in_aligned_evidence_count=float(len(in_evidence)),
        answer_core_entity_missing_from_aligned_evidence_count=float(len(missing)),
        aligned_evidence_entity_count=float(len(aligned_entities)),
        aligned_alternative_entity_count=float(len(alternatives)),
        aligned_entity_mismatch_candidate=float(mismatch),
        aligned_entity_confidence=float(min(1.0, confidence)),
    )


def diagnostic_columns(prompt: str, answer: str, compact_json: str, *, aligned_title_entity_match: float, top_overlap: float) -> dict[str, Any]:
    compact = parse_compact_json(compact_json)
    aligned = compact.get("aligned_snippets", [])
    aligned_texts = [f"{item.get('title', '')} {item.get('text', '')}" for item in aligned if isinstance(item, dict)]
    diag = entity_diagnostics(
        prompt,
        answer,
        aligned_texts,
        aligned_title_entity_match=aligned_title_entity_match > 0,
        top_overlap=top_overlap,
    )
    return {
        "evidence_answer_core_entities": json.dumps(list(diag.answer_core_entities), ensure_ascii=False),
        "evidence_aligned_evidence_entities": json.dumps(list(diag.aligned_evidence_entities), ensure_ascii=False),
        "evidence_answer_core_entity_count": diag.answer_core_entity_count,
        "evidence_answer_core_entity_in_aligned_evidence_count": diag.answer_core_entity_in_aligned_evidence_count,
        "evidence_answer_core_entity_missing_from_aligned_evidence_count": diag.answer_core_entity_missing_from_aligned_evidence_count,
        "evidence_aligned_evidence_entity_count": diag.aligned_evidence_entity_count,
        "evidence_aligned_alternative_entity_count": diag.aligned_alternative_entity_count,
        "evidence_aligned_entity_mismatch_candidate": diag.aligned_entity_mismatch_candidate,
        "evidence_aligned_entity_confidence": diag.aligned_entity_confidence,
    }


def overlay_decision_v5(row: Any, *, cap: float = 0.82) -> tuple[float, str, str, float]:
    base = _num(row, "is_hallucination_proba")
    profile = str(row.get("_profile") or FeatureExtractor()._question_profile(str(row.get("prompt", ""))))
    kind = expected_answer_kind(str(row.get("prompt", "")))
    if kind in {"year", "date_or_month", "count"}:
        return base, "unchanged_expected_numeric", profile, 0.0
    if profile not in ALLOWED_ENTITY_PROFILES:
        return base, "unchanged_profile_not_allowed", profile, 0.0
    if _word_count(str(row.get("model_answer", ""))) > 40:
        return base, "unchanged_long_answer", profile, 0.0
    if _num(row, "evidence_answer_core_entity_count") > 4:
        return base, "unchanged_too_many_entities", profile, 0.0
    if _num(row, "evidence_top_evidence_overlap") < MIN_ENTITY_TOP_OVERLAP:
        return base, "unchanged_low_overlap", profile, 0.0
    if _num(row, "evidence_aligned_hit_count") < 1:
        return base, "unchanged_no_aligned_hits", profile, 0.0
    if str(row.get("evidence_status", "missing")) != "ok":
        return base, "unchanged_no_aligned_hits", profile, 0.0
    if base >= 0.82:
        return base, "unchanged_already_high_score", profile, 0.0
    if _num(row, "evidence_answer_core_entity_count") < 1:
        return base, "unchanged_no_answer_entity", profile, 0.0
    if _num(row, "evidence_answer_core_entity_missing_from_aligned_evidence_count") < 1:
        return base, "unchanged_answer_entity_supported", profile, 0.0
    if _num(row, "evidence_aligned_alternative_entity_count") < 1:
        return base, "unchanged_no_alternative_entity", profile, 0.0
    if _num(row, "evidence_aligned_title_entity_match") < 1:
        return base, "unchanged_no_alternative_entity", profile, 0.0
    if not _has_trusted_aligned_source(str(row.get("evidence_compact_json", "{}"))):
        return base, "unchanged_untrusted_evidence", profile, 0.0
    if profile == "where" and not _looks_location_like(row):
        return base, "unchanged_no_alternative_entity", profile, 0.0
    if profile == "title_name" and not _looks_title_like(str(row.get("model_answer", ""))):
        return base, "unchanged_long_answer", profile, 0.0
    score = max(base, cap)
    return score, "aligned_entity_mismatch", profile, score - base


def apply_entity_overlay_frame(frame: pd.DataFrame, *, cap: float = 0.82) -> pd.DataFrame:
    output = frame.copy()
    extractor = FeatureExtractor()
    if "_profile" not in output.columns:
        output["_profile"] = output["prompt"].map(lambda value: extractor._question_profile(str(value)))
    _ensure_diagnostics(output)
    base = output["is_hallucination_proba"].astype(float).copy()
    decisions = output.apply(lambda row: overlay_decision_v5(row, cap=cap), axis=1)
    output["base_is_hallucination_proba"] = base
    output["is_hallucination_proba"] = [score for score, _, _, _ in decisions]
    output["entity_overlay_v5"] = [reason == "aligned_entity_mismatch" for _, reason, _, _ in decisions]
    output["entity_overlay_v5_cap"] = cap
    output["entity_overlay_v5_profile"] = [profile for _, _, profile, _ in decisions]
    output["entity_overlay_v5_delta"] = output["is_hallucination_proba"] - output["base_is_hallucination_proba"]
    output["entity_overlay_v5_reason"] = [reason for _, reason, _, _ in decisions]
    return output


def _ensure_diagnostics(frame: pd.DataFrame) -> None:
    missing = [column for column in ENTITY_DIAGNOSTIC_COLUMNS if column not in frame.columns]
    if not missing:
        return
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            diagnostic_columns(
                str(row.get("prompt", "")),
                str(row.get("model_answer", "")),
                str(row.get("evidence_compact_json", "{}")),
                aligned_title_entity_match=_num(row, "evidence_aligned_title_entity_match"),
                top_overlap=_num(row, "evidence_top_evidence_overlap"),
            )
        )
    diag_frame = pd.DataFrame(rows, index=frame.index)
    for column in ENTITY_DIAGNOSTIC_COLUMNS:
        frame[column] = diag_frame[column]


def parse_compact_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _keep_entity(raw: str, normalized: str, *, prompt_entities: set[str]) -> bool:
    if not normalized or normalized in GENERIC_ENTITY_WORDS or normalized in prompt_entities:
        return False
    parts = normalized.split()
    if len(parts) == 1 and (len(parts[0]) <= 2 or parts[0] in GENERIC_ENTITY_WORDS):
        return False
    if len(parts) == 1 and raw[:1].isupper() and raw.strip().endswith("."):
        return False
    return True


def _num(row: Any, column: str) -> float:
    value = row.get(column, 0.0)
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _word_count(value: str) -> int:
    return len(re.findall(r"\b[\w\-]+\b", strip_markup(value), flags=re.UNICODE))


def _has_trusted_aligned_source(compact_json: str) -> bool:
    compact = parse_compact_json(compact_json)
    for item in compact.get("aligned_snippets", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).lower()
        if source.startswith(TRUSTED_SOURCE_PREFIXES):
            return True
    return False


def _looks_location_like(row: Any) -> bool:
    answer = normalize_entity(str(row.get("model_answer", "")))
    compact = parse_compact_json(str(row.get("evidence_compact_json", "{}")))
    evidence_text = normalize_entity(
        " ".join(
            f"{item.get('title', '')} {item.get('text', '')}"
            for item in compact.get("aligned_snippets", [])
            if isinstance(item, dict)
        )
    )
    return any(hint in answer or hint in evidence_text for hint in LOCATION_HINTS) or _word_count(str(row.get("model_answer", ""))) <= 5


def _looks_title_like(answer: str) -> bool:
    return _word_count(answer) <= 12 and len(extract_core_entities(answer, allow_prompt_entities=True)) <= 3
