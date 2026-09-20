from __future__ import annotations

import html
import re
from dataclasses import dataclass


WORD_RE = re.compile(r"\b[\w\-]+\b", flags=re.UNICODE)
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
CAPITALIZED_RE = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:[\s\-][А-ЯЁA-Z][а-яёa-z]+){0,4}\b")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_TOKEN_RE = re.compile(r"[*_`>#]+")


@dataclass(frozen=True)
class AtomicClaim:
    text: str
    is_core: bool
    entities: tuple[str, ...]
    numbers: tuple[str, ...]
    years: tuple[str, ...]


@dataclass(frozen=True)
class ExtractedClaims:
    core_answer: str
    tail_text: str
    claims: tuple[AtomicClaim, ...]
    entities: tuple[str, ...]
    numbers: tuple[str, ...]
    years: tuple[str, ...]


def strip_markup(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = MARKDOWN_TOKEN_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    clean = strip_markup(text)
    if not clean:
        return []
    parts = [part.strip(" -;\t") for part in re.split(r"(?<=[.!?])\s+|(?:\n+)", clean) if part.strip(" -;\t")]
    if len(parts) <= 1:
        parts = [part.strip() for part in re.split(r"\s*[;•]\s*", clean) if part.strip()]
    return parts or [clean]


def normalize_number(value: str) -> str:
    return value.strip().rstrip("%").replace(",", ".")


def extract_numbers(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalize_number(value) for value in NUMBER_RE.findall(text)))


def extract_years(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(YEAR_RE.findall(text)))


def extract_entities(text: str) -> tuple[str, ...]:
    entities: list[str] = []
    for raw in CAPITALIZED_RE.findall(text):
        value = re.sub(r"\s+", " ", raw).strip()
        if len(value) <= 1:
            continue
        if value.lower() in {"the", "a", "an", "что", "кто", "где", "когда"}:
            continue
        entities.append(value)
    return tuple(dict.fromkeys(entities))


def content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "в",
        "во",
        "где",
        "и",
        "из",
        "как",
        "какие",
        "какой",
        "когда",
        "кто",
        "на",
        "о",
        "по",
        "сколько",
        "что",
        "это",
    }
    return {token.lower() for token in WORD_RE.findall(strip_markup(text)) if len(token) > 1 and token.lower() not in stopwords}


def extract_claims(answer: str) -> ExtractedClaims:
    sentences = split_sentences(answer)
    core = sentences[0] if sentences else ""
    tail = " ".join(sentences[1:])
    claims: list[AtomicClaim] = []
    for idx, sentence in enumerate(sentences):
        entities = extract_entities(sentence)
        numbers = extract_numbers(sentence)
        years = extract_years(sentence)
        has_fact_shape = bool(entities or numbers or years or len(content_tokens(sentence)) >= 4)
        if not has_fact_shape:
            continue
        claims.append(
            AtomicClaim(
                text=sentence,
                is_core=idx == 0,
                entities=entities,
                numbers=numbers,
                years=years,
            )
        )
    return ExtractedClaims(
        core_answer=core,
        tail_text=tail,
        claims=tuple(claims),
        entities=extract_entities(answer),
        numbers=extract_numbers(answer),
        years=extract_years(answer),
    )
