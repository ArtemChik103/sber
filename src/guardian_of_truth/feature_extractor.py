from __future__ import annotations

import re

import numpy as np

from guardian_of_truth.api_client import AuditPayload


class FeatureExtractor:
    api_feature_names = ["h", "n", "e", "inv_r", "u", "c", "x"]
    text_feature_names = [
        "answer_len_words",
        "digit_density",
        "year_count",
        "prompt_overlap_ratio",
        "prompt_entity_coverage",
        "hedging_ratio",
        "question_type_mismatch",
    ]

    HEDGING_WORDS = {
        "возможно",
        "кажется",
        "вероятно",
        "предположительно",
        "примерно",
        "около",
        "может",
        "может быть",
    }
    STOPWORDS = {
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
        "какая",
        "какие",
        "каким",
        "каких",
        "какой",
        "как",
        "когда",
        "кто",
        "на",
        "о",
        "по",
        "сколько",
        "чем",
        "что",
        "это",
        "этот",
        "эта",
    }
    WHO_HINTS = {"кто", "who"}
    WHEN_HINTS = {"когда", "в каком году", "в каком", "when", "what year"}
    WHERE_HINTS = {"где", "в какой стране", "в каком городе", "where"}
    COUNT_HINTS = {"сколько", "how many", "how much"}

    WORD_RE = re.compile(r"\b[\w\-]+\b", flags=re.UNICODE)
    YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b")
    CAPITALIZED_RE = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]+\b")
    NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

    def extract(self, prompt: str, answer: str, audit: AuditPayload) -> np.ndarray:
        api = np.array(
            [
                audit.h,
                audit.n,
                audit.e,
                1.0 - audit.r,
                audit.u,
                audit.c,
                audit.x,
            ],
            dtype=np.float32,
        )
        text = np.array(self._text_features(prompt, answer), dtype=np.float32)
        return np.concatenate([api, text]).astype(np.float32)

    def extract_text_only(self, prompt: str, answer: str) -> np.ndarray:
        return np.array(self._text_features(prompt, answer), dtype=np.float32)

    def _text_features(self, prompt: str, answer: str) -> list[float]:
        prompt_words = self.WORD_RE.findall(prompt)
        answer_words = self.WORD_RE.findall(answer)
        word_count = len(answer_words)
        char_count = max(1, len(answer))
        lower_answer_words = [word.lower() for word in answer_words]
        hedging_hits = sum(1 for word in lower_answer_words if word in self.HEDGING_WORDS)
        prompt_tokens = self._content_tokens(prompt)
        answer_tokens = self._content_tokens(answer)
        prompt_entities = self._entity_tokens(prompt)
        answer_entities = self._entity_tokens(answer)
        prompt_overlap_ratio = len(prompt_tokens & answer_tokens) / max(1, len(prompt_tokens))
        prompt_entity_coverage = len(prompt_entities & answer_entities) / max(1, len(prompt_entities))
        question_type_mismatch = self._question_type_mismatch(
            prompt,
            answer,
            prompt_words=prompt_words,
            answer_words=answer_words,
            answer_entities=answer_entities,
        )

        return [
            float(word_count),
            sum(ch.isdigit() for ch in answer) / char_count,
            float(len(self.YEAR_RE.findall(answer))),
            prompt_overlap_ratio,
            prompt_entity_coverage,
            hedging_hits / max(1, word_count),
            question_type_mismatch,
        ]

    def _content_tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in self.WORD_RE.findall(text)
            if len(token) > 1 and token.lower() not in self.STOPWORDS
        }

    def _entity_tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in self.CAPITALIZED_RE.findall(text)
            if token.lower() not in self.STOPWORDS
        }

    def _question_type_mismatch(
        self,
        prompt: str,
        answer: str,
        *,
        prompt_words: list[str],
        answer_words: list[str],
        answer_entities: set[str],
    ) -> float:
        prompt_lower = prompt.lower()
        answer_numbers = self.NUMBER_RE.findall(answer)
        answer_years = self.YEAR_RE.findall(answer)
        word_count = len(answer_words)
        mismatch = 0.0

        if any(hint in prompt_lower for hint in self.WHO_HINTS):
            if not answer_entities:
                mismatch += 0.75
            if word_count > 18:
                mismatch += 0.25

        if any(hint in prompt_lower for hint in self.WHEN_HINTS):
            if not answer_numbers and not answer_years:
                mismatch += 0.8
            if word_count > 16:
                mismatch += 0.2

        if any(hint in prompt_lower for hint in self.WHERE_HINTS):
            if not answer_entities:
                mismatch += 0.7
            if word_count > 20:
                mismatch += 0.15

        if any(hint in prompt_lower for hint in self.COUNT_HINTS):
            if not answer_numbers:
                mismatch += 0.85
            if word_count > 18:
                mismatch += 0.15

        if mismatch == 0.0:
            prompt_token_count = max(1, len(self._content_tokens(prompt)))
            overlap = len(self._content_tokens(prompt) & self._content_tokens(answer)) / prompt_token_count
            if word_count > 48 and overlap < 0.2:
                mismatch = 0.35

        return float(min(1.0, mismatch))
