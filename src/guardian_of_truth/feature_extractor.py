from __future__ import annotations

import re

import numpy as np

from guardian_of_truth.api_client import AuditPayload


class FeatureExtractor:
    api_feature_names = [
        "h",
        "n",
        "e",
        "inv_r",
        "u",
        "c",
        "x",
        "inv_q",
        "inv_s",
        "m",
        "inv_sem",
        "wrong_entity",
        "wrong_number",
        "unsupported_extra",
        "too_broad",
        "verifier_confidence",
    ]
    text_feature_names = [
        "answer_len_words",
        "prompt_overlap_ratio",
        "prompt_entity_coverage",
        "prompt_number_coverage",
        "answer_new_content_ratio",
        "answer_new_number_ratio",
        "question_type_mismatch",
        "answer_number_count",
        "answer_new_number_count",
        "answer_year_count",
        "answer_has_exactly_one_numeric_answer",
        "typed_answer_len_bucket",
        "typed_long_answer_penalty",
        "count_has_exactly_one_numeric_answer",
        "typed_short_answer_expected",
        "answer_sentence_count",
        "answer_first_sentence_word_count",
        "answer_tail_word_count",
        "tail_new_number_count",
        "tail_new_entity_count",
        "first_sentence_has_numeric_answer",
        "first_sentence_has_entity_answer",
        "typed_exact_answer_shape",
        "generic_long_supported_shape",
        "list_like_answer_count",
        "question_expected_list_count",
        "answer_overexplains_typed_question",
    ]
    evidence_feature_names = [
        "retrieval_hit_count",
        "top_bm25_score",
        "top_evidence_overlap",
        "core_supported",
        "core_refuted",
        "entity_supported_ratio",
        "entity_refuted_count",
        "number_supported_ratio",
        "number_refuted_count",
        "year_supported_ratio",
        "year_refuted_count",
        "claim_supported_count",
        "claim_refuted_count",
        "claim_unknown_count",
        "tail_unsupported_entity_count",
        "tail_unsupported_number_count",
        "evidence_missing_for_typed_question",
        "answer_entity_not_in_evidence_ratio",
        "answer_number_not_in_evidence_ratio",
        "evidence_status_missing",
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
    WHICH_LIST_HINTS = {"какие", "which", "list", "перечислите", "назовите"}
    WHAT_PROPERTY_HINTS = {
        "какова",
        "каков",
        "какое значение",
        "what is the frequency",
        "what is the name",
        "what is the value",
        "what frequency",
        "what name",
        "what value",
    }
    BY_WHOM_HINTS = {"кем", "by whom", "who was", "who were"}
    TITLE_NAME_HINTS = {"какое название", "как называется", "what title", "what name"}
    DEFINITION_HINTS = {"что такое", "what is"}
    WHEN_HINTS = {"когда", "в каком году", "в каком", "when", "what year"}
    WHERE_HINTS = {"где", "в какой стране", "в каком городе", "where"}
    COUNT_HINTS = {"сколько", "how many", "how much"}
    TYPED_SHORT_PROFILES = {"who", "when", "where", "count", "what_property", "by_whom", "title_name"}
    NORMALIZE_SUFFIXES = (
        "иями",
        "ями",
        "ами",
        "его",
        "ого",
        "ему",
        "ому",
        "ыми",
        "ими",
        "ий",
        "ый",
        "ой",
        "ая",
        "ое",
        "ее",
        "ые",
        "ие",
        "ых",
        "их",
        "ам",
        "ям",
        "ах",
        "ях",
        "ом",
        "ем",
        "ою",
        "ею",
        "ия",
        "ии",
        "ья",
        "ие",
        "ий",
        "ов",
        "ев",
        "а",
        "я",
        "ы",
        "и",
        "е",
        "у",
        "ю",
    )

    WORD_RE = re.compile(r"\b[\w\-]+\b", flags=re.UNICODE)
    YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b")
    CAPITALIZED_RE = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]+\b")
    NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

    def extract(self, prompt: str, answer: str, audit: AuditPayload, evidence_audit=None) -> np.ndarray:
        api = np.array(
            [
                audit.h,
                audit.n,
                audit.e,
                1.0 - audit.r,
                audit.u,
                audit.c,
                audit.x,
                1.0 - audit.q,
                1.0 - audit.s,
                audit.m,
                1.0 - audit.sem,
                audit.we,
                audit.wn,
                audit.ue,
                audit.bt,
                audit.conf,
            ],
            dtype=np.float32,
        )
        text = np.array(self._text_features(prompt, answer), dtype=np.float32)
        if evidence_audit is None:
            return np.concatenate([api, text]).astype(np.float32)
        from guardian_of_truth.evidence import evidence_feature_values

        evidence = np.array(evidence_feature_values(evidence_audit), dtype=np.float32)
        return np.concatenate([api, text, evidence]).astype(np.float32)

    def extract_text_only(self, prompt: str, answer: str) -> np.ndarray:
        return np.array(self._text_features(prompt, answer), dtype=np.float32)

    def _text_features(self, prompt: str, answer: str) -> list[float]:
        prompt_words = self.WORD_RE.findall(prompt)
        answer_words = self.WORD_RE.findall(answer)
        word_count = len(answer_words)
        prompt_tokens = self._content_tokens(prompt)
        answer_tokens = self._content_tokens(answer)
        prompt_entities = self._entity_tokens(prompt)
        answer_entities = self._entity_tokens(answer)
        prompt_numbers = self._number_tokens(prompt)
        answer_numbers = self._number_tokens(answer)
        answer_years = self.YEAR_RE.findall(answer)
        profile = self._question_profile(prompt)
        is_typed_short_answer = profile in self.TYPED_SHORT_PROFILES
        sentences = self._sentences(answer)
        first_sentence = sentences[0] if sentences else answer
        tail_text = " ".join(sentences[1:])
        first_sentence_words = self.WORD_RE.findall(first_sentence)
        tail_words = self.WORD_RE.findall(tail_text)
        first_sentence_numbers = self._number_tokens(first_sentence)
        first_sentence_entities = self._entity_tokens(first_sentence)
        tail_numbers = self._number_tokens(tail_text)
        tail_entities = self._entity_tokens(tail_text)
        prompt_overlap_ratio = len(prompt_tokens & answer_tokens) / max(1, len(prompt_tokens))
        prompt_entity_coverage = len(prompt_entities & answer_entities) / max(1, len(prompt_entities))
        prompt_number_coverage = len(prompt_numbers & answer_numbers) / max(1, len(prompt_numbers))
        answer_new_content_ratio = len(answer_tokens - prompt_tokens) / max(1, len(answer_tokens))
        answer_new_number_ratio = len(answer_numbers - prompt_numbers) / max(1, len(answer_numbers))
        question_type_mismatch = self._question_type_mismatch(
            prompt,
            answer,
            prompt_words=prompt_words,
            answer_words=answer_words,
            answer_entities=answer_entities,
            answer_numbers=answer_numbers,
            answer_new_content_ratio=answer_new_content_ratio,
        )
        answer_number_count = float(min(6, len(answer_numbers)))
        answer_new_number_count = float(min(6, len(answer_numbers - prompt_numbers)))
        answer_year_count = float(min(6, len(answer_years)))
        answer_has_exactly_one_numeric_answer = float(len(answer_numbers) == 1)
        typed_answer_len_bucket = self._typed_answer_len_bucket(word_count) if is_typed_short_answer else 0.0
        typed_long_answer_penalty = self._typed_long_answer_penalty(profile, word_count)
        count_has_exactly_one_numeric_answer = float(profile == "count" and len(answer_numbers) == 1)
        answer_sentence_count = float(min(6, len(sentences)))
        answer_first_sentence_word_count = float(min(80, len(first_sentence_words)))
        answer_tail_word_count = float(min(160, len(tail_words)))
        tail_new_number_count = float(min(6, len(tail_numbers - prompt_numbers - first_sentence_numbers)))
        tail_new_entity_count = float(min(6, len(tail_entities - prompt_entities - first_sentence_entities)))
        first_sentence_has_numeric_answer = float(bool(first_sentence_numbers))
        first_sentence_has_entity_answer = float(bool(first_sentence_entities))
        typed_exact_answer_shape = float(
            is_typed_short_answer
            and len(first_sentence_words) <= 14
            and (
                bool(first_sentence_numbers)
                if profile in {"when", "count"}
                else bool(first_sentence_entities) or len(first_sentence_words) <= 5
            )
        )
        generic_long_supported_shape = float(
            profile in {"generic", "definition"}
            and word_count >= 28
            and prompt_overlap_ratio >= 0.25
            and answer_new_content_ratio <= 0.85
        )
        list_like_answer_count = float(min(6, self._list_like_count(answer)))
        question_expected_list_count = float(min(6, self._expected_list_count(prompt)))
        answer_overexplains_typed_question = float(
            is_typed_short_answer
            and len(tail_words) >= 12
            and (len(tail_numbers - prompt_numbers) > 0 or len(tail_entities - prompt_entities) > 0)
        )

        return [
            float(word_count),
            prompt_overlap_ratio,
            prompt_entity_coverage,
            prompt_number_coverage,
            answer_new_content_ratio,
            answer_new_number_ratio,
            question_type_mismatch,
            answer_number_count,
            answer_new_number_count,
            answer_year_count,
            answer_has_exactly_one_numeric_answer,
            typed_answer_len_bucket,
            typed_long_answer_penalty,
            count_has_exactly_one_numeric_answer,
            float(is_typed_short_answer),
            answer_sentence_count,
            answer_first_sentence_word_count,
            answer_tail_word_count,
            tail_new_number_count,
            tail_new_entity_count,
            first_sentence_has_numeric_answer,
            first_sentence_has_entity_answer,
            typed_exact_answer_shape,
            generic_long_supported_shape,
            list_like_answer_count,
            question_expected_list_count,
            answer_overexplains_typed_question,
        ]

    @staticmethod
    def _sentences(text: str) -> list[str]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
        if sentences:
            return sentences
        return [text.strip()] if text.strip() else []

    @staticmethod
    def _list_like_count(text: str) -> int:
        numbered = len(re.findall(r"(?:^|\n|\s)(?:\d+[.)]|[-*])\s+", text))
        separators = text.count(";") + max(0, text.count(",") - 1)
        return max(numbered, min(6, separators + 1 if separators else 0))

    @staticmethod
    def _expected_list_count(prompt: str) -> int:
        prompt_lower = prompt.lower()
        match = re.search(r"\b(?:какие|назовите|перечислите)\s+(\d+|три|две|два|четыре|пять)\b", prompt_lower)
        if not match:
            return 0
        raw = match.group(1)
        words = {"две": 2, "два": 2, "три": 3, "четыре": 4, "пять": 5}
        return int(words.get(raw, raw))

    def _content_tokens(self, text: str) -> set[str]:
        return {
            self._normalize_token(token)
            for token in self.WORD_RE.findall(text)
            if len(token) > 1 and token.lower() not in self.STOPWORDS
        }

    def _entity_tokens(self, text: str) -> set[str]:
        return {
            self._normalize_token(token)
            for token in self.CAPITALIZED_RE.findall(text)
            if token.lower() not in self.STOPWORDS
        }

    def _number_tokens(self, text: str) -> set[str]:
        return {token.replace(",", ".") for token in self.NUMBER_RE.findall(text)}

    def _normalize_token(self, token: str) -> str:
        normalized = token.lower()
        if len(normalized) <= 4:
            return normalized
        for suffix in self.NORMALIZE_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
                return normalized[: -len(suffix)]
        return normalized

    def _question_type_mismatch(
        self,
        prompt: str,
        answer: str,
        *,
        prompt_words: list[str],
        answer_words: list[str],
        answer_entities: set[str],
        answer_numbers: set[str],
        answer_new_content_ratio: float,
    ) -> float:
        prompt_lower = prompt.lower()
        answer_years = self.YEAR_RE.findall(answer)
        word_count = len(answer_words)
        mismatch = 0.0

        profile = self._question_profile(prompt)

        if profile in {"who", "by_whom"}:
            if not answer_entities:
                mismatch += 0.75
            if word_count > 18 or answer_new_content_ratio > 0.72:
                mismatch += 0.25

        if any(hint in prompt_lower for hint in self.WHEN_HINTS):
            if not answer_numbers and not answer_years:
                mismatch += 0.8
            if word_count > 16 or answer_new_content_ratio > 0.65:
                mismatch += 0.2

        if any(hint in prompt_lower for hint in self.WHERE_HINTS):
            if not answer_entities:
                mismatch += 0.7
            if word_count > 20 or answer_new_content_ratio > 0.75:
                mismatch += 0.15

        if profile == "count":
            if not answer_numbers:
                mismatch += 0.85
            if word_count > 18 or answer_new_content_ratio > 0.7:
                mismatch += 0.15

        if profile == "which_list":
            if self._list_like_count(answer) == 0 and len(answer_words) < 3:
                mismatch += 0.35

        if profile in {"what_property", "title_name"}:
            if word_count > 20 or answer_new_content_ratio > 0.75:
                mismatch += 0.25

        if mismatch == 0.0:
            prompt_token_count = max(1, len(self._content_tokens(prompt)))
            overlap = len(self._content_tokens(prompt) & self._content_tokens(answer)) / prompt_token_count
            if word_count > 48 and overlap < 0.2:
                mismatch = 0.35
            elif answer_new_content_ratio > 0.85 and word_count > 10:
                mismatch = 0.25

        return float(min(1.0, mismatch))

    def _question_profile(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(hint in prompt_lower for hint in self.TITLE_NAME_HINTS):
            return "title_name"
        if any(hint in prompt_lower for hint in self.BY_WHOM_HINTS):
            return "by_whom"
        if any(hint in prompt_lower for hint in self.WHICH_LIST_HINTS):
            return "which_list"
        if any(hint in prompt_lower for hint in self.WHAT_PROPERTY_HINTS):
            return "what_property"
        if any(hint in prompt_lower for hint in self.WHO_HINTS):
            return "who"
        if any(hint in prompt_lower for hint in self.WHEN_HINTS):
            return "when"
        if any(hint in prompt_lower for hint in self.WHERE_HINTS):
            return "where"
        if any(hint in prompt_lower for hint in self.COUNT_HINTS):
            return "count"
        if any(hint in prompt_lower for hint in self.DEFINITION_HINTS):
            return "definition"
        return "generic"

    @staticmethod
    def _typed_answer_len_bucket(word_count: int) -> float:
        if word_count <= 4:
            return 0.0
        if word_count <= 10:
            return 0.33
        if word_count <= 20:
            return 0.66
        return 1.0

    @staticmethod
    def _typed_long_answer_penalty(profile: str, word_count: int) -> float:
        if profile in {"when", "count"}:
            return float(min(1.0, max(0, word_count - 8) / 24.0))
        if profile in {"who", "where", "what_property", "by_whom", "title_name"}:
            return float(min(1.0, max(0, word_count - 12) / 28.0))
        return 0.0
