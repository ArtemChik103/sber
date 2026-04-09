from __future__ import annotations

import re
import string

import numpy as np

from guardian_of_truth.api_client import AuditPayload


class FeatureExtractor:
    api_feature_names = ["h", "n", "e", "inv_r", "u", "c", "x"]
    text_feature_names = [
        "answer_len_words",
        "digit_density",
        "year_count",
        "capitalized_ratio",
        "hedging_ratio",
        "punct_density",
        "lexical_diversity",
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

    WORD_RE = re.compile(r"\b[\w\-]+\b", flags=re.UNICODE)
    YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b")
    CAPITALIZED_RE = re.compile(r"\b[А-ЯЁA-Z][а-яёa-z]+\b")

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
        text = np.array(self._text_features(answer), dtype=np.float32)
        return np.concatenate([api, text]).astype(np.float32)

    def extract_text_only(self, answer: str) -> np.ndarray:
        return np.array(self._text_features(answer), dtype=np.float32)

    def _text_features(self, answer: str) -> list[float]:
        words = self.WORD_RE.findall(answer)
        word_count = len(words)
        char_count = max(1, len(answer))
        lower_words = [word.lower() for word in words]
        hedging_hits = sum(1 for word in lower_words if word in self.HEDGING_WORDS)

        return [
            float(word_count),
            sum(ch.isdigit() for ch in answer) / char_count,
            float(len(self.YEAR_RE.findall(answer))),
            (len(self.CAPITALIZED_RE.findall(answer)) / max(1, word_count)),
            hedging_hits / max(1, word_count),
            sum(ch in string.punctuation for ch in answer) / char_count,
            len(set(lower_words)) / max(1, word_count),
        ]
