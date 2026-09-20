from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from guardian_of_truth.claims import content_tokens, extract_claims
from guardian_of_truth.entity_overlay import diagnostic_columns
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.utils import sha256_hexdigest


SCHEMA_VERSION = "evidence_v1"
RETRIEVER_VERSION = "fts5_rules_v2"
FORBIDDEN_SOURCE_COLUMNS = {"is_hallucination", "correct_answer", "comment"}
MAX_QUERY_CANDIDATES = 5
MAX_FTS_TOKENS = 12

GENERIC_SUBJECT_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "year",
    "years",
    "date",
    "name",
    "role",
    "form",
    "approach",
    "impact",
    "experiment",
    "technique",
    "какой",
    "какая",
    "какое",
    "какие",
    "какую",
    "какого",
    "какому",
    "каким",
    "каким",
    "каком",
    "который",
    "которая",
    "которое",
    "которые",
    "кто",
    "что",
    "чем",
    "чего",
    "где",
    "когда",
    "сколько",
    "чей",
    "чья",
    "чье",
    "чьё",
    "год",
    "году",
    "года",
    "лет",
    "месяц",
    "месяце",
    "числа",
    "дата",
    "название",
    "роль",
    "форма",
    "подход",
    "влияние",
    "эксперимент",
    "техника",
    "термин",
    "автор",
    "авторы",
    "имя",
    "изображение",
    "получил",
    "получила",
    "называется",
    "является",
    "был",
    "была",
    "были",
    "это",
}


@dataclass(frozen=True)
class EvidenceSnippet:
    id: int
    title: str
    text: str
    source: str
    score: float


@dataclass(frozen=True)
class ClaimVerdict:
    text: str
    is_core: bool
    verdict: str
    reason: str


@dataclass(frozen=True)
class EvidenceAudit:
    status: str
    retrieval_latency_sec: float
    retrieval_hit_count: int
    top_bm25_score: float
    top_evidence_overlap: float
    core_supported: float
    core_refuted: float
    entity_supported_ratio: float
    entity_refuted_count: float
    number_supported_ratio: float
    number_refuted_count: float
    year_supported_ratio: float
    year_refuted_count: float
    claim_supported_count: float
    claim_refuted_count: float
    claim_unknown_count: float
    tail_unsupported_entity_count: float
    tail_unsupported_number_count: float
    evidence_missing_for_typed_question: float
    answer_entity_not_in_evidence_ratio: float
    answer_number_not_in_evidence_ratio: float
    aligned_hit_count: float = 0.0
    aligned_year_refuted_count: float = 0.0
    aligned_number_refuted_count: float = 0.0
    answer_year_in_prompt: float = 0.0
    answer_number_in_prompt: float = 0.0
    aligned_title_entity_match: float = 0.0
    answer_core_entity_count: float = 0.0
    answer_core_entity_in_aligned_evidence_count: float = 0.0
    answer_core_entity_missing_from_aligned_evidence_count: float = 0.0
    aligned_evidence_entity_count: float = 0.0
    aligned_alternative_entity_count: float = 0.0
    aligned_entity_mismatch_candidate: float = 0.0
    aligned_entity_confidence: float = 0.0
    compact_json: str = "{}"


EVIDENCE_FEATURE_NAMES = [
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


def neutral_evidence_audit(status: str = "missing") -> EvidenceAudit:
    return EvidenceAudit(
        status=status,
        retrieval_latency_sec=0.0,
        retrieval_hit_count=0,
        top_bm25_score=0.0,
        top_evidence_overlap=0.0,
        core_supported=0.0,
        core_refuted=0.0,
        entity_supported_ratio=0.0,
        entity_refuted_count=0.0,
        number_supported_ratio=0.0,
        number_refuted_count=0.0,
        year_supported_ratio=0.0,
        year_refuted_count=0.0,
        claim_supported_count=0.0,
        claim_refuted_count=0.0,
        claim_unknown_count=0.0,
        tail_unsupported_entity_count=0.0,
        tail_unsupported_number_count=0.0,
        evidence_missing_for_typed_question=0.0,
        answer_entity_not_in_evidence_ratio=0.0,
        answer_number_not_in_evidence_ratio=0.0,
        aligned_hit_count=0.0,
        aligned_year_refuted_count=0.0,
        aligned_number_refuted_count=0.0,
        answer_year_in_prompt=0.0,
        answer_number_in_prompt=0.0,
        aligned_title_entity_match=0.0,
        answer_core_entity_count=0.0,
        answer_core_entity_in_aligned_evidence_count=0.0,
        answer_core_entity_missing_from_aligned_evidence_count=0.0,
        aligned_evidence_entity_count=0.0,
        aligned_alternative_entity_count=0.0,
        aligned_entity_mismatch_candidate=0.0,
        aligned_entity_confidence=0.0,
        compact_json="{}",
    )


def evidence_feature_values(audit: EvidenceAudit | None) -> list[float]:
    if audit is None:
        audit = neutral_evidence_audit()
        missing = 1.0
    else:
        missing = float(audit.status in {"missing", "kb_missing", "disabled"})
    return [
        float(audit.retrieval_hit_count),
        float(audit.top_bm25_score),
        float(audit.top_evidence_overlap),
        float(audit.core_supported),
        float(audit.core_refuted),
        float(audit.entity_supported_ratio),
        float(audit.entity_refuted_count),
        float(audit.number_supported_ratio),
        float(audit.number_refuted_count),
        float(audit.year_supported_ratio),
        float(audit.year_refuted_count),
        float(audit.claim_supported_count),
        float(audit.claim_refuted_count),
        float(audit.claim_unknown_count),
        float(audit.tail_unsupported_entity_count),
        float(audit.tail_unsupported_number_count),
        float(audit.evidence_missing_for_typed_question),
        float(audit.answer_entity_not_in_evidence_ratio),
        float(audit.answer_number_not_in_evidence_ratio),
        missing,
    ]


class EvidenceRetriever:
    def __init__(self, db_path: str | Path, *, top_k: int = 8, retriever_version: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.top_k = top_k
        self.retriever_version = retriever_version or RETRIEVER_VERSION
        self.kb_version = self._read_meta("kb_version") if self.db_path.exists() else "missing"
        self.last_queries: list[str] = []

    @property
    def available(self) -> bool:
        return self.db_path.exists()

    def audit(self, prompt: str, answer: str) -> EvidenceAudit:
        started = time.perf_counter()
        if not self.db_path.exists():
            audit = neutral_evidence_audit("kb_missing")
            return audit.__class__(**{**audit.__dict__, "retrieval_latency_sec": time.perf_counter() - started})
        snippets = self.retrieve(prompt, answer)
        latency = time.perf_counter() - started
        return verify_against_evidence(
            prompt,
            answer,
            snippets,
            retrieval_latency_sec=latency,
            alignment_version=self.retriever_version,
            retriever_version=self.retriever_version,
            kb_version=self.kb_version,
            queries=self.last_queries,
        )

    def retrieve(self, prompt: str, answer: str) -> list[EvidenceSnippet]:
        cache_key = self.cache_key(prompt, answer)
        cached = self._cached_snippets(cache_key)
        if cached is not None:
            return cached
        if self.retriever_version in {"fts5_rules_v4", "fts5_rules_v5", "fts5_rules_v6"}:
            return self._retrieve_v4(prompt, answer, cache_key)
        if self.retriever_version == "fts5_rules_v3":
            query_candidates = query_terms_for_retrieval(prompt, answer)
        else:
            claims = extract_claims(answer)
            prompt_keywords = " ".join(sorted(content_tokens(prompt))[:10])
            query_candidates = [
                prompt,
                f"{claims.core_answer} {prompt_keywords}",
                f"{' '.join(claims.entities[:6])} {prompt_keywords}",
                f"{' '.join(claims.numbers[:6])} {prompt_keywords}",
            ]
        seen_ids: set[int] = set()
        snippets: list[EvidenceSnippet] = []
        with sqlite3.connect(self.db_path) as con:
            for query in query_candidates:
                for snippet in self._query(con, query, self.top_k * 4):
                    if snippet.id in seen_ids:
                        continue
                    seen_ids.add(snippet.id)
                    snippets.append(snippet)
        reranked = self._rerank(prompt, answer, snippets)[: self.top_k]
        self.last_queries = query_candidates
        self._store_cache(cache_key, prompt, answer, " || ".join(query_candidates), [snippet.id for snippet in reranked])
        return reranked

    def _retrieve_v4(self, prompt: str, answer: str, cache_key: str) -> list[EvidenceSnippet]:
        query_candidates = query_terms_for_retrieval_v4(prompt, answer)
        seen_ids: set[int] = set()
        snippets: list[EvidenceSnippet] = []
        with sqlite3.connect(self.db_path) as con:
            for query in query_candidates:
                for snippet in self._query(con, query, self.top_k * 8):
                    if snippet.id in seen_ids:
                        continue
                    seen_ids.add(snippet.id)
                    snippets.append(snippet)
            if len(snippets) < 3:
                recall_query = controlled_recall_query(prompt)
                if recall_query and recall_query not in query_candidates:
                    query_candidates.append(recall_query)
                    for snippet in self._query(con, recall_query, self.top_k * 8):
                        if snippet.id in seen_ids:
                            continue
                        seen_ids.add(snippet.id)
                        snippets.append(snippet)
        reranked = self._rerank_v4(prompt, answer, snippets)[: self.top_k]
        self.last_queries = query_candidates
        self._store_cache(cache_key, prompt, answer, " || ".join(query_candidates), [snippet.id for snippet in reranked])
        return reranked

    def cache_key(self, prompt: str, answer: str) -> str:
        return sha256_hexdigest(prompt, answer, self.kb_version, self.retriever_version)

    def _query(self, con: sqlite3.Connection, query: str, limit: int) -> list[EvidenceSnippet]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        try:
            rows = con.execute(
                """
                SELECT s.id, s.title, s.text, s.source, bm25(snippet_fts) AS score
                FROM snippet_fts
                JOIN snippets s ON s.id = snippet_fts.rowid
                WHERE snippet_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            EvidenceSnippet(id=int(row[0]), title=str(row[1] or ""), text=str(row[2] or ""), source=str(row[3] or ""), score=float(-row[4]))
            for row in rows
        ]

    def _rerank(self, prompt: str, answer: str, snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        if self.retriever_version != "fts5_rules_v3":
            return self._rerank_v2(prompt, answer, snippets)
        prompt_tokens = subject_terms_from_prompt(prompt)
        core_tokens = subject_terms_from_answer_core(extract_claims(answer).core_answer)
        quoted_phrases = _quoted_phrases(prompt)
        core_entities = _entity_phrases(extract_claims(answer).core_answer)
        max_bm25 = max((max(0.0, snippet.score) for snippet in snippets), default=0.0)
        ranked: list[tuple[float, EvidenceSnippet]] = []
        for snippet in snippets:
            diag = _snippet_alignment_diag(snippet, prompt_tokens, core_tokens, quoted_phrases, core_entities)
            if not _candidate_passes_precision_filter(diag):
                continue
            rank = (
                0.35 * diag["title_subject_overlap_ratio"]
                + 0.30 * diag["prompt_subject_overlap_ratio"]
                + 0.20 * diag["answer_core_overlap_ratio"]
                + 0.10 * float(diag["quoted_phrase_hit"])
                + 0.05 * (max(0.0, snippet.score) / max(1.0, max_bm25))
            )
            if diag["generic_only_match"]:
                rank -= 0.25
            if diag["title_subject_overlap"] == 0 and diag["prompt_subject_overlap"] < 2:
                rank -= 0.20
            if _weak_source(snippet.source) and not (diag["title_subject_overlap"] or diag["answer_core_entity_hit"]):
                rank -= 0.15
            if rank >= 0.18:
                ranked.append((rank, snippet))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidenceSnippet(id=snippet.id, title=snippet.title, text=snippet.text, source=snippet.source, score=float(rank))
            for rank, snippet in ranked
        ]

    def _rerank_v4(self, prompt: str, answer: str, snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        prompt_tokens = subject_terms_from_prompt(prompt)
        core_tokens = subject_terms_from_answer_core(extract_claims(answer).core_answer)
        quoted_phrases = _quoted_phrases(prompt)
        core_entities = _entity_phrases(extract_claims(answer).core_answer)
        max_bm25 = max((max(0.0, snippet.score) for snippet in snippets), default=0.0)
        ranked: list[tuple[float, EvidenceSnippet]] = []
        for snippet in snippets:
            diag = _snippet_alignment_diag(snippet, prompt_tokens, core_tokens, quoted_phrases, core_entities)
            reject_reason = _raw_candidate_rejection_reason(diag)
            if reject_reason:
                continue
            bm25_norm = max(0.0, snippet.score) / max(1.0, max_bm25)
            rank = (
                0.30 * diag["title_subject_overlap_ratio"]
                + 0.25 * diag["prompt_subject_overlap_ratio"]
                + 0.15 * diag["answer_core_overlap_ratio"]
                + 0.15 * float(diag["quoted_phrase_hit"])
                + 0.10 * float(diag["answer_core_entity_hit"] and diag["prompt_subject_overlap"] >= 1)
                + 0.05 * bm25_norm
            )
            if _weak_source(snippet.source) and not (diag["title_subject_overlap"] or diag["quoted_phrase_hit"]):
                rank -= 0.10
            if rank >= 0.08:
                ranked.append((rank, snippet))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidenceSnippet(id=snippet.id, title=snippet.title, text=snippet.text, source=snippet.source, score=float(rank))
            for rank, snippet in ranked
        ]

    def _rerank_v2(self, prompt: str, answer: str, snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        prompt_tokens = content_tokens(prompt)
        answer_tokens = content_tokens(answer)
        core_tokens = content_tokens(extract_claims(answer).core_answer)
        ranked: list[tuple[float, EvidenceSnippet]] = []
        for snippet in snippets:
            text_tokens = content_tokens(f"{snippet.title} {snippet.text}")
            prompt_overlap = len(prompt_tokens & text_tokens) / max(1, len(prompt_tokens))
            answer_overlap = len(answer_tokens & text_tokens) / max(1, len(answer_tokens))
            core_overlap = len(core_tokens & text_tokens) / max(1, len(core_tokens))
            typed_bonus = 0.0
            claims = extract_claims(answer)
            lower_text = snippet.text.lower()
            if any(entity.lower() in lower_text for entity in claims.entities):
                typed_bonus += 0.25
            snippet_numbers = {value.rstrip("%").replace(",", ".") for value in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", snippet.text)}
            if set(claims.numbers) & snippet_numbers:
                typed_bonus += 0.25
            rank = 0.35 * prompt_overlap + 0.25 * core_overlap + 0.20 * answer_overlap + typed_bonus + 0.02 * max(0.0, snippet.score)
            if prompt_overlap >= 0.08 or core_overlap >= 0.12 or typed_bonus > 0.0:
                ranked.append((rank, snippet))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidenceSnippet(id=snippet.id, title=snippet.title, text=snippet.text, source=snippet.source, score=float(rank))
            for rank, snippet in ranked
        ]

    def _read_meta(self, key: str) -> str:
        try:
            with sqlite3.connect(self.db_path) as con:
                row = con.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error:
            return "unknown"
        return str(row[0]) if row else "unknown"

    def _cached_snippets(self, key: str) -> list[EvidenceSnippet] | None:
        try:
            with sqlite3.connect(self.db_path) as con:
                row = con.execute("SELECT snippet_ids FROM evidence_cache WHERE key = ?", (key,)).fetchone()
                if row is None:
                    return None
                snippet_ids = json.loads(str(row[0] or "[]"))
                if not snippet_ids:
                    return []
                placeholders = ",".join("?" for _ in snippet_ids)
                rows = con.execute(
                    f"SELECT id, title, text, source FROM snippets WHERE id IN ({placeholders})",
                    tuple(snippet_ids),
                ).fetchall()
        except (sqlite3.Error, json.JSONDecodeError):
            return None
        by_id = {
            int(row[0]): EvidenceSnippet(id=int(row[0]), title=str(row[1] or ""), text=str(row[2] or ""), source=str(row[3] or ""), score=0.0)
            for row in rows
        }
        return [by_id[item_id] for item_id in snippet_ids if item_id in by_id]

    def _store_cache(self, key: str, prompt: str, answer: str, query: str, snippet_ids: list[int]) -> None:
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute(
                    """
                    INSERT OR REPLACE INTO evidence_cache(key, prompt, answer, query, snippet_ids)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (key, prompt, answer, query, json.dumps(snippet_ids)),
                )
                con.commit()
        except sqlite3.Error:
            return


def verify_against_evidence(
    prompt: str,
    answer: str,
    snippets: list[EvidenceSnippet],
    *,
    retrieval_latency_sec: float = 0.0,
    alignment_version: str = RETRIEVER_VERSION,
    retriever_version: str | None = None,
    kb_version: str | None = None,
    queries: list[str] | None = None,
) -> EvidenceAudit:
    extractor = FeatureExtractor()
    profile = extractor._question_profile(prompt)
    typed = profile in FeatureExtractor.TYPED_SHORT_PROFILES or profile == "which_list"
    extracted = extract_claims(answer)
    aligned = _aligned_evidence(prompt, extracted.core_answer, snippets, alignment_version=alignment_version)
    if alignment_version in {"fts5_rules_v5", "fts5_rules_v6"}:
        feature_snippets = list(aligned["feature_snippets"])
        overlay_snippets = list(aligned["overlay_eligible_snippets"])
        evidence_snippets = feature_snippets
        top_overlap_snippets = feature_snippets or overlay_snippets
    else:
        feature_snippets = snippets
        overlay_snippets = list(aligned["snippets"])
        evidence_snippets = snippets
        top_overlap_snippets = snippets
    evidence_text = " ".join(snippet.text for snippet in evidence_snippets)
    evidence_tokens = content_tokens(evidence_text)
    evidence_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", evidence_text))
    evidence_numbers = {value.rstrip("%").replace(",", ".") for value in evidence_numbers}
    evidence_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", evidence_text))
    evidence_lower = evidence_text.lower()
    prompt_tokens = content_tokens(prompt)
    aligned_text = " ".join(snippet.text for snippet in overlay_snippets)
    aligned_numbers = {value.rstrip("%").replace(",", ".") for value in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", aligned_text)}
    aligned_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", aligned_text))
    prompt_numbers = {value.rstrip("%").replace(",", ".") for value in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", prompt)}
    prompt_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", prompt))
    answer_year_in_prompt = bool(set(extracted.years) & prompt_years)
    answer_number_in_prompt = bool(set(extracted.numbers) & prompt_numbers)
    answer_years_for_refute = set(extracted.years) - prompt_years
    answer_numbers_for_refute = set(extracted.numbers) - prompt_numbers
    aligned_year_refuted_count = float(len(answer_years_for_refute - aligned_years) if aligned_years else 0)
    aligned_number_refuted_count = float(len(answer_numbers_for_refute - aligned_numbers) if aligned_numbers else 0)
    relation_validation: dict[str, Any] = {
        "relation_conflicts": [],
        "rejected_relation_candidates": [],
        "answer_refute_values": {
            "years": sorted(answer_years_for_refute),
            "numbers": sorted(answer_numbers_for_refute),
        },
        "prompt_relation_kind": relation_kind_from_prompt(prompt),
        "relation_validation_version": "legacy",
    }
    if alignment_version == "fts5_rules_v6":
        relation_validation = relation_refute_validation(
            prompt,
            answer,
            overlay_snippets,
            answer_years=answer_years_for_refute,
            answer_numbers=answer_numbers_for_refute,
            prompt_years=prompt_years,
            prompt_numbers=prompt_numbers,
            expected_kind=_expected_answer_kind_for_retrieval(prompt),
        )
        aligned_year_refuted_count = float(len([item for item in relation_validation["relation_conflicts"] if item.get("kind") == "year"]))
        aligned_number_refuted_count = float(len([item for item in relation_validation["relation_conflicts"] if item.get("kind") == "count"]))

    top_overlap = 0.0
    if top_overlap_snippets:
        top_overlap = len((content_tokens(top_overlap_snippets[0].text) & (content_tokens(answer) | prompt_tokens))) / max(1, len(content_tokens(answer) | prompt_tokens))

    verdicts: list[ClaimVerdict] = []
    for claim in extracted.claims:
        claim_tokens = content_tokens(claim.text)
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        entity_hits = sum(1 for entity in claim.entities if entity.lower() in evidence_lower)
        number_hits = sum(1 for value in claim.numbers if value in evidence_numbers)
        year_hits = sum(1 for value in claim.years if value in evidence_years)
        has_refuted_number = bool((set(claim.numbers) - evidence_numbers) and evidence_numbers and overlap >= 0.25)
        has_refuted_year = bool((set(claim.years) - evidence_years) and evidence_years and overlap >= 0.25)
        if not evidence_snippets:
            verdicts.append(ClaimVerdict(claim.text, claim.is_core, "unknown", "no_evidence"))
        elif has_refuted_year or (profile == "count" and has_refuted_number):
            verdicts.append(ClaimVerdict(claim.text, claim.is_core, "refuted", "numeric_conflict"))
        elif overlap >= 0.45 or entity_hits or number_hits or year_hits:
            verdicts.append(ClaimVerdict(claim.text, claim.is_core, "supported", "lexical_or_typed_match"))
        elif claim.entities or claim.numbers or claim.years:
            verdicts.append(ClaimVerdict(claim.text, claim.is_core, "unknown", "fact_not_found"))
        else:
            verdicts.append(ClaimVerdict(claim.text, claim.is_core, "not_checkable", "weak_fact_shape"))

    supported = [v for v in verdicts if v.verdict == "supported"]
    refuted = [v for v in verdicts if v.verdict == "refuted"]
    unknown = [v for v in verdicts if v.verdict == "unknown"]
    core_supported = float(any(v.is_core and v.verdict == "supported" for v in verdicts))
    core_refuted = float(any(v.is_core and v.verdict == "refuted" for v in verdicts))

    entity_hits = sum(1 for entity in extracted.entities if entity.lower() in evidence_lower)
    number_hits = sum(1 for number in extracted.numbers if number in evidence_numbers)
    year_hits = sum(1 for year in extracted.years if year in evidence_years)
    entity_total = len(extracted.entities)
    number_total = len(extracted.numbers)
    year_total = len(extracted.years)
    tail_claims = [claim for claim in extracted.claims if not claim.is_core]
    tail_unknown_entities = sum(len(claim.entities) for claim in tail_claims if not any(entity.lower() in evidence_lower for entity in claim.entities))
    tail_unknown_numbers = sum(len(claim.numbers) for claim in tail_claims if not any(number in evidence_numbers for number in claim.numbers))

    top_diagnostics = _snippet_diagnostics(prompt, extracted.core_answer, snippets[:8])
    compact_aligned_snippets = feature_snippets if alignment_version in {"fts5_rules_v5", "fts5_rules_v6"} else overlay_snippets
    compact = {
        "profile": profile,
        "retriever_version": retriever_version or alignment_version,
        "kb_version": kb_version or "unknown",
        "queries": queries or [],
        "raw_snippets": [
            {
                "id": s.id,
                "title": s.title,
                "text": s.text[:450],
                "source": s.source,
                "score": s.score,
                "diagnostics": top_diagnostics.get(s.id, {}),
            }
            for s in snippets[:8]
        ],
        "snippets": [
            {
                "id": s.id,
                "title": s.title,
                "text": s.text[:450],
                "source": s.source,
                "score": s.score,
                "diagnostics": top_diagnostics.get(s.id, {}),
            }
            for s in snippets[:3]
        ],
        "aligned_snippets": [
            {
                "id": s.id,
                "title": s.title,
                "text": s.text[:450],
                "source": s.source,
                "score": s.score,
                "alignment_reason": aligned["reasons"].get(s.id, "unknown"),
                "alignment_tier": aligned.get("tiers", {}).get(s.id, "aligned"),
            }
            for s in compact_aligned_snippets[:3]
        ],
        "feature_snippets": [
            {
                "id": s.id,
                "title": s.title,
                "text": s.text[:450],
                "source": s.source,
                "score": s.score,
                "alignment_reason": aligned["reasons"].get(s.id, "unknown"),
                "alignment_tier": aligned.get("tiers", {}).get(s.id, "feature"),
            }
            for s in feature_snippets[:8]
        ],
        "overlay_eligible_snippets": [
            {
                "id": s.id,
                "title": s.title,
                "text": s.text[:450],
                "source": s.source,
                "score": s.score,
                "alignment_reason": aligned["reasons"].get(s.id, "unknown"),
                "alignment_tier": aligned.get("tiers", {}).get(s.id, "strong"),
            }
            for s in overlay_snippets[:8]
        ],
        "weak_aligned_snippets": [
            {
                "id": s.id,
                "title": s.title,
                "text": s.text[:450],
                "source": s.source,
                "score": s.score,
                "alignment_reason": aligned["reasons"].get(s.id, "unknown"),
                "alignment_tier": "weak",
            }
            for s in aligned.get("weak_snippets", [])[:8]
        ],
        "rejected_snippets": aligned["rejected"][:8],
        "rejected_top_snippets": aligned["rejected"][:5],
        "claims": [v.__dict__ for v in verdicts[:8]],
        **relation_validation,
    }
    diag = diagnostic_columns(
        prompt,
        answer,
        json.dumps(compact, ensure_ascii=False),
        aligned_title_entity_match=float(aligned["feature_title_entity_match"] if alignment_version == "fts5_rules_v5" else aligned["title_entity_match"]),
        top_overlap=float(top_overlap),
    )
    status = "ok" if snippets else "no_evidence"
    return EvidenceAudit(
        status=status,
        retrieval_latency_sec=float(retrieval_latency_sec),
        retrieval_hit_count=len(snippets),
        top_bm25_score=float(snippets[0].score if snippets else 0.0),
        top_evidence_overlap=float(top_overlap),
        core_supported=core_supported,
        core_refuted=core_refuted,
        entity_supported_ratio=float(entity_hits / max(1, entity_total)),
        entity_refuted_count=float(max(0, entity_total - entity_hits) if evidence_snippets and entity_total else 0),
        number_supported_ratio=float(number_hits / max(1, number_total)),
        number_refuted_count=float(max(0, number_total - number_hits) if evidence_snippets and number_total and profile == "count" else 0),
        year_supported_ratio=float(year_hits / max(1, year_total)),
        year_refuted_count=float(max(0, year_total - year_hits) if evidence_snippets and year_total and profile == "when" else 0),
        claim_supported_count=float(len(supported)),
        claim_refuted_count=float(len(refuted)),
        claim_unknown_count=float(len(unknown)),
        tail_unsupported_entity_count=float(tail_unknown_entities),
        tail_unsupported_number_count=float(tail_unknown_numbers),
        evidence_missing_for_typed_question=float(typed and not snippets),
        answer_entity_not_in_evidence_ratio=float(max(0, entity_total - entity_hits) / max(1, entity_total)),
        answer_number_not_in_evidence_ratio=float(max(0, number_total - number_hits) / max(1, number_total)),
        aligned_hit_count=float(len(overlay_snippets)),
        aligned_year_refuted_count=aligned_year_refuted_count,
        aligned_number_refuted_count=aligned_number_refuted_count,
        answer_year_in_prompt=float(answer_year_in_prompt),
        answer_number_in_prompt=float(answer_number_in_prompt),
        aligned_title_entity_match=float(aligned["title_entity_match"]),
        answer_core_entity_count=float(diag["evidence_answer_core_entity_count"]),
        answer_core_entity_in_aligned_evidence_count=float(diag["evidence_answer_core_entity_in_aligned_evidence_count"]),
        answer_core_entity_missing_from_aligned_evidence_count=float(diag["evidence_answer_core_entity_missing_from_aligned_evidence_count"]),
        aligned_evidence_entity_count=float(diag["evidence_aligned_evidence_entity_count"]),
        aligned_alternative_entity_count=float(diag["evidence_aligned_alternative_entity_count"]),
        aligned_entity_mismatch_candidate=float(diag["evidence_aligned_entity_mismatch_candidate"]),
        aligned_entity_confidence=float(diag["evidence_aligned_entity_confidence"]),
        compact_json=json.dumps(compact, ensure_ascii=False),
    )


def _aligned_evidence(prompt: str, core_answer: str, snippets: list[EvidenceSnippet], *, alignment_version: str = RETRIEVER_VERSION) -> dict[str, Any]:
    if alignment_version in {"fts5_rules_v5", "fts5_rules_v6"}:
        return _tiered_evidence_alignment(prompt, core_answer, snippets)
    if alignment_version not in {"fts5_rules_v3", "fts5_rules_v4"}:
        prompt_tokens = _subject_tokens_v2(prompt)
        core_tokens = _subject_tokens_v2(core_answer)
        subject_tokens = prompt_tokens | core_tokens
        core_entities = [entity.lower() for entity in re.findall(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:[\s\-][А-ЯЁA-Z][а-яёa-z]+){0,4}\b", core_answer)]
        aligned: list[EvidenceSnippet] = []
        title_entity_match = False
        for snippet in snippets:
            title_tokens = _subject_tokens_v2(snippet.title)
            text_tokens = _subject_tokens_v2(snippet.text)
            combined_lower = f"{snippet.title} {snippet.text}".lower()
            title_overlap = bool(subject_tokens & title_tokens)
            text_overlap = len(subject_tokens & text_tokens) >= 2
            core_entity_hit = any(entity and entity in combined_lower for entity in core_entities)
            if title_overlap or text_overlap or core_entity_hit:
                aligned.append(snippet)
                title_entity_match = title_entity_match or title_overlap or core_entity_hit
        return {
            "snippets": aligned,
            "feature_snippets": aligned,
            "overlay_eligible_snippets": aligned,
            "weak_snippets": [],
            "title_entity_match": title_entity_match,
            "feature_title_entity_match": title_entity_match,
            "reasons": {},
            "tiers": {},
            "rejected": [],
        }

    prompt_tokens = subject_terms_from_prompt(prompt)
    core_tokens = subject_terms_from_answer_core(core_answer)
    quoted_phrases = _quoted_phrases(prompt)
    core_entities = _entity_phrases(core_answer)
    aligned: list[EvidenceSnippet] = []
    title_entity_match = False
    reasons: dict[int, str] = {}
    rejected: list[dict[str, Any]] = []
    for snippet in snippets:
        diag = _snippet_alignment_diag(snippet, prompt_tokens, core_tokens, quoted_phrases, core_entities)
        reason = ""
        if diag["numeric_only_match"]:
            reason = ""
        elif diag["generic_only_match"]:
            reason = ""
        elif diag["title_subject_overlap"] >= 1:
            reason = "title_subject_overlap"
        elif diag["quoted_phrase_hit"]:
            reason = "quoted_phrase_hit"
        elif diag["answer_core_entity_hit"] and diag["prompt_subject_overlap"] >= 1:
            reason = "answer_core_entity_plus_subject"
        elif diag["text_subject_overlap"] >= 3:
            reason = "text_subject_overlap_3"
        if reason and not (diag["title_subject_overlap"] == 0 and not diag["answer_core_entity_hit"] and diag["text_subject_overlap"] < 3):
            aligned.append(snippet)
            reasons[snippet.id] = reason
            title_entity_match = title_entity_match or bool(diag["title_subject_overlap"] or diag["answer_core_entity_hit"])
        else:
            rejected.append({"id": snippet.id, "title": snippet.title, "reason": _rejection_reason(diag), "diagnostics": diag})
    return {
        "snippets": aligned,
        "feature_snippets": aligned,
        "overlay_eligible_snippets": aligned,
        "weak_snippets": [],
        "title_entity_match": title_entity_match,
        "feature_title_entity_match": title_entity_match,
        "reasons": reasons,
        "tiers": {snippet.id: "aligned" for snippet in aligned},
        "rejected": rejected,
    }


RELATION_PREDICATES: dict[str, tuple[str, ...]] = {
    "born": ("родил", "born", "birth", "рожд"),
    "died": ("умер", "умерл", "сконч", "died", "death"),
    "founded": ("основан", "основал", "основана", "founded", "established", "создан", "создал"),
    "released": ("вышел", "выпущ", "релиз", "released", "premiered"),
    "published": ("опублик", "издан", "published"),
    "held": ("прошел", "прошла", "прошло", "прошли", "состоя", "held", "took place"),
    "appointed": ("назнач", "appointed"),
    "awarded": ("получ", "награжд", "awarded", "won"),
    "count": ("имеет", "насчиты", "составля", "количество", "число", "has", "have", "with", "number"),
}

GENERIC_COUNT_UNITS = {
    "сколько",
    "how",
    "many",
    "much",
    "number",
    "count",
    "количество",
    "число",
    "есть",
    "имеет",
    "have",
    "has",
    "does",
    "did",
    "составил",
    "составила",
    "составили",
}

RELATION_ANCHOR_GENERIC = {"роман", "novel", "фильм", "film", "спектакль", "play"}

DATE_NUMBER_UNITS = {
    "год",
    "году",
    "года",
    "лет",
    "year",
    "years",
    "date",
    "дата",
    "месяц",
    "месяце",
    "month",
    "page",
    "страница",
    "season",
    "сезон",
    "version",
    "версия",
}


def relation_kind_from_prompt(prompt: str) -> str:
    lower = str(prompt or "").lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("born", ("родился", "родилась", "родился", "born")),
        ("died", ("умер", "умерла", "скончался", "скончалась", "died")),
        ("founded", ("основан", "основана", "основал", "founded", "established")),
        ("released", ("вышел", "вышла", "релиз", "released", "premiered")),
        ("published", ("опубликован", "опубликована", "издан", "published")),
        ("held", ("прошел", "прошла", "состоялся", "состоялась", "премьера", "held", "take place", "premiere")),
        ("appointed", ("назначен", "назначена", "appointed")),
        ("awarded", ("получил", "получила", "награжден", "awarded", "won")),
    )
    if any(term in lower for term in ("сколько", "how many", "how much")):
        return "count"
    for kind, terms in rules:
        if any(term in lower for term in terms):
            return kind
    return "unknown"


def split_relation_windows(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    windows = [part.strip() for part in re.split(r"(?<=[.!?。])\s+|[;]\s+", cleaned) if part.strip()]
    return windows or [cleaned]


def numeric_values_in_window(window: str) -> dict[str, list[str]]:
    numbers = [value.rstrip("%").replace(",", ".") for value in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", window or "")]
    years = re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", window or "")
    generic_numbers = [value for value in numbers if value not in set(years)]
    return {"years": list(dict.fromkeys(years)), "numbers": list(dict.fromkeys(generic_numbers))}


def subject_anchor_hit(prompt: str, title: str, window: str) -> bool:
    title_lower = str(title or "").lower()
    window_lower = str(window or "").lower()
    combined = f"{title_lower} {window_lower}"
    quoted = _quoted_phrases(prompt)
    if any(phrase and (phrase.lower() in title_lower or phrase.lower() in window_lower) for phrase in quoted):
        return True
    if quoted:
        return False
    entities = _entity_phrases(prompt)
    for entity in entities:
        entity_tokens = [token for token in _query_tokens(entity) if not is_generic_subject_token(token)]
        if len(entity_tokens) < 2 and not any(phrase and phrase.lower() == entity.lower() for phrase in _quoted_phrases(prompt)):
            continue
        if entity and (entity.lower() in title_lower or entity.lower() in window_lower):
            return True
    prompt_tokens = subject_terms_from_prompt(prompt) - RELATION_ANCHOR_GENERIC
    title_tokens = _subject_tokens(title) - RELATION_ANCHOR_GENERIC
    window_tokens = _subject_tokens(window) - RELATION_ANCHOR_GENERIC
    return bool(len(prompt_tokens & title_tokens) >= 1 or len(prompt_tokens & window_tokens) >= 2 or len(prompt_tokens & (title_tokens | window_tokens)) >= 2 and combined)


def predicate_hit(kind: str, window: str) -> bool:
    if kind == "unknown":
        return False
    lower = str(window or "").lower()
    return any(term in lower for term in RELATION_PREDICATES.get(kind, ()))


def count_unit_from_prompt(prompt: str, answer: str) -> list[str]:
    prompt_tokens = _query_tokens(prompt)
    answer_numbers = {value.rstrip("%").replace(",", ".") for value in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", answer or "")}
    candidate_tokens: list[str] = []
    for idx, token in enumerate(prompt_tokens):
        if token in {"сколько", "many", "much"}:
            for following in prompt_tokens[idx + 1 : idx + 5]:
                if following in {"у", "of", "does", "do", "did", "is", "are", "was", "were", "doesn", "составил", "составила", "составили"}:
                    break
                candidate_tokens.append(following)
            break
    if not candidate_tokens:
        candidate_tokens = prompt_tokens
    units: list[str] = []
    for token in candidate_tokens:
        if token in GENERIC_COUNT_UNITS or token in DATE_NUMBER_UNITS or token in answer_numbers or is_generic_subject_token(token):
            continue
        units.append(token)
    return list(dict.fromkeys(units[:4]))


def count_unit_hit(unit_terms: list[str], window: str) -> bool:
    if not unit_terms:
        return False
    lower_tokens = set(_query_tokens(window))
    for term in unit_terms:
        if term in lower_tokens:
            return True
        if len(term) >= 5 and any(token.startswith(term[:5]) or term.startswith(token[:5]) for token in lower_tokens):
            return True
    return False


def _value_context_unit_rejected(window: str) -> bool:
    tokens = set(_query_tokens(window))
    return bool(tokens & DATE_NUMBER_UNITS)


def _relation_candidate_record(
    *,
    kind: str,
    value: str,
    snippet: EvidenceSnippet,
    window: str,
    decision: str,
    reason: str,
    relation_kind: str,
    predicate: bool,
    unit_terms: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": value,
        "title": snippet.title,
        "snippet_id": snippet.id,
        "window": window[:500],
        "decision": decision,
        "reason": reason,
        "relation_kind": relation_kind,
        "predicate_hit": bool(predicate),
        "unit_terms": unit_terms or [],
    }


def relation_refute_validation(
    prompt: str,
    answer: str,
    overlay_snippets: list[EvidenceSnippet],
    *,
    answer_years: set[str],
    answer_numbers: set[str],
    prompt_years: set[str],
    prompt_numbers: set[str],
    expected_kind: str,
) -> dict[str, Any]:
    relation_kind = relation_kind_from_prompt(prompt)
    unit_terms = count_unit_from_prompt(prompt, answer)
    conflicts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    answer_year_supported = False
    answer_number_supported = False
    valid_windows: list[tuple[EvidenceSnippet, str, dict[str, list[str]], bool]] = []

    for snippet in overlay_snippets:
        for window in split_relation_windows(snippet.text):
            if not subject_anchor_hit(prompt, snippet.title, window):
                continue
            values = numeric_values_in_window(window)
            pred = predicate_hit(relation_kind, window)
            title_exact = any(phrase.lower() in str(snippet.title).lower() for phrase in _quoted_phrases(prompt) if phrase)
            valid_unknown_year = relation_kind == "unknown" and title_exact and len(values["years"]) == 1
            if pred or valid_unknown_year:
                if answer_years & set(values["years"]):
                    answer_year_supported = True
                if expected_kind == "count" and count_unit_hit(unit_terms, window) and answer_numbers & set(values["numbers"]):
                    answer_number_supported = True
            valid_windows.append((snippet, window, values, pred or valid_unknown_year))

    for snippet, window, values, relation_ok in valid_windows:
        year_values = [value for value in values["years"] if value not in answer_years and value not in prompt_years]
        number_values = [value for value in values["numbers"] if value not in answer_numbers and value not in prompt_numbers]
        pred = predicate_hit(relation_kind, window)
        title_exact = any(phrase.lower() in str(snippet.title).lower() for phrase in _quoted_phrases(prompt) if phrase)

        if expected_kind in {"year", "date_or_month"}:
            for value in year_values:
                reason = "accepted_relation_predicate" if pred else "accepted_exact_subject_single_year"
                decision = "accept_relation_conflict"
                if len(values["years"]) > 3 and not pred:
                    decision, reason = "reject_broad_list", "reject_many_dates"
                elif relation_kind == "unknown" and not (title_exact and len(values["years"]) == 1):
                    decision, reason = "reject_no_predicate", "reject_contextual_year"
                elif answer_year_supported:
                    decision, reason = "reject_answer_value_supported", "reject_answer_value_supported"
                elif not relation_ok:
                    decision, reason = "reject_no_predicate", "reject_no_predicate"
                record = _relation_candidate_record(
                    kind="year",
                    value=value,
                    snippet=snippet,
                    window=window,
                    decision=decision,
                    reason=reason,
                    relation_kind=relation_kind,
                    predicate=pred,
                )
                (conflicts if decision == "accept_relation_conflict" else rejected).append(record)

        if expected_kind == "count":
            compatible_unit = count_unit_hit(unit_terms, window)
            for value in number_values:
                decision = "accept_relation_conflict"
                reason = "accepted_compatible_count_unit"
                if len(values["numbers"]) + len(values["years"]) > 5:
                    decision, reason = "reject_broad_list", "reject_broad_list"
                elif _value_context_unit_rejected(window):
                    decision, reason = "reject_wrong_unit", "reject_wrong_unit"
                elif not compatible_unit:
                    decision, reason = "reject_wrong_unit", "reject_wrong_unit"
                elif answer_number_supported:
                    decision, reason = "reject_answer_value_supported", "reject_answer_value_supported"
                elif not subject_anchor_hit(prompt, snippet.title, window):
                    decision, reason = "reject_no_predicate", "reject_no_predicate"
                record = _relation_candidate_record(
                    kind="count",
                    value=value,
                    snippet=snippet,
                    window=window,
                    decision=decision,
                    reason=reason,
                    relation_kind=relation_kind,
                    predicate=pred,
                    unit_terms=unit_terms,
                )
                (conflicts if decision == "accept_relation_conflict" else rejected).append(record)

    return {
        "relation_conflicts": conflicts[:8],
        "rejected_relation_candidates": rejected[:16],
        "answer_refute_values": {
            "years": sorted(answer_years),
            "numbers": sorted(answer_numbers),
            "prompt_years": sorted(prompt_years),
            "prompt_numbers": sorted(prompt_numbers),
        },
        "prompt_relation_kind": relation_kind,
        "relation_validation_version": "relation_refute_v10",
    }


def _tiered_evidence_alignment(prompt: str, core_answer: str, snippets: list[EvidenceSnippet]) -> dict[str, Any]:
    prompt_tokens = subject_terms_from_prompt(prompt)
    core_tokens = subject_terms_from_answer_core(core_answer)
    quoted_phrases = _quoted_phrases(prompt)
    prompt_entities = _entity_phrases(prompt)
    core_entities = _entity_phrases(core_answer)
    expected_kind = _expected_answer_kind_for_retrieval(prompt)
    answer_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", core_answer))
    answer_numbers = {value.rstrip("%").replace(",", ".") for value in answer_numbers}
    answer_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", core_answer))

    feature: list[EvidenceSnippet] = []
    overlay: list[EvidenceSnippet] = []
    weak: list[EvidenceSnippet] = []
    rejected: list[dict[str, Any]] = []
    reasons: dict[int, str] = {}
    tiers: dict[int, str] = {}
    title_entity_match = False
    feature_title_entity_match = False

    for snippet in snippets:
        diag = _snippet_alignment_diag(snippet, prompt_tokens, core_tokens, quoted_phrases, core_entities)
        tier, reason = _alignment_tier_v5(
            snippet,
            diag,
            prompt_entities=prompt_entities,
            quoted_phrases=quoted_phrases,
            expected_kind=expected_kind,
            answer_numbers=answer_numbers,
            answer_years=answer_years,
        )
        if tier == "strong":
            overlay.append(snippet)
            feature.append(snippet)
            title_entity_match = title_entity_match or bool(diag["title_subject_overlap"] or diag["prompt_entity_title_hit"] or diag["quoted_phrase_title_hit"])
            feature_title_entity_match = feature_title_entity_match or bool(diag["title_subject_overlap"] or diag["answer_core_entity_hit"])
        elif tier == "medium":
            feature.append(snippet)
            feature_title_entity_match = feature_title_entity_match or bool(diag["title_subject_overlap"] or diag["answer_core_entity_hit"])
        elif tier == "weak":
            weak.append(snippet)
        else:
            rejected.append({"id": snippet.id, "title": snippet.title, "reason": reason, "diagnostics": diag})
            continue
        reasons[snippet.id] = reason
        tiers[snippet.id] = tier

    return {
        "snippets": overlay,
        "feature_snippets": feature,
        "overlay_eligible_snippets": overlay,
        "weak_snippets": weak,
        "title_entity_match": title_entity_match,
        "feature_title_entity_match": feature_title_entity_match,
        "reasons": reasons,
        "tiers": tiers,
        "rejected": rejected,
    }


def _subject_tokens(text: str) -> set[str]:
    return {token for token in content_tokens(text) if not is_generic_subject_token(token)}


def _subject_tokens_v2(text: str) -> set[str]:
    tokens = set(content_tokens(text))
    return {
        token
        for token in tokens
        if not re.fullmatch(r"\d+(?:[.,]\d+)?%?", token)
        and token
        not in {
            "год",
            "году",
            "года",
            "лет",
            "месяце",
            "месяц",
            "числа",
            "дата",
            "date",
            "year",
            "years",
            "month",
            "number",
            "many",
            "much",
        }
    }


def is_generic_subject_token(token: str) -> bool:
    value = str(token or "").strip().lower().strip("*_`-–—.,:;!?()[]{}\"'«»")
    if len(value) < 4:
        return True
    if re.fullmatch(r"\d+(?:[.,]\d+)?%?", value):
        return True
    if re.fullmatch(r"[*_`#>\-]+", value):
        return True
    return value in GENERIC_SUBJECT_TOKENS


def subject_terms_from_prompt(prompt: str) -> set[str]:
    terms = _subject_tokens(prompt)
    for phrase in _quoted_phrases(prompt) + _entity_phrases(prompt):
        terms.update(_subject_tokens(phrase))
    return terms


def subject_terms_from_answer_core(answer: str) -> set[str]:
    terms = _subject_tokens(answer)
    for phrase in _quoted_phrases(answer) + _entity_phrases(answer):
        terms.update(_subject_tokens(phrase))
    return terms


def query_terms_for_retrieval(prompt: str, answer: str) -> list[str]:
    claims = extract_claims(answer)
    prompt_subject = _ordered_subject_terms(prompt)
    core_subject = _ordered_subject_terms(claims.core_answer)
    prompt_phrases = _quoted_phrases(prompt) + _entity_phrases(prompt)
    core_entities = _quoted_phrases(claims.core_answer) + _entity_phrases(claims.core_answer)
    expected_kind = _expected_answer_kind_for_retrieval(prompt)
    candidates: list[str] = []

    for phrase in prompt_phrases[:3]:
        candidates.append(_join_query_parts([phrase]))
    if core_entities and prompt_subject:
        candidates.append(_join_query_parts([core_entities[0], *prompt_subject[:8]]))
    if len(prompt_subject) >= 2 or prompt_phrases:
        candidates.append(_join_query_parts(prompt_subject[:MAX_FTS_TOKENS]))
    kind_terms = {
        "year": ["год", "year", "родился", "основан"],
        "count": ["сколько", "число", "количество", "number"],
        "who": ["кто", "who"],
        "where": ["город", "страна", "where"],
    }.get(expected_kind, [])
    if prompt_subject and kind_terms:
        candidates.append(_join_query_parts([*prompt_subject[:9], *kind_terms[:3]]))
    if expected_kind in {"year", "count"} and claims.numbers:
        candidates.append(_join_query_parts([*prompt_subject[:8], *claims.numbers[:4]]))
    if len(prompt_subject) >= 3 and core_subject:
        candidates.append(_join_query_parts([*prompt_subject[:8], *core_subject[:4]]))

    cleaned: list[str] = []
    seen: set[str] = set()
    has_strong_subject = bool(prompt_phrases) or len(prompt_subject) >= 3
    for candidate in candidates:
        tokens = [token for token in _query_tokens(candidate) if token]
        non_generic = [token for token in tokens if not is_generic_subject_token(token)]
        has_phrase = any(phrase.lower() in candidate.lower() for phrase in prompt_phrases)
        if not has_phrase and len(non_generic) < 2:
            continue
        if not has_strong_subject and not has_phrase:
            continue
        text = " ".join(tokens[:MAX_FTS_TOKENS])
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned[:MAX_QUERY_CANDIDATES]


def query_terms_for_retrieval_v4(prompt: str, answer: str) -> list[str]:
    candidates = query_terms_for_retrieval(prompt, answer)
    prompt_subject = _ordered_subject_terms(prompt)
    prompt_phrases = _quoted_phrases(prompt) + _entity_phrases(prompt)
    for phrase in prompt_phrases[:3]:
        candidates.insert(0, phrase)
    if prompt_subject:
        candidates.append(_join_query_parts(prompt_subject[:MAX_FTS_TOKENS]))
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tokens = [token for token in _query_tokens(candidate) if token]
        non_generic = [token for token in tokens if not is_generic_subject_token(token)]
        has_phrase = any(phrase.lower() in candidate.lower() for phrase in prompt_phrases)
        if not has_phrase and len(non_generic) < 2:
            continue
        text = " ".join(tokens[:MAX_FTS_TOKENS])
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned[:MAX_QUERY_CANDIDATES]


def controlled_recall_query(prompt: str) -> str:
    terms = _ordered_subject_terms(prompt)
    if len(terms) < 2:
        return ""
    tokens = [token for token in terms if not is_generic_subject_token(token)][:10]
    if len(tokens) < 2:
        return ""
    return " ".join(tokens)


def _ordered_subject_terms(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for phrase in _quoted_phrases(text) + _entity_phrases(text):
        for token in _query_tokens(phrase):
            if not is_generic_subject_token(token) and token not in seen:
                seen.add(token)
                ordered.append(token)
    for token in content_tokens(text):
        if not is_generic_subject_token(token) and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _quoted_phrases(text: str) -> list[str]:
    phrases = re.findall(r"[«\"]([^»\"]{3,80})[»\"]|'([^']{3,80})'|“([^”]{3,80})”", text or "")
    flattened = [next(part for part in match if part).strip() for match in phrases if any(match)]
    return list(dict.fromkeys(flattened))


def _entity_phrases(text: str) -> list[str]:
    pattern = r"\b[А-ЯЁA-Z][а-яёa-zA-Z0-9]+(?:[\s\-][А-ЯЁA-Z0-9][а-яёa-zA-Z0-9]+){0,5}\b"
    phrases = [match.strip() for match in re.findall(pattern, text or "")]
    phrases.extend(match.strip() for match in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+){1,5}\b", text or ""))
    return [phrase for phrase in dict.fromkeys(phrases) if len(_subject_tokens(phrase)) >= 1]


def _query_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[\w]+", text or "", flags=re.UNICODE) if len(token) > 1]


def _join_query_parts(parts: Iterable[str]) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _expected_answer_kind_for_retrieval(prompt: str) -> str:
    lower = (prompt or "").lower()
    if "сколько" in lower or "how many" in lower or "how much" in lower:
        return "count"
    if "в каком году" in lower or "когда" in lower or "what year" in lower or "when" in lower:
        return "year"
    if "кто" in lower or "who" in lower:
        return "who"
    if "где" in lower or "в какой стране" in lower or "в каком городе" in lower or "where" in lower:
        return "where"
    return "generic"


def _snippet_alignment_diag(
    snippet: EvidenceSnippet,
    prompt_tokens: set[str],
    core_tokens: set[str],
    quoted_phrases: list[str],
    core_entities: list[str],
) -> dict[str, Any]:
    title_tokens = _subject_tokens(snippet.title)
    text_tokens = _subject_tokens(snippet.text)
    all_tokens = title_tokens | text_tokens
    combined_lower = f"{snippet.title} {snippet.text}".lower()
    title_lower = snippet.title.lower()
    text_lower = snippet.text.lower()
    prompt_overlap_tokens = prompt_tokens & all_tokens
    text_subject_overlap_tokens = prompt_tokens & text_tokens
    title_subject_overlap_tokens = prompt_tokens & title_tokens
    answer_core_overlap_tokens = core_tokens & all_tokens
    quoted_phrase_title_hit = any(phrase and phrase.lower() in title_lower for phrase in quoted_phrases)
    quoted_phrase_text_hit = any(phrase and phrase.lower() in text_lower for phrase in quoted_phrases)
    quoted_phrase_hit = quoted_phrase_title_hit or quoted_phrase_text_hit
    answer_core_entity_hit = any(entity and entity.lower() in combined_lower for entity in core_entities)
    answer_core_entity_title_hit = any(entity and entity.lower() in title_lower for entity in core_entities)
    prompt_entity_title_hit = any(entity and entity.lower() in title_lower for entity in _entity_phrases(snippet.title))
    overlap_tokens = prompt_overlap_tokens | answer_core_overlap_tokens
    generic_only = bool(overlap_tokens) and all(is_generic_subject_token(token) for token in overlap_tokens)
    numeric_only = bool(overlap_tokens) and all(re.fullmatch(r"\d+(?:[.,]\d+)?%?", token) for token in overlap_tokens)
    return {
        "prompt_subject_overlap": len(prompt_overlap_tokens),
        "text_subject_overlap": len(text_subject_overlap_tokens),
        "answer_core_overlap": len(answer_core_overlap_tokens),
        "title_subject_overlap": len(title_subject_overlap_tokens),
        "prompt_subject_overlap_ratio": len(prompt_overlap_tokens) / max(1, len(prompt_tokens)),
        "answer_core_overlap_ratio": len(answer_core_overlap_tokens) / max(1, len(core_tokens)),
        "title_subject_overlap_ratio": len(title_subject_overlap_tokens) / max(1, len(prompt_tokens)),
        "quoted_phrase_hit": quoted_phrase_hit,
        "quoted_phrase_title_hit": quoted_phrase_title_hit,
        "quoted_phrase_text_hit": quoted_phrase_text_hit,
        "answer_core_entity_hit": answer_core_entity_hit,
        "answer_core_entity_title_hit": answer_core_entity_title_hit,
        "prompt_entity_title_hit": prompt_entity_title_hit,
        "generic_only_match": generic_only,
        "numeric_only_match": numeric_only,
    }


def _alignment_tier_v5(
    snippet: EvidenceSnippet,
    diag: dict[str, Any],
    *,
    prompt_entities: list[str],
    quoted_phrases: list[str],
    expected_kind: str,
    answer_numbers: set[str],
    answer_years: set[str],
) -> tuple[str, str]:
    if diag["numeric_only_match"]:
        return "rejected", "numeric_only_overlap"
    if diag["generic_only_match"]:
        return "rejected", "generic_only_match"
    if diag["prompt_subject_overlap"] < 1 and diag["answer_core_overlap"] > 0:
        return "rejected", "answer_tail_only"

    combined_lower = f"{snippet.title} {snippet.text}".lower()
    title_lower = snippet.title.lower()
    text_numbers = {value.rstrip("%").replace(",", ".") for value in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", snippet.text)}
    text_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", snippet.text))
    title_numbers = {value.rstrip("%").replace(",", ".") for value in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", snippet.title)}
    title_years = set(re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", snippet.title))
    quoted_title_hit = any(phrase and phrase.lower() in title_lower for phrase in quoted_phrases)
    quoted_text_hit = any(phrase and phrase.lower() in combined_lower for phrase in quoted_phrases)
    prompt_entity_hit = any(entity and entity.lower() in combined_lower for entity in prompt_entities)
    prompt_entity_title_hit = any(entity and entity.lower() in title_lower for entity in prompt_entities)
    numeric_context = bool(text_years if expected_kind == "year" else text_numbers)

    if (answer_numbers & (text_numbers | title_numbers) or answer_years & (text_years | title_years)) and diag["prompt_subject_overlap"] == 0:
        return "rejected", "numeric_only_overlap"

    if quoted_title_hit:
        return "strong", "quoted_prompt_title_in_title"
    if prompt_entity_title_hit and diag["title_subject_overlap"] >= 1:
        return "strong", "prompt_entity_title_subject_overlap"
    if quoted_text_hit and diag["title_subject_overlap"] >= 1:
        return "strong", "quoted_phrase_text_with_title_subject"
    if prompt_entity_hit and diag["answer_core_entity_hit"] and diag["prompt_subject_overlap"] >= 1:
        return "strong", "prompt_entity_answer_core_entity_subject"
    if expected_kind in {"year", "count"} and diag["title_subject_overlap"] >= 1 and numeric_context:
        return "strong", "title_subject_with_answer_numeric_context"

    if diag["title_subject_overlap"] >= 1 and diag["text_subject_overlap"] >= 2:
        return "medium", "title_subject_text_subject_overlap"
    if diag["text_subject_overlap"] >= 4:
        return "medium", "text_subject_overlap_4"
    if quoted_text_hit:
        return "medium", "quoted_phrase_text_only"

    if 2 <= diag["text_subject_overlap"] <= 3:
        return "weak", "weak_text_subject_overlap"
    if diag["answer_core_entity_hit"] and diag["prompt_subject_overlap"] < 1:
        return "weak", "answer_core_entity_without_subject"
    if diag["title_subject_overlap"] >= 1 or diag["text_subject_overlap"] >= 1:
        return "weak", "partial_subject_overlap"
    return "rejected", "unrelated_title_or_text"


def _candidate_passes_precision_filter(diag: dict[str, Any]) -> bool:
    if diag["numeric_only_match"] or diag["generic_only_match"]:
        return False
    return bool(
        diag["title_subject_overlap"] >= 1
        or diag["text_subject_overlap"] >= 2
        or diag["quoted_phrase_hit"]
        or (diag["answer_core_entity_hit"] and diag["prompt_subject_overlap"] >= 1)
    )


def _raw_candidate_rejection_reason(diag: dict[str, Any]) -> str:
    if diag["numeric_only_match"]:
        return "numeric_only_overlap"
    if diag["generic_only_match"]:
        return "generic_only_match"
    if diag["prompt_subject_overlap"] < 1 and diag["answer_core_overlap"] > 0:
        return "answer_tail_only"
    if diag["title_subject_overlap"] == 0 and diag["text_subject_overlap"] < 1:
        return "title_unrelated_no_text_subject"
    return ""


def _snippet_diagnostics(prompt: str, core_answer: str, snippets: list[EvidenceSnippet]) -> dict[int, dict[str, Any]]:
    prompt_tokens = subject_terms_from_prompt(prompt)
    core_tokens = subject_terms_from_answer_core(core_answer)
    quoted_phrases = _quoted_phrases(prompt)
    core_entities = _entity_phrases(core_answer)
    return {
        snippet.id: _snippet_alignment_diag(snippet, prompt_tokens, core_tokens, quoted_phrases, core_entities)
        for snippet in snippets
    }


def _weak_source(source: str) -> bool:
    return str(source or "").lower() in {"manual_seed", "seed", "fever"}


def _rejection_reason(diag: dict[str, Any]) -> str:
    if diag["numeric_only_match"]:
        return "numeric_only_overlap"
    if diag["generic_only_match"]:
        return "generic_only_match"
    if diag["title_subject_overlap"] == 0 and diag["text_subject_overlap"] < 3:
        return "weak_subject_alignment"
    return "below_alignment_threshold"


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY,
            source TEXT,
            title TEXT,
            url TEXT,
            lang TEXT,
            text TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS snippets(
            id INTEGER PRIMARY KEY,
            document_id INTEGER,
            title TEXT,
            text TEXT NOT NULL,
            source TEXT,
            lang TEXT,
            text_hash TEXT UNIQUE
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS snippet_fts USING fts5(
            title,
            text,
            content='snippets',
            content_rowid='id',
            tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS evidence_cache(
            key TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            answer TEXT NOT NULL,
            query TEXT,
            snippet_ids TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    con.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)", (SCHEMA_VERSION,))


def build_kb(
    source_paths: Iterable[str | Path],
    db_path: str | Path,
    *,
    kb_version: str = "local_v1",
    min_chars: int = 150,
    max_chars: int = 900,
) -> dict[str, int]:
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    document_count = 0
    snippet_count = 0
    with sqlite3.connect(target) as con:
        create_schema(con)
        con.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('kb_version', ?)", (kb_version,))
        for path in source_paths:
            for record in _iter_source_records(Path(path)):
                doc_id = _insert_document(con, record)
                document_count += 1
                for snippet in _chunk_text(str(record.get("text", "")), min_chars=min_chars, max_chars=max_chars):
                    text_hash = sha256_hexdigest(record.get("source"), record.get("title"), snippet)
                    cur = con.execute(
                        """
                        INSERT OR IGNORE INTO snippets(document_id, title, text, source, lang, text_hash)
                        VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            str(record.get("title", "")),
                            snippet,
                            str(record.get("source", "")),
                            str(record.get("lang", "")),
                            text_hash,
                        ),
                    )
                    if cur.rowcount:
                        row_id = int(cur.lastrowid)
                        con.execute(
                            "INSERT INTO snippet_fts(rowid, title, text) VALUES(?, ?, ?)",
                            (row_id, str(record.get("title", "")), snippet),
                        )
                        snippet_count += 1
        con.commit()
    return {"documents": document_count, "snippets": snippet_count}


def _insert_document(con: sqlite3.Connection, record: dict[str, Any]) -> int:
    cur = con.execute(
        "INSERT INTO documents(source, title, url, lang, text) VALUES(?, ?, ?, ?, ?)",
        (
            str(record.get("source", "")),
            str(record.get("title", "")),
            str(record.get("url", "")),
            str(record.get("lang", "")),
            str(record.get("text", "")),
        ),
    )
    return int(cur.lastrowid)


def _iter_source_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield _record_to_document(record, path)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            forbidden = FORBIDDEN_SOURCE_COLUMNS & set(reader.fieldnames or [])
            if forbidden:
                raise ValueError(f"KB build source {path} contains forbidden columns: {sorted(forbidden)}")
            for row in reader:
                yield _record_to_document(row, path)


def _record_to_document(record: dict[str, Any], path: Path) -> dict[str, str]:
    forbidden = FORBIDDEN_SOURCE_COLUMNS & set(record)
    if forbidden:
        raise ValueError(f"KB build source {path} contains forbidden columns: {sorted(forbidden)}")
    prompt = str(record.get("prompt", "") or record.get("question", ""))
    answer = str(record.get("answer", "") or record.get("model_answer", ""))
    if str(record.get("label", "0")) not in {"0", "0.0", "False", "false"}:
        prompt = ""
        answer = ""
    text = str(record.get("text", "") or f"{prompt} {answer}").strip()
    return {
        "source": str(record.get("source", path.stem)),
        "title": str(record.get("title", prompt[:120])),
        "url": str(record.get("url", "")),
        "lang": str(record.get("lang", "")),
        "text": _clean_text(text),
    }


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, *, min_chars: int = 150, max_chars: int = 900) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and len(current) >= min_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _fts_query(text: str) -> str:
    tokens = _query_tokens(text)
    tokens = list(dict.fromkeys(tokens))[:MAX_FTS_TOKENS]
    return " OR ".join(f'"{token}"' for token in tokens)
