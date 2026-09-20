from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.evidence import (
    EvidenceSnippet,
    _entity_phrases,
    _quoted_phrases,
    _query_tokens,
    _subject_tokens,
    controlled_recall_query,
    subject_terms_from_prompt,
)
from guardian_of_truth.utils import sha256_hexdigest


def _json_loads(value: Any) -> dict[str, Any]:
    try:
        return json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}


def _titles_from_compact(value: Any) -> list[str]:
    payload = _json_loads(value)
    snippets = payload.get("raw_snippets") or payload.get("snippets") or []
    return [str(item.get("title", "")) for item in snippets[:5] if isinstance(item, dict)]


def _hit_count(row: pd.Series) -> int:
    for column in ("evidence_retrieval_hit_count", "evidence_aligned_hit_count"):
        try:
            value = row.get(column)
            if pd.notna(value):
                return int(float(value))
        except (TypeError, ValueError):
            pass
    return 0


def _strong_terms(prompt: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for phrase in _quoted_phrases(prompt) + _entity_phrases(prompt):
        for token in _query_tokens(phrase):
            if token not in seen and len(token) >= 4:
                seen.add(token)
                terms.append(token)
    for token in subject_terms_from_prompt(prompt):
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def _kb_candidates(db_path: Path, terms: list[str], *, top_n: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not db_path.exists() or not terms:
        return [], []
    fts_query = " OR ".join(f'"{token}"' for token in terms[:12])
    try:
        with sqlite3.connect(db_path) as con:
            title_rows = con.execute(
                """
                SELECT s.id, s.title, s.text, s.source, bm25(snippet_fts) AS score
                FROM snippet_fts
                JOIN snippets s ON s.id = snippet_fts.rowid
                WHERE snippet_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, top_n * 4),
            ).fetchall()
    except sqlite3.Error:
        return [], []
    term_set = set(terms)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row in title_rows:
        snippet = EvidenceSnippet(int(row[0]), str(row[1] or ""), str(row[2] or ""), str(row[3] or ""), float(-row[4]))
        title_overlap = len(_subject_tokens(snippet.title) & term_set)
        text_overlap = len(_subject_tokens(snippet.text) & term_set)
        record = {
            "id": snippet.id,
            "title": snippet.title,
            "source": snippet.source,
            "score": snippet.score,
            "title_overlap": title_overlap,
            "text_overlap": text_overlap,
        }
        candidates.append((title_overlap * 3 + text_overlap, record))
    candidates.sort(key=lambda item: (item[0], item[1]["score"]), reverse=True)
    title_candidates = [record for score, record in candidates if record["title_overlap"] > 0][:top_n]
    text_candidates = [record for score, record in candidates if record["text_overlap"] > 0][:top_n]
    return title_candidates, text_candidates


def _classify(row: pd.Series, baseline_row: pd.Series | None, title_candidates: list[dict[str, Any]], text_candidates: list[dict[str, Any]], terms: list[str]) -> str:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("model_answer", ""))
    v7_count = _hit_count(row)
    v2_count = _hit_count(baseline_row) if baseline_row is not None else 0
    has_kb_candidate = bool(title_candidates or text_candidates)
    prompt_entities = _quoted_phrases(prompt) + _entity_phrases(prompt)
    answer_entities = _entity_phrases(answer)
    if len(terms) < 2 and not prompt_entities:
        return "ambiguous_subject"
    if not has_kb_candidate:
        return "kb_missing_relevant_doc"
    if v7_count == 0 and has_kb_candidate:
        return "retriever_missed_existing_doc"
    if v2_count > 0 and not (_titles_from_compact(row.get("evidence_compact_json"))):
        return "raw_noise_only"
    if answer_entities and not prompt_entities and len(terms) < 3:
        return "answer_entity_misleading"
    if not controlled_recall_query(prompt):
        return "query_too_weak"
    return "covered_or_unclear"


def _suggest_queries(prompt: str, answer: str, *, limit: int = 5) -> list[str]:
    candidates: list[str] = []
    for phrase in _quoted_phrases(prompt) + _entity_phrases(prompt):
        candidates.append(phrase)
    recall = controlled_recall_query(prompt)
    if recall:
        candidates.append(recall)
    for phrase in _entity_phrases(answer):
        combined = f"{phrase} {recall}".strip()
        if recall and combined:
            candidates.append(combined)
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tokens = _query_tokens(candidate)[:12]
        if not tokens or all(re.fullmatch(r"\d+", token) for token in tokens):
            continue
        text = " ".join(tokens)
        if text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned[:limit]


def _nested_counts(frame: pd.DataFrame, left: str, right: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for (left_value, right_value), count in frame.groupby([left, right], dropna=False).size().items():
        counts.setdefault(str(left_value), {})[str(right_value)] = int(count)
    return counts


def analyze(scored_csv: Path, baseline_csv: Path, evidence_db_path: Path, *, top_n: int) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(scored_csv)
    baseline = pd.read_csv(baseline_csv) if baseline_csv.exists() else pd.DataFrame()
    baseline_by_key = {
        sha256_hexdigest(row.get("prompt", ""), row.get("model_answer", "")): row
        for _, row in baseline.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("model_answer", ""))
        key = sha256_hexdigest(prompt, answer)
        baseline_row = baseline_by_key.get(key)
        terms = _strong_terms(prompt)
        title_candidates, text_candidates = _kb_candidates(evidence_db_path, terms, top_n=5)
        reason = _classify(row, baseline_row, title_candidates, text_candidates, terms)
        rows.append(
            {
                "row_key": key,
                "prompt": prompt,
                "model_answer": answer,
                "is_hallucination": row.get("is_hallucination"),
                "profile": row.get("question_profile") or row.get("profile"),
                "score_path": row.get("score_path"),
                "audit_status": row.get("audit_status"),
                "v2_evidence_count": _hit_count(baseline_row) if baseline_row is not None else 0,
                "v2_top_titles": " | ".join(_titles_from_compact(baseline_row.get("evidence_compact_json")) if baseline_row is not None else []),
                "v7_evidence_count": _hit_count(row),
                "v7_top_titles": " | ".join(_titles_from_compact(row.get("evidence_compact_json"))),
                "subject_terms": " ".join(terms),
                "quoted_phrases_entities": " | ".join(_quoted_phrases(prompt) + _entity_phrases(prompt)),
                "best_kb_title_candidates": " | ".join(item["title"] for item in title_candidates),
                "best_kb_text_candidates": " | ".join(item["title"] for item in text_candidates),
                "failure_reason": reason,
                "suggested_fetch_queries": " | ".join(_suggest_queries(prompt, answer)),
            }
        )
    out = pd.DataFrame(rows)
    no_evidence = out[out["v7_evidence_count"] == 0]
    missing = out[out["failure_reason"] == "kb_missing_relevant_doc"]
    report = {
        "total_rows": int(len(out)),
        "no_evidence_under_v7": int(len(no_evidence)),
        "rows_where_kb_has_plausible_candidate": int(((out["best_kb_title_candidates"] != "") | (out["best_kb_text_candidates"] != "")).sum()),
        "rows_where_kb_appears_missing_relevant_candidate": int(len(missing)),
        "top_missing_subject_patterns": Counter(" ".join(str(value).split()[:4]) for value in missing["subject_terms"]).most_common(top_n),
        "top_missing_quoted_title_entities": Counter(
            entity for value in missing["quoted_phrases_entities"] for entity in str(value).split(" | ") if entity
        ).most_common(top_n),
        "failure_counts_by_profile": _nested_counts(out, "profile", "failure_reason"),
        "failure_counts_by_score_path": _nested_counts(out, "score_path", "failure_reason"),
        "failure_counts": out["failure_reason"].value_counts().astype(int).to_dict(),
    }
    if "is_hallucination" in out.columns:
        labels = pd.to_numeric(out["is_hallucination"], errors="coerce")
        report["false_negatives_by_failure_reason"] = out[labels == 1]["failure_reason"].value_counts().astype(int).to_dict()
        report["false_positives_by_failure_reason"] = out[labels == 0]["failure_reason"].value_counts().astype(int).to_dict()
    return report, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit KB recall failure modes for v8 retrieval work.")
    parser.add_argument("--scored-csv", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--evidence-db-path", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--top-n", type=int, default=300)
    args = parser.parse_args()

    report, rows = analyze(Path(args.scored_csv), Path(args.baseline_csv), Path(args.evidence_db_path), top_n=args.top_n)
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.to_csv(args.csv_output, index=False)


if __name__ == "__main__":
    main()
