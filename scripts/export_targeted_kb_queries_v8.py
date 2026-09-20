from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.evidence import _entity_phrases, _quoted_phrases, _query_tokens, controlled_recall_query, is_generic_subject_token
from guardian_of_truth.utils import sha256_hexdigest

FORBIDDEN_EXPORT_FIELDS = {"is_hallucination", "correct_answer", "comment", "label"}
DEFAULT_INCLUDE_REASONS = {
    "kb_missing_relevant_doc",
    "query_too_weak",
    "answer_entity_misleading",
    "ambiguous_subject",
    "raw_noise_only",
}


def _clean_query(text: str) -> str:
    tokens = _query_tokens(text)[:12]
    if not tokens:
        return ""
    if all(re.fullmatch(r"\d+(?:[.,]\d+)?", token) for token in tokens):
        return ""
    if all(is_generic_subject_token(token) for token in tokens):
        return ""
    return " ".join(tokens)


def _latin_title_candidates(prompt: str) -> list[str]:
    pattern = r"\b[A-Z][A-Za-z0-9]+(?:[\s:\-][A-Z0-9][A-Za-z0-9]+){1,6}\b"
    return [match.strip() for match in re.findall(pattern, prompt or "")]


def queries_for_row(row: dict[str, Any], *, max_queries: int) -> list[dict[str, Any]]:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("model_answer", ""))
    row_key = str(row.get("row_key") or sha256_hexdigest(prompt, answer))
    recall = controlled_recall_query(prompt)
    candidates: list[tuple[int, str, str]] = []
    for phrase in _quoted_phrases(prompt):
        candidates.append((1, "quoted_prompt_title", phrase))
    for phrase in _entity_phrases(prompt):
        candidates.append((2, "prompt_entity", phrase))
    for phrase in _latin_title_candidates(prompt):
        candidates.append((3, "latin_title", phrase))
    if recall:
        candidates.append((4, "strong_subject", recall))
    for phrase in _entity_phrases(answer):
        if recall:
            candidates.append((5, "answer_core_secondary", f"{phrase} {recall}"))

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for priority, query_type, query in sorted(candidates, key=lambda item: item[0]):
        cleaned = _clean_query(query)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        records.append(
            {
                "row_key": row_key,
                "query": cleaned,
                "query_type": query_type,
                "priority": priority,
                "prompt": prompt,
                "model_answer": answer,
            }
        )
        if len(records) >= max_queries:
            break
    return records


def export_queries(
    kb_recall_csv: Path,
    output_jsonl: Path,
    *,
    max_queries_per_row: int,
    include_reasons: set[str] | None = None,
) -> dict[str, int]:
    frame = pd.read_csv(kb_recall_csv)
    if "failure_reason" in frame.columns:
        reasons = include_reasons or DEFAULT_INCLUDE_REASONS
        frame = frame[frame["failure_reason"].astype(str).isin(reasons)].copy()
    forbidden = FORBIDDEN_EXPORT_FIELDS & set(frame.columns)
    allowed_frame = frame.drop(columns=sorted(forbidden), errors="ignore")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    source_rows = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for _, row in allowed_frame.iterrows():
            source_rows += 1
            for record in queries_for_row(row.to_dict(), max_queries=max_queries_per_row):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1
    return {"source_rows": source_rows, "queries": rows_written}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export targeted KB fetch queries from v8 recall audit rows.")
    parser.add_argument("--kb-recall-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-queries-per-row", type=int, default=5)
    parser.add_argument(
        "--include-reason",
        action="append",
        default=None,
        help="Failure reason to export. Defaults to KB-missing/query-weak profiles when failure_reason exists.",
    )
    args = parser.parse_args()

    summary = export_queries(
        Path(args.kb_recall_csv),
        Path(args.output_jsonl),
        max_queries_per_row=args.max_queries_per_row,
        include_reasons=set(args.include_reason) if args.include_reason else None,
    )
    print(json.dumps({"output_jsonl": args.output_jsonl, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
