from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from guardian_of_truth.claims import content_tokens, extract_claims, extract_entities
from guardian_of_truth.evidence import FORBIDDEN_SOURCE_COLUMNS
from guardian_of_truth.utils import iter_jsonl, sha256_hexdigest, write_jsonl


SEARCH_URL = "https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&srlimit={limit}"
EXTRACT_URL = (
    "https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts|info"
    "&exintro=1&explaintext=1&redirects=1&inprop=url&format=json&titles={title}"
)
DISAMBIG_HINTS = (
    "may refer to",
    "может означать",
    "значения",
    "disambiguation",
)


def _url_json(url: str, *, timeout: float = 8.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "guardian-of-truth-offline-kb/0.3"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            forbidden = FORBIDDEN_SOURCE_COLUMNS & set(reader.fieldnames or [])
            if forbidden:
                raise ValueError(f"Wikipedia fetch source {path} contains forbidden columns: {sorted(forbidden)}")
            for row in reader:
                yield dict(row)


def _keywords(text: str, limit: int = 8) -> list[str]:
    return sorted(content_tokens(text), key=lambda token: (-len(token), token))[:limit]


def build_queries(record: dict[str, Any]) -> list[dict[str, str]]:
    if record.get("query"):
        return [
            {
                "query_type": str(record.get("query_type", "targeted_query")),
                "query": str(record.get("query", "")).strip(),
            }
        ]
    prompt = str(record.get("prompt", "") or record.get("question", ""))
    answer = str(record.get("model_answer", "") or record.get("answer", ""))
    prompt_entities = list(extract_entities(prompt))
    answer_claims = extract_claims(answer)
    prompt_keywords = _keywords(prompt, 8)
    answer_entities = list(answer_claims.entities)
    answer_core_entities = list(extract_entities(answer_claims.core_answer))
    years_or_numbers = list(answer_claims.years or answer_claims.numbers)
    candidates = [
        ("prompt_entities_keywords", " ".join(prompt_entities[:4] + prompt_keywords[:6])),
        ("answer_core_prompt_keywords", " ".join(answer_core_entities[:3] + prompt_keywords[:6])),
        ("answer_entities_prompt_entities", " ".join(answer_entities[:4] + prompt_entities[:4])),
        ("prompt_keywords_answer_number", " ".join(prompt_keywords[:7] + years_or_numbers[:2])),
        ("prompt_token_fallback", " ".join(prompt_keywords[:8])),
    ]
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for kind, query in candidates:
        query = re.sub(r"\s+", " ", query).strip()
        if len(query) < 3 or query.lower() in seen:
            continue
        seen.add(query.lower())
        queries.append({"query_type": kind, "query": query})
    return queries


def _page_from_title(lang: str, title: str, *, timeout: float) -> dict[str, str] | None:
    url = EXTRACT_URL.format(lang=lang, title=urllib.parse.quote(title.replace(" ", "_")))
    payload = _url_json(url, timeout=timeout)
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = re.sub(r"\s+", " ", str(page.get("extract", ""))).strip()
        if len(extract) < 120:
            continue
        canonical_title = str(page.get("title", title)).strip()
        canonical_url = str(page.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(canonical_title.replace(' ', '_'))}"))
        return {"title": canonical_title, "url": canonical_url, "text": extract}
    return None


def _is_disambiguation(title: str, text: str) -> bool:
    lower = f"{title}\n{text[:500]}".lower()
    return any(hint in lower for hint in DISAMBIG_HINTS)


def fetch_wikipedia_rows(
    source_paths: list[Path],
    *,
    limit: int | None = None,
    row_limit: int | None = None,
    langs: list[str] | None = None,
    search_results: int = 5,
    pages_per_row: int = 3,
    sleep_sec: float = 0.05,
    timeout_sec: float = 8.0,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    seen_pages: set[str] = set()
    seen_texts: set[str] = set()
    scanned = 0
    langs = langs or ["ru", "en"]
    for path in source_paths:
        for record in _iter_records(path):
            scanned += 1
            if row_limit is not None and scanned > row_limit:
                return rows, diagnostics
            if limit is not None and len(rows) >= limit:
                return rows, diagnostics
            row_key = str(record.get("row_key") or sha256_hexdigest(record.get("prompt", ""), record.get("model_answer", "")))
            row_added = 0
            for query_item in build_queries(record):
                if row_added >= pages_per_row or (limit is not None and len(rows) >= limit):
                    break
                query = query_item["query"]
                for lang in langs:
                    if row_added >= pages_per_row or (limit is not None and len(rows) >= limit):
                        break
                    diag = {
                        "row_key": row_key,
                        "query": query,
                        "query_type": query_item["query_type"],
                        "lang": lang,
                        "search_hits": 0,
                        "pages_added": 0,
                        "skip_reason": "",
                    }
                    try:
                        search_url = SEARCH_URL.format(lang=lang, query=urllib.parse.quote(query), limit=search_results)
                        search = _url_json(search_url, timeout=timeout_sec)
                    except Exception as exc:
                        diag["skip_reason"] = f"search_error:{exc.__class__.__name__}"
                        diagnostics.append(diag)
                        continue
                    hits = search.get("query", {}).get("search", [])
                    diag["search_hits"] = len(hits)
                    page_candidates: list[dict[str, str]] = []
                    for item in hits:
                        title = str(item.get("title", "")).strip()
                        if not title:
                            continue
                        key = f"{lang}:{title.lower()}"
                        if key in seen_pages:
                            continue
                        try:
                            page = _page_from_title(lang, title, timeout=timeout_sec)
                        except Exception:
                            continue
                        if page is None:
                            continue
                        page_key = f"{lang}:{page['title'].lower()}"
                        text_hash = sha256_hexdigest(page["text"])
                        if page_key in seen_pages or text_hash in seen_texts:
                            continue
                        page_candidates.append(page | {"page_key": page_key, "text_hash": text_hash})
                    normal_pages = [page for page in page_candidates if not _is_disambiguation(page["title"], page["text"])]
                    selected = normal_pages or page_candidates
                    for page in selected:
                        if row_added >= pages_per_row:
                            break
                        seen_pages.add(page["page_key"])
                        seen_texts.add(page["text_hash"])
                        rows.append(
                            {
                                "row_key": row_key,
                                "prompt": str(record.get("prompt", "") or record.get("question", "")),
                                "answer": "",
                                "title": page["title"],
                                "text": page["text"],
                                "source": f"wikipedia:{lang}",
                                "url": page["url"],
                                "lang": lang,
                                "fetch_key": sha256_hexdigest(path, row_key, query, lang, page["title"]),
                            }
                        )
                        row_added += 1
                        diag["pages_added"] += 1
                        time.sleep(sleep_sec)
                    if diag["pages_added"] == 0 and not diag["skip_reason"]:
                        diag["skip_reason"] = "no_new_pages"
                    diagnostics.append(diag)
    return rows, diagnostics


def _load_seen_outputs(output: Path) -> tuple[set[str], set[str], int]:
    seen_pages: set[str] = set()
    seen_texts: set[str] = set()
    count = 0
    if not output.exists():
        return seen_pages, seen_texts, count
    for record in iter_jsonl(output):
        lang = str(record.get("lang", ""))
        title = str(record.get("title", "")).lower()
        text = str(record.get("text", ""))
        if lang and title:
            seen_pages.add(f"{lang}:{title}")
        if text:
            seen_texts.add(sha256_hexdigest(text))
        count += 1
    return seen_pages, seen_texts, count


def stream_wikipedia_rows(
    source_paths: list[Path],
    output: Path,
    diagnostics_output: Path,
    *,
    limit: int | None = None,
    row_limit: int | None = None,
    langs: list[str] | None = None,
    search_results: int = 5,
    pages_per_row: int = 3,
    sleep_sec: float = 0.05,
    timeout_sec: float = 8.0,
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    seen_pages, seen_texts, existing = _load_seen_outputs(output)
    rows_written = existing
    diagnostics_written = 0
    scanned = 0
    langs = langs or ["ru", "en"]
    mode = "a" if output.exists() else "w"
    diag_mode = "a" if diagnostics_output.exists() else "w"
    with output.open(mode, encoding="utf-8") as out_handle, diagnostics_output.open(diag_mode, encoding="utf-8") as diag_handle:
        for path in source_paths:
            for record in _iter_records(path):
                scanned += 1
                if row_limit is not None and scanned > row_limit:
                    return {"rows": rows_written, "diagnostics": diagnostics_written, "scanned": scanned - 1}
                if limit is not None and rows_written >= limit:
                    return {"rows": rows_written, "diagnostics": diagnostics_written, "scanned": scanned}
                row_key = str(record.get("row_key") or sha256_hexdigest(record.get("prompt", ""), record.get("model_answer", "")))
                row_added = 0
                for query_item in build_queries(record):
                    if row_added >= pages_per_row or (limit is not None and rows_written >= limit):
                        break
                    query = query_item["query"]
                    for lang in langs:
                        if row_added >= pages_per_row or (limit is not None and rows_written >= limit):
                            break
                        diag = {
                            "row_key": row_key,
                            "query": query,
                            "query_type": query_item["query_type"],
                            "lang": lang,
                            "search_hits": 0,
                            "pages_added": 0,
                            "skip_reason": "",
                        }
                        try:
                            search_url = SEARCH_URL.format(lang=lang, query=urllib.parse.quote(query), limit=search_results)
                            search = _url_json(search_url, timeout=timeout_sec)
                        except Exception as exc:
                            diag["skip_reason"] = f"search_error:{exc.__class__.__name__}"
                            diag_handle.write(json.dumps(diag, ensure_ascii=False) + "\n")
                            diagnostics_written += 1
                            continue
                        hits = search.get("query", {}).get("search", [])
                        diag["search_hits"] = len(hits)
                        page_candidates: list[dict[str, str]] = []
                        for item in hits:
                            title = str(item.get("title", "")).strip()
                            if not title:
                                continue
                            key = f"{lang}:{title.lower()}"
                            if key in seen_pages:
                                continue
                            try:
                                page = _page_from_title(lang, title, timeout=timeout_sec)
                            except Exception:
                                continue
                            if page is None:
                                continue
                            page_key = f"{lang}:{page['title'].lower()}"
                            text_hash = sha256_hexdigest(page["text"])
                            if page_key in seen_pages or text_hash in seen_texts:
                                continue
                            page_candidates.append(page | {"page_key": page_key, "text_hash": text_hash})
                        normal_pages = [page for page in page_candidates if not _is_disambiguation(page["title"], page["text"])]
                        selected = normal_pages or page_candidates
                        for page in selected:
                            if row_added >= pages_per_row or (limit is not None and rows_written >= limit):
                                break
                            seen_pages.add(page["page_key"])
                            seen_texts.add(page["text_hash"])
                            row = {
                                "row_key": row_key,
                                "prompt": str(record.get("prompt", "") or record.get("question", "")),
                                "answer": "",
                                "title": page["title"],
                                "text": page["text"],
                                "source": f"wikipedia:{lang}",
                                "url": page["url"],
                                "lang": lang,
                                "fetch_key": sha256_hexdigest(path, row_key, query, lang, page["title"]),
                            }
                            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out_handle.flush()
                            rows_written += 1
                            row_added += 1
                            diag["pages_added"] += 1
                            time.sleep(sleep_sec)
                        if diag["pages_added"] == 0 and not diag["skip_reason"]:
                            diag["skip_reason"] = "no_new_pages"
                        diag_handle.write(json.dumps(diag, ensure_ascii=False) + "\n")
                        diagnostics_written += 1
                        diag_handle.flush()
    return {"rows": rows_written, "diagnostics": diagnostics_written, "scanned": scanned}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch targeted Wikipedia extracts for unlabeled benchmark retrieval queries.")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--query-jsonl", action="append", default=[], help="JSONL generated by export_targeted_kb_queries_v8.py.")
    parser.add_argument("--output", default="data/external/wikipedia_targeted_v3.jsonl")
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--langs", default="ru,en")
    parser.add_argument("--search-results", type=int, default=5)
    parser.add_argument("--pages-per-row", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    args = parser.parse_args()

    output = Path(args.output)
    diagnostics_output = Path(args.diagnostics_output or str(output.with_suffix(".diagnostics.jsonl")))
    summary = stream_wikipedia_rows(
        [Path(path) for path in [*args.source, *args.query_jsonl]],
        output,
        diagnostics_output,
        limit=args.limit,
        row_limit=args.row_limit,
        langs=[lang.strip() for lang in args.langs.split(",") if lang.strip()],
        search_results=args.search_results,
        pages_per_row=args.pages_per_row,
        sleep_sec=args.sleep_sec,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({"output": args.output, "diagnostics_output": str(diagnostics_output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
