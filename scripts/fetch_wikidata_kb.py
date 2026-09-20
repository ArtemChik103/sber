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

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardian_of_truth.evidence import FORBIDDEN_SOURCE_COLUMNS
from guardian_of_truth.utils import iter_jsonl, sha256_hexdigest
from scripts.fetch_wikipedia_kb import build_queries


SEARCH_URL = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language={lang}&format=json&limit={limit}"
ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"


def _url_json(url: str, *, timeout: float = 8.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "guardian-of-truth-offline-kb/0.3"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        forbidden = FORBIDDEN_SOURCE_COLUMNS & set(reader.fieldnames or [])
        if forbidden:
            raise ValueError(f"Wikidata fetch source {path} contains forbidden columns: {sorted(forbidden)}")
        for row in reader:
            yield dict(row)


def _entity_text(entity: dict[str, Any], *, langs: list[str]) -> str:
    parts: list[str] = []
    for lang in langs:
        label = entity.get("labels", {}).get(lang, {}).get("value")
        description = entity.get("descriptions", {}).get(lang, {}).get("value")
        aliases = [item.get("value", "") for item in entity.get("aliases", {}).get(lang, [])[:12]]
        if label:
            parts.append(str(label))
        if description:
            parts.append(str(description))
        if aliases:
            parts.append("Aliases: " + ", ".join(alias for alias in aliases if alias))
    text = ". ".join(part for part in parts if part)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_wikidata_rows(
    source_paths: list[Path],
    output: Path,
    diagnostics_output: Path,
    *,
    limit: int = 5000,
    langs: list[str] | None = None,
    search_results: int = 3,
    entities_per_row: int = 2,
    sleep_sec: float = 0.02,
    timeout_sec: float = 8.0,
) -> dict[str, int]:
    langs = langs or ["ru", "en"]
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    seen_qids: set[str] = set()
    if output.exists():
        for record in iter_jsonl(output):
            qid = str(record.get("wikidata_id", ""))
            if qid:
                seen_qids.add(qid)
    rows = len(seen_qids)
    diagnostics = 0
    scanned = 0
    with output.open("a" if output.exists() else "w", encoding="utf-8") as out, diagnostics_output.open(
        "a" if diagnostics_output.exists() else "w", encoding="utf-8"
    ) as diag_out:
        for path in source_paths:
            for record in _iter_records(path):
                scanned += 1
                if rows >= limit:
                    return {"rows": rows, "diagnostics": diagnostics, "scanned": scanned}
                row_key = str(record.get("row_key") or sha256_hexdigest(record.get("prompt", ""), record.get("model_answer", "")))
                added_for_row = 0
                for query_item in build_queries(record):
                    if added_for_row >= entities_per_row or rows >= limit:
                        break
                    for lang in langs:
                        if added_for_row >= entities_per_row or rows >= limit:
                            break
                        query = query_item["query"]
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
                            payload = _url_json(
                                SEARCH_URL.format(query=urllib.parse.quote(query), lang=lang, limit=search_results),
                                timeout=timeout_sec,
                            )
                        except Exception as exc:
                            diag["skip_reason"] = f"search_error:{exc.__class__.__name__}"
                            diag_out.write(json.dumps(diag, ensure_ascii=False) + "\n")
                            diagnostics += 1
                            continue
                        hits = payload.get("search", [])
                        diag["search_hits"] = len(hits)
                        for hit in hits:
                            if added_for_row >= entities_per_row or rows >= limit:
                                break
                            qid = str(hit.get("id", ""))
                            if not qid or qid in seen_qids:
                                continue
                            try:
                                entity_payload = _url_json(ENTITY_URL.format(qid=qid), timeout=timeout_sec)
                            except Exception:
                                continue
                            entity = entity_payload.get("entities", {}).get(qid, {})
                            text = _entity_text(entity, langs=langs)
                            if len(text) < 120:
                                label = str(hit.get("label", ""))
                                description = str(hit.get("description", ""))
                                text = re.sub(r"\s+", " ", f"{label}. {description}. {text}").strip()
                            if len(text) < 120:
                                continue
                            title = entity.get("labels", {}).get(lang, {}).get("value") or hit.get("label") or qid
                            row = {
                                "row_key": row_key,
                                "prompt": str(record.get("prompt", "") or record.get("question", "")),
                                "answer": "",
                                "title": str(title),
                                "text": text,
                                "source": "wikidata",
                                "url": f"https://www.wikidata.org/wiki/{qid}",
                                "lang": lang,
                                "wikidata_id": qid,
                                "fetch_key": sha256_hexdigest(path, row_key, query, lang, qid),
                            }
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out.flush()
                            seen_qids.add(qid)
                            added_for_row += 1
                            rows += 1
                            diag["pages_added"] += 1
                            time.sleep(sleep_sec)
                        if diag["pages_added"] == 0 and not diag["skip_reason"]:
                            diag["skip_reason"] = "no_new_entities"
                        diag_out.write(json.dumps(diag, ensure_ascii=False) + "\n")
                        diag_out.flush()
                        diagnostics += 1
    return {"rows": rows, "diagnostics": diagnostics, "scanned": scanned}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wikidata entity descriptions for retrieval-only benchmark queries.")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--query-jsonl", action="append", default=[], help="JSONL generated by export_targeted_kb_queries_v8.py.")
    parser.add_argument("--output", default="data/external/wikidata_targeted_v3.jsonl")
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--langs", default="ru,en")
    parser.add_argument("--search-results", type=int, default=3)
    parser.add_argument("--entities-per-row", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    args = parser.parse_args()

    output = Path(args.output)
    diagnostics_output = Path(args.diagnostics_output or str(output.with_suffix(".diagnostics.jsonl")))
    summary = fetch_wikidata_rows(
        [Path(path) for path in [*args.source, *args.query_jsonl]],
        output,
        diagnostics_output,
        limit=args.limit,
        langs=[lang.strip() for lang in args.langs.split(",") if lang.strip()],
        search_results=args.search_results,
        entities_per_row=args.entities_per_row,
        sleep_sec=args.sleep_sec,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({"output": str(output), "diagnostics_output": str(diagnostics_output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
