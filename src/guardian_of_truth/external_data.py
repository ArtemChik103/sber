from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import httpx

from guardian_of_truth.utils import DATA_DIR, read_jsonl, sha256_hexdigest, write_jsonl


POPQA_URL = "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv?download=true"
FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"

DEFAULT_EXTERNAL_SEED_PATH = DATA_DIR / "raw" / "external_seed_qa.jsonl"
DEFAULT_EXTERNAL_SYNTHETIC_PATH = DATA_DIR / "raw" / "external_synthetic_factual_data.jsonl"
DEFAULT_MAIN_SEED_PATH = DATA_DIR / "raw" / "seed_qa.jsonl"
DEFAULT_MAIN_SYNTHETIC_PATH = DATA_DIR / "raw" / "synthetic_factual_data.jsonl"


def ingest_popqa(
    *,
    limit: int = 1000,
    seed_output_path: str | Path = DEFAULT_EXTERNAL_SEED_PATH,
    synthetic_output_path: str | Path = DEFAULT_EXTERNAL_SYNTHETIC_PATH,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_popqa_rows(limit=limit)
    property_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        property_buckets[row["prop"]].append(row)

    existing_seed = read_jsonl(seed_output_path) if resume and Path(seed_output_path).exists() else []
    existing_synth = read_jsonl(synthetic_output_path) if resume and Path(synthetic_output_path).exists() else []
    seed_seen = {sha256_hexdigest(item["prompt"], item["answer"], item.get("source")) for item in existing_seed}
    synth_seen = {sha256_hexdigest(item["prompt"], item["answer"], item["label"], item.get("source"), item.get("variant_type")) for item in existing_synth}

    seed_records = list(existing_seed)
    synth_records = list(existing_synth)

    for prop, bucket in property_buckets.items():
        if len(bucket) < 2:
            continue
        ordered = sorted(bucket, key=lambda item: item["question"])
        for idx, row in enumerate(ordered):
            positive_answer = row["answer"]
            negative_answer = ordered[(idx + 1) % len(ordered)]["answer"]
            if positive_answer == negative_answer:
                continue

            seed_record = {
                "prompt": row["question"],
                "answer": positive_answer,
                "answer_type": _infer_answer_type(positive_answer, row["prop"]),
                "source": f"popqa:{row['prop']}",
            }
            seed_key = sha256_hexdigest(seed_record["prompt"], seed_record["answer"], seed_record["source"])
            if seed_key not in seed_seen:
                seed_records.append(seed_record)
                seed_seen.add(seed_key)

            positive = {
                "prompt": row["question"],
                "answer": positive_answer,
                "label": 0,
                "source": f"popqa:{row['prop']}",
                "variant_type": "popqa_positive",
            }
            negative = {
                "prompt": row["question"],
                "answer": negative_answer,
                "label": 1,
                "source": f"popqa:{row['prop']}",
                "variant_type": "popqa_negative",
            }
            for record in (positive, negative):
                key = sha256_hexdigest(record["prompt"], record["answer"], record["label"], record["source"], record["variant_type"])
                if key not in synth_seen:
                    synth_records.append(record)
                    synth_seen.add(key)

    write_jsonl(seed_output_path, seed_records)
    write_jsonl(synthetic_output_path, synth_records)
    return seed_records, synth_records


def ingest_fever(
    *,
    limit: int = 1000,
    synthetic_output_path: str | Path = DEFAULT_EXTERNAL_SYNTHETIC_PATH,
    resume: bool = True,
    include_nei: bool = False,
) -> list[dict[str, Any]]:
    existing = read_jsonl(synthetic_output_path) if resume and Path(synthetic_output_path).exists() else []
    seen = {sha256_hexdigest(item["prompt"], item["answer"], item["label"], item.get("source"), item.get("variant_type")) for item in existing}
    records = list(existing)

    with httpx.stream("GET", FEVER_TRAIN_URL, timeout=60.0) as response:
        response.raise_for_status()
        added = 0
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            item = json.loads(raw_line)
            label = item.get("label")
            if label not in {"SUPPORTS", "REFUTES"} and not (include_nei and label == "NOT ENOUGH INFO"):
                continue

            record = {
                "prompt": "Assess whether the following claim is factually correct.",
                "answer": str(item["claim"]).strip(),
                "label": 0 if label == "SUPPORTS" else 1,
                "source": "fever",
                "variant_type": f"fever_{label.lower().replace(' ', '_')}",
            }
            key = sha256_hexdigest(record["prompt"], record["answer"], record["label"], record["source"], record["variant_type"])
            if key in seen:
                continue

            records.append(record)
            seen.add(key)
            added += 1
            if added >= limit:
                break

    write_jsonl(synthetic_output_path, records)
    return records


def merge_external_into_main(
    *,
    external_seed_path: str | Path = DEFAULT_EXTERNAL_SEED_PATH,
    external_synthetic_path: str | Path = DEFAULT_EXTERNAL_SYNTHETIC_PATH,
    main_seed_path: str | Path = DEFAULT_MAIN_SEED_PATH,
    main_synthetic_path: str | Path = DEFAULT_MAIN_SYNTHETIC_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_seed = read_jsonl(main_seed_path) if Path(main_seed_path).exists() else []
    main_synth = read_jsonl(main_synthetic_path) if Path(main_synthetic_path).exists() else []
    external_seed = read_jsonl(external_seed_path) if Path(external_seed_path).exists() else []
    external_synth = read_jsonl(external_synthetic_path) if Path(external_synthetic_path).exists() else []

    merged_seed = _merge_unique(main_seed, external_seed, lambda r: sha256_hexdigest(r["prompt"], r["answer"], r.get("source")))
    merged_synth = _merge_unique(main_synth, external_synth, lambda r: sha256_hexdigest(r["prompt"], r["answer"], r["label"], r.get("source"), r.get("variant_type")))

    write_jsonl(main_seed_path, merged_seed)
    write_jsonl(main_synthetic_path, merged_synth)
    return merged_seed, merged_synth


def _load_popqa_rows(*, limit: int) -> list[dict[str, str]]:
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(POPQA_URL)
        response.raise_for_status()
    stream = io.StringIO(response.text)
    reader = csv.DictReader(stream, delimiter="\t")
    rows: list[dict[str, str]] = []
    for row in reader:
        answers = _parse_possible_answers(row.get("possible_answers", "[]"))
        answer = _choose_primary_answer(answers)
        question = str(row.get("question", "")).strip()
        prop = str(row.get("prop", "")).strip() or "unknown"
        if not question or not answer:
            continue
        if len(question) > 220 or len(answer) > 120:
            continue
        rows.append({"question": question, "answer": answer, "prop": prop})
        if len(rows) >= limit:
            break
    return rows


def _parse_possible_answers(raw_value: str) -> list[str]:
    try:
        answers = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    return [str(item).strip() for item in answers if str(item).strip()]


def _choose_primary_answer(answers: list[str]) -> str:
    for answer in answers:
        if 1 <= len(answer.split()) <= 8 and len(answer) <= 80:
            return answer
    return answers[0] if answers else ""


def _infer_answer_type(answer: str, prop: str) -> str:
    if answer.isdigit() and len(answer) == 4:
        return "year"
    if any(ch.isdigit() for ch in answer):
        return "number"
    if prop in {"place of birth", "country", "capital"}:
        return "place"
    return "entity"


def _merge_unique(
    current: list[dict[str, Any]],
    extra: Iterable[dict[str, Any]],
    key_fn,
) -> list[dict[str, Any]]:
    result = list(current)
    seen = {key_fn(item) for item in current}
    for item in extra:
        key = key_fn(item)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest external datasets for Guardian of Truth.")
    parser.add_argument("--stage", choices=["popqa", "fever", "all", "merge"], required=True)
    parser.add_argument("--popqa-limit", type=int, default=1000)
    parser.add_argument("--fever-limit", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-main", action="store_true")
    args = parser.parse_args()

    summary: dict[str, Any] = {"stage": args.stage}

    if args.stage in {"popqa", "all"}:
        seeds, synth = ingest_popqa(limit=args.popqa_limit, resume=args.resume)
        summary["external_seed_rows"] = len(seeds)
        summary["external_synth_rows_after_popqa"] = len(synth)

    if args.stage in {"fever", "all"}:
        synth = ingest_fever(limit=args.fever_limit, resume=args.resume)
        summary["external_synth_rows_after_fever"] = len(synth)

    if args.stage == "merge" or args.merge_main:
        seed, synth = merge_external_into_main()
        summary["main_seed_rows"] = len(seed)
        summary["main_synth_rows"] = len(synth)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
