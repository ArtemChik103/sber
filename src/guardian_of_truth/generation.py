from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from guardian_of_truth.utils import DATA_DIR, iter_jsonl, read_jsonl, sha256_hexdigest, write_jsonl


DEFAULT_SEED_PATH = DATA_DIR / "raw" / "seed_qa.jsonl"
DEFAULT_SYNTHETIC_PATH = DATA_DIR / "raw" / "synthetic_factual_data.jsonl"

HEDGED_FILLERS = [
    "примерно",
    "около",
    "ориентировочно",
]

ENTITY_SWAPS = {
    "Москва": "Санкт-Петербург",
    "Россия": "Казахстан",
    "Франция": "Италия",
    "Лондон": "Манчестер",
    "Афины": "Спарта",
    "Ньютон": "Эйнштейн",
    "Пушкин": "Лермонтов",
    "Италия": "Испания",
    "Париж": "Лион",
    "Фёдор Достоевский": "Лев Толстой",
    "Джордж Вашингтон": "Томас Джефферсон",
    "Нил": "Амазонка",
    "Юпитер": "Сатурн",
}

MONTH_SWAPS = {
    "января": "марта",
    "февраля": "апреля",
    "марта": "июня",
    "апреля": "июля",
    "мая": "августа",
    "июня": "сентября",
    "июля": "октября",
    "августа": "ноября",
    "сентября": "декабря",
    "октября": "января",
    "ноября": "февраля",
    "декабря": "мая",
}

NUMBER_WORD_SWAPS = {
    "один": "два",
    "два": "три",
    "три": "четыре",
    "пять": "шесть",
    "семь": "восемь",
    "десять": "двенадцать",
    "девяносто": "сто",
}

SYMBOL_SWAPS = {
    "Au": "Ag",
    "H2O": "CO2",
    "DNA": "RNA",
}


def load_seed_questions(seed_path: str | Path = DEFAULT_SEED_PATH) -> list[dict[str, Any]]:
    return read_jsonl(seed_path)


def mutate_answer_rule_based(answer: str, answer_type: str | None = None) -> str:
    text = answer.strip()

    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b", text)
    if year_match:
        year = int(year_match.group(0))
        return text[: year_match.start()] + str(year + 3) + text[year_match.end() :]

    date_match = re.search(r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b", text, flags=re.IGNORECASE)
    if date_match:
        day = min(int(date_match.group(1)) + 5, 31)
        month = MONTH_SWAPS.get(date_match.group(2).lower(), date_match.group(2))
        replacement = f"{day} {month}"
        return text[: date_match.start()] + replacement + text[date_match.end() :]

    number_match = re.search(r"\b\d+(?:[.,]\d+)?\b", text)
    if number_match:
        raw = number_match.group(0).replace(",", ".")
        if "." in raw:
            value = float(raw)
            replacement = f"{round(value * 1.2, 2)}".rstrip("0").rstrip(".")
        else:
            value = int(raw)
            replacement = str(value + max(1, value // 10 or 1))
        return text[: number_match.start()] + replacement + text[number_match.end() :]

    for src, dst in NUMBER_WORD_SWAPS.items():
        pattern = re.compile(rf"\b{src}\b", flags=re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(dst, text, count=1)

    for src, dst in ENTITY_SWAPS.items():
        if src in text:
            return text.replace(src, dst, 1)

    for src, dst in SYMBOL_SWAPS.items():
        if src in text:
            return text.replace(src, dst, 1)

    capitalized = re.findall(r"\b[А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)?\b", text)
    if capitalized:
        entity = capitalized[-1]
        if entity in ENTITY_SWAPS:
            return text.replace(entity, ENTITY_SWAPS[entity], 1)

    words = text.split()
    if not words:
        return text
    insert_at = min(2, len(words))
    filler = random.Random(42).choice(HEDGED_FILLERS)
    return " ".join(words[:insert_at] + [filler] + words[insert_at:])


def mutate_answer_groq(prompt: str, answer: str) -> str:
    raise RuntimeError(
        "Groq-backed negative generation is implemented in phase 3 via the API client. "
        "Use --stage groq-negatives after the verifier client is available."
    )


def _to_positive_record(seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": seed["prompt"],
        "answer": seed["answer"],
        "label": 0,
        "source": seed.get("source", "seed"),
        "variant_type": "positive",
    }


def _to_rule_negative_record(seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": seed["prompt"],
        "answer": mutate_answer_rule_based(seed["answer"], seed.get("answer_type")),
        "label": 1,
        "source": seed.get("source", "seed"),
        "variant_type": "rule_negative",
    }


def build_dataset(
    seed_path: str | Path = DEFAULT_SEED_PATH,
    output_path: str | Path = DEFAULT_SYNTHETIC_PATH,
    include_rule_negatives: bool = True,
    include_groq_negatives: bool = False,
    resume: bool = False,
) -> list[dict[str, Any]]:
    seeds = load_seed_questions(seed_path)
    existing_keys: set[str] = set()
    records: list[dict[str, Any]] = []

    if resume and Path(output_path).exists():
        records = read_jsonl(output_path)
        existing_keys = {
            sha256_hexdigest(row["prompt"], row["answer"], row["variant_type"])
            for row in records
        }

    for seed in seeds:
        positive = _to_positive_record(seed)
        pos_key = sha256_hexdigest(positive["prompt"], positive["answer"], positive["variant_type"])
        if pos_key not in existing_keys:
            records.append(positive)
            existing_keys.add(pos_key)

        if include_rule_negatives:
            negative = _to_rule_negative_record(seed)
            neg_key = sha256_hexdigest(negative["prompt"], negative["answer"], negative["variant_type"])
            if neg_key not in existing_keys:
                records.append(negative)
                existing_keys.add(neg_key)

    if include_groq_negatives:
        raise RuntimeError("Groq negatives require the Groq mutation client from phase 3.")

    write_jsonl(output_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic factual dataset.")
    parser.add_argument("--stage", choices=["seed", "rule-negatives", "groq-negatives"], required=True)
    parser.add_argument("--seed-path", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_SYNTHETIC_PATH))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seed_path = Path(args.seed_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "seed":
        seeds = load_seed_questions(seed_path)
        positives = [_to_positive_record(seed) for seed in seeds]
        write_jsonl(output_path, positives)
        print(json.dumps({"stage": args.stage, "rows": len(positives)}, ensure_ascii=False))
        return

    if args.stage == "rule-negatives":
        records = build_dataset(
            seed_path=seed_path,
            output_path=output_path,
            include_rule_negatives=True,
            include_groq_negatives=False,
            resume=args.resume,
        )
        print(json.dumps({"stage": args.stage, "rows": len(records)}, ensure_ascii=False))
        return

    mutate_answer_groq("", "")


if __name__ == "__main__":
    main()
