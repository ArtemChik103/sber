from __future__ import annotations

import argparse
import math
import json
import os
import random
import re
import time
import inspect
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import httpx
import groq
from groq import AsyncGroq

from guardian_of_truth.api_client import ApiSettings, SlidingWindowRateLimiter
from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import DATA_DIR, read_jsonl, run_coro_sync, sha256_hexdigest, write_jsonl


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

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "guardian-of-truth/0.1 (seed-harvest; contact: github.com/ArtemChik103/sber)",
}

SEED_SPECS = [
    {
        "name": "country_capital",
        "answer_type": "place",
        "query": """
            SELECT ?countryLabel ?capitalLabel WHERE {
              ?country wdt:P31 wd:Q6256; wdt:P36 ?capital.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?countryLabel
            LIMIT {limit}
        """,
        "prompt": "Как называется столица {countryLabel}?",
        "answer": "Столица {countryLabel} — {capitalLabel}.",
    },
    {
        "name": "country_currency",
        "answer_type": "entity",
        "query": """
            SELECT ?countryLabel ?currencyLabel WHERE {
              ?country wdt:P31 wd:Q6256; wdt:P38 ?currency.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?countryLabel
            LIMIT {limit}
        """,
        "prompt": "Какая валюта используется в {countryLabel}?",
        "answer": "В {countryLabel} используется валюта {currencyLabel}.",
    },
    {
        "name": "country_language",
        "answer_type": "entity",
        "query": """
            SELECT ?countryLabel ?languageLabel WHERE {
              ?country wdt:P31 wd:Q6256; wdt:P37 ?language.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?countryLabel
            LIMIT {limit}
        """,
        "prompt": "Какой официальный язык у {countryLabel}?",
        "answer": "Официальный язык {countryLabel} — {languageLabel}.",
    },
    {
        "name": "person_birth_year",
        "answer_type": "year",
        "query": """
            SELECT ?personLabel ?dob WHERE {
              ?person wdt:P31 wd:Q5; wdt:P569 ?dob.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?personLabel
            LIMIT {limit}
        """,
        "prompt": "В каком году родился {personLabel}?",
        "answer": "{personLabel} родился в {dob_year} году.",
    },
    {
        "name": "person_death_year",
        "answer_type": "year",
        "query": """
            SELECT ?personLabel ?dod WHERE {
              ?person wdt:P31 wd:Q5; wdt:P570 ?dod.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?personLabel
            LIMIT {limit}
        """,
        "prompt": "В каком году умер {personLabel}?",
        "answer": "{personLabel} умер в {dod_year} году.",
    },
    {
        "name": "person_citizenship",
        "answer_type": "place",
        "query": """
            SELECT ?personLabel ?countryLabel WHERE {
              ?person wdt:P31 wd:Q5; wdt:P27 ?country.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?personLabel
            LIMIT {limit}
        """,
        "prompt": "Гражданином какой страны был {personLabel}?",
        "answer": "{personLabel} был гражданином {countryLabel}.",
    },
    {
        "name": "book_author",
        "answer_type": "person",
        "query": """
            SELECT ?workLabel ?authorLabel WHERE {
              ?work wdt:P31 wd:Q571; wdt:P50 ?author.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?workLabel
            LIMIT {limit}
        """,
        "prompt": "Кто написал книгу «{workLabel}»?",
        "answer": "Книгу «{workLabel}» написал {authorLabel}.",
    },
    {
        "name": "element_symbol",
        "answer_type": "entity",
        "query": """
            SELECT ?elementLabel ?symbol WHERE {
              ?element wdt:P31 wd:Q11344; wdt:P246 ?symbol.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
            }
            ORDER BY ?elementLabel
            LIMIT {limit}
        """,
        "prompt": "Какой химический символ у элемента {elementLabel}?",
        "answer": "Химический символ элемента {elementLabel} — {symbol}.",
    },
]


def load_seed_questions(seed_path: str | Path = DEFAULT_SEED_PATH) -> list[dict[str, Any]]:
    return read_jsonl(seed_path)


def harvest_seed_questions(
    output_path: str | Path = DEFAULT_SEED_PATH,
    *,
    target_size: int = 1800,
    resume: bool = True,
) -> list[dict[str, Any]]:
    per_spec = max(25, math.ceil(target_size / len(SEED_SPECS)))
    query_limit = min(100, max(per_spec, per_spec + 20))
    output = Path(output_path)
    existing: list[dict[str, Any]] = []
    if resume and output.exists():
        existing = read_jsonl(output)

    seen_prompts = {row["prompt"] for row in existing}
    harvested: list[dict[str, Any]] = list(existing)

    for spec in SEED_SPECS:
        query = spec["query"].replace("{limit}", str(query_limit))
        try:
            bindings = _run_wikidata_query(query)
        except Exception:
            continue

        added = 0
        for binding in bindings:
            normalized = _normalize_binding(binding)
            if not normalized:
                continue
            prompt = spec["prompt"].format(**normalized).strip()
            answer = spec["answer"].format(**normalized).strip()
            if prompt in seen_prompts or not _is_seed_record_usable(prompt, answer):
                continue
            harvested.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "answer_type": spec["answer_type"],
                    "source": f"wikidata:{spec['name']}",
                }
            )
            seen_prompts.add(prompt)
            added += 1
            if added >= per_spec:
                break

    write_jsonl(output, harvested)
    return harvested


def _run_wikidata_query(query: str) -> list[dict[str, dict[str, str]]]:
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(f"{WIKIDATA_SPARQL_URL}?{params}", headers=WIKIDATA_HEADERS)
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["results"]["bindings"]


def _normalize_binding(binding: dict[str, dict[str, str]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in binding.items():
        raw = value.get("value", "").strip()
        if not raw:
            return {}
        normalized[key] = raw
        if key in {"dob", "dod"}:
            year = re.search(r"\b(\d{4})\b", raw)
            if not year:
                return {}
            normalized[f"{key}_year"] = year.group(1)
        if key == "height":
            try:
                normalized["height_int"] = str(int(round(float(raw))))
            except ValueError:
                return {}
    return normalized


def _is_seed_record_usable(prompt: str, answer: str) -> bool:
    if len(prompt) < 12 or len(answer) < 8:
        return False
    if len(prompt) > 220 or len(answer) > 260:
        return False
    if "http://" in prompt or "http://" in answer or "https://" in prompt or "https://" in answer:
        return False
    if any(token in prompt for token in ("Q", "P")) and re.search(r"\bQ\d+\b|\bP\d+\b", prompt):
        return False
    return True


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
    return GroqNegativeGenerator().mutate(prompt, answer)


class GroqNegativeGenerator:
    def __init__(self, api_key: str | None = None, settings: ApiSettings | None = None, cache: SQLiteCache | None = None) -> None:
        self.settings = settings or ApiSettings.from_yaml()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.cache = cache or SQLiteCache(DATA_DIR / "cache" / "groq_cache.sqlite")
        self.rate_limiter = SlidingWindowRateLimiter(
            rpm=self.settings.target_rpm,
            tpm=self.settings.target_tpm,
        )

    def mutate(self, prompt: str, answer: str) -> str | None:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required for groq-negative generation.")
        cache_key = sha256_hexdigest(
            self.settings.runtime_model,
            self.settings.dataset_prompt_version,
            prompt,
            answer,
            "groq_negative",
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return str(cached["answer"])

        estimated_tokens = max(48, (len(prompt) + len(answer)) // 4 + 48)
        delay = self.rate_limiter.reserve_delay(estimated_tokens)
        if delay > 0:
            time.sleep(delay)
            second_delay = self.rate_limiter.reserve_delay(estimated_tokens)
            if second_delay > 0:
                time.sleep(second_delay)

        try:
            mutated = run_coro_sync(self._mutate_async(prompt, answer))
        except (groq.APIError, RuntimeError):
            return None
        self.cache.set(cache_key, {"answer": mutated})
        return mutated

    async def _mutate_async(self, prompt: str, answer: str) -> str:
        timeout = httpx.Timeout(
            timeout=max(2.5, self.settings.total_timeout_sec),
            connect=max(0.5, self.settings.connect_timeout_sec),
            read=max(2.0, self.settings.read_timeout_sec),
            write=max(2.0, self.settings.read_timeout_sec),
        )
        answer_word_count = max(1, len(answer.split()))
        max_tokens = min(self.settings.max_tokens_dataset, max(20, answer_word_count * 3 + 4))
        client = AsyncGroq(
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,
        )
        try:
            response = await client.chat.completions.create(
                model=self.settings.runtime_model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the answer so that it contains exactly one plausible factual mistake. "
                            "Keep the same language, sentence count, and answer template. "
                            "Change only one key fact such as a person, place, year, number, title, or symbol. "
                            "Do not add any explanation, second clause, date, biography, or extra fact. "
                            "Output only the rewritten answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {prompt}\n"
                            f"Correct answer: {answer}\n"
                            "Return one short rewritten answer with exactly one wrong fact and no extra text."
                        ),
                    },
                ],
                timeout=timeout,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content or content == answer:
                raise RuntimeError("Groq negative generation returned an empty or unchanged answer.")
            if len(content.split()) > max(4, int(len(answer.split()) * 1.35) + 2):
                raise RuntimeError("Groq negative generation expanded the answer too much.")
            return content
        finally:
            close_result = client.close()
            if inspect.isawaitable(close_result):
                await close_result


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
    groq_limit: int | None = None,
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
        generator = GroqNegativeGenerator()
        produced = 0
        attempts = 0
        max_attempts = max(25, (groq_limit or 0) * 4) if groq_limit is not None else len(seeds)
        for seed in seeds:
            if groq_limit is not None and produced >= groq_limit:
                break
            if attempts >= max_attempts:
                break
            attempts += 1
            mutated = generator.mutate(seed["prompt"], seed["answer"])
            if not mutated or mutated == seed["answer"]:
                continue
            negative = {
                "prompt": seed["prompt"],
                "answer": mutated,
                "label": 1,
                "source": seed.get("source", "seed"),
                "variant_type": "groq_negative",
            }
            neg_key = sha256_hexdigest(negative["prompt"], negative["answer"], negative["variant_type"])
            if neg_key not in existing_keys:
                records.append(negative)
                existing_keys.add(neg_key)
                produced += 1

    write_jsonl(output_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic factual dataset.")
    parser.add_argument("--stage", choices=["seed", "seed-harvest", "rule-negatives", "groq-negatives"], required=True)
    parser.add_argument("--seed-path", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_SYNTHETIC_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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

    if args.stage == "seed-harvest":
        rows = harvest_seed_questions(output_path=seed_path, target_size=args.limit or 1800, resume=args.resume)
        print(json.dumps({"stage": args.stage, "rows": len(rows)}, ensure_ascii=False))
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

    records = build_dataset(
        seed_path=seed_path,
        output_path=output_path,
        include_rule_negatives=True,
        include_groq_negatives=True,
        resume=args.resume,
        groq_limit=args.limit,
    )
    print(json.dumps({"stage": args.stage, "rows": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
