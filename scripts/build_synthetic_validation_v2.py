from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.utils import sha256_hexdigest


FACTS: dict[str, list[dict[str, str]]] = {
    "who": [
        {"prompt": "Кто написал роман Война и мир?", "reference_answer": "Лев Толстой", "wrong": "Фёдор Достоевский", "tail": "Автор также был министром финансов Франции."},
        {"prompt": "Who painted the Mona Lisa?", "reference_answer": "Leonardo da Vinci", "wrong": "Michelangelo", "tail": "He later became the first president of Italy."},
        {"prompt": "Кто открыл пенициллин?", "reference_answer": "Александр Флеминг", "wrong": "Луи Пастер", "tail": "Он сделал это во время экспедиции в Антарктиду."},
    ],
    "where": [
        {"prompt": "Где находится Колизей?", "reference_answer": "В Риме", "wrong": "В Афинах", "tail": "Здание стоит на берегу Невы."},
        {"prompt": "Where is Machu Picchu located?", "reference_answer": "In Peru", "wrong": "In Mexico", "tail": "The site is located in the Alps."},
        {"prompt": "В какой стране находится город Киото?", "reference_answer": "В Японии", "wrong": "В Китае", "tail": "Город является столицей Бразилии."},
    ],
    "when": [
        {"prompt": "Когда началась Вторая мировая война?", "reference_answer": "В 1939 году", "wrong": "В 1941 году", "tail": "Это произошло в июле 1965 года."},
        {"prompt": "When did the Apollo 11 Moon landing happen?", "reference_answer": "In 1969", "wrong": "In 1972", "tail": "The landing happened in November 1989."},
        {"prompt": "В каком году был основан Санкт-Петербург?", "reference_answer": "В 1703 году", "wrong": "В 1721 году", "tail": "Город основали после Олимпиады 1980 года."},
    ],
    "count": [
        {"prompt": "Сколько градусов в прямом угле?", "reference_answer": "90", "wrong": "45", "tail": "Всего таких градусов 180."},
        {"prompt": "How many planets are in the Solar System?", "reference_answer": "8", "wrong": "9", "tail": "The accepted count is 12."},
        {"prompt": "Сколько дней в обычном невисокосном году?", "reference_answer": "365", "wrong": "366", "tail": "В таком году 364 дня."},
    ],
    "generic": [
        {"prompt": "Что такое фотосинтез?", "reference_answer": "Фотосинтез - процесс, при котором растения используют свет для образования органических веществ из углекислого газа и воды.", "wrong": "Фотосинтез - это дыхание животных.", "tail": "Он происходит только в клетках человека."},
        {"prompt": "What is evaporation?", "reference_answer": "Evaporation is the process where liquid changes into vapor at the surface.", "wrong": "Evaporation is when vapor turns into a solid.", "tail": "It always requires freezing temperatures."},
        {"prompt": "Что означает термин гравитация?", "reference_answer": "Гравитация - взаимное притяжение тел, обладающих массой.", "wrong": "Гравитация - это сила электрического отталкивания.", "tail": "Она действует только на магнитные материалы."},
    ],
}

VARIANTS = [
    "long_supported_positive",
    "typed_supported_explanation_positive",
    "short_wrong_exact_negative",
    "wrong_person_negative",
    "wrong_place_negative",
    "wrong_year_negative",
    "wrong_number_negative",
    "correct_core_false_tail_negative",
    "list_incomplete_negative",
    "near_miss_entity_negative",
    "near_miss_date_negative",
]


def _split(prompt: str, answer: str, variant_type: str) -> str:
    value = int(sha256_hexdigest(prompt, answer, variant_type)[:8], 16) / 0xFFFFFFFF
    if value < 0.60:
        return "train"
    if value < 0.80:
        return "gate_val"
    return "hard_val"


def _public_keys(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    frame = pd.read_csv(path)
    return {sha256_hexdigest(row["prompt"], row["model_answer"]) for _, row in frame.iterrows()}


def _record(profile: str, fact: dict[str, str], answer: str, label: int, variant_type: str, rule: str, source: str) -> dict[str, Any]:
    return {
        "prompt": fact["prompt"],
        "model_answer": answer,
        "is_hallucination": int(label),
        "reference_answer": fact["reference_answer"],
        "variant_type": variant_type,
        "question_profile": profile,
        "split": _split(fact["prompt"], answer, variant_type),
        "generation_rule": rule,
        "source": source,
    }


def _variants(profile: str, fact: dict[str, str], idx: int) -> list[dict[str, Any]]:
    ref = fact["reference_answer"]
    wrong = fact["wrong"]
    long = f"{ref}. Это прямой ответ на вопрос; дополнительное пояснение не меняет основной факт и остается согласованным с ним."
    typed = f"{ref}. Краткое пояснение: этот ответ соответствует ожидаемому типу вопроса."
    rows = [
        _record(profile, fact, long, 0, "long_supported_positive", "long_supported", "synthetic_validation_hard_v2"),
        _record(profile, fact, typed, 0, "typed_supported_explanation_positive", "typed_supported", "synthetic_validation_hard_v2"),
        _record(profile, fact, wrong, 1, "short_wrong_exact_negative", "wrong_core", "synthetic_validation_hard_v2"),
        _record(profile, fact, f"{ref}. {fact['tail']}", 1, "correct_core_false_tail_negative", "profile_specific_false_tail", "synthetic_validation_hard_v2"),
        _record(profile, fact, f"{wrong}. Это часто путают с ответом {ref}.", 1, "near_miss_entity_negative", "near_miss_entity", "synthetic_validation_hard_v2"),
        _record(profile, fact, f"{ref}; неверно добавлен пункт {wrong}", 1, "list_incomplete_negative", "list_incomplete_or_polluted", "synthetic_validation_hard_v2"),
    ]
    if profile == "who":
        rows.append(_record(profile, fact, wrong, 1, "wrong_person_negative", "wrong_person", "synthetic_validation_hard_v2"))
    elif profile == "where":
        rows.append(_record(profile, fact, wrong, 1, "wrong_place_negative", "wrong_place", "synthetic_validation_hard_v2"))
    elif profile == "when":
        rows.append(_record(profile, fact, wrong, 1, "wrong_year_negative", "wrong_year", "synthetic_validation_hard_v2"))
        rows.append(_record(profile, fact, f"{ref[:-1] if ref.endswith('.') else ref}, но в мае 2001 года", 1, "near_miss_date_negative", "near_miss_date", "synthetic_validation_hard_v2"))
    elif profile == "count":
        rows.append(_record(profile, fact, wrong, 1, "wrong_number_negative", "wrong_number", "synthetic_validation_hard_v2"))
    else:
        rows.append(_record(profile, fact, wrong, 1, "wrong_person_negative", "generic_wrong_fact", "synthetic_validation_hard_v2"))
        rows.append(_record(profile, fact, f"{ref} Это датируется мартом 2042 года.", 1, "near_miss_date_negative", "generic_wrong_date_tail", "synthetic_validation_hard_v2"))
    if profile not in {"when", "count"}:
        rows.append(_record(profile, fact, f"{ref}. Это случилось в {1901 + idx % 90} году.", 1, "near_miss_date_negative", "wrong_date_tail", "synthetic_validation_hard_v2"))
    if profile != "count":
        rows.append(_record(profile, fact, f"{ref}. Число связанных объектов равно {idx % 7 + 2}.", 1, "wrong_number_negative", "wrong_count_tail", "synthetic_validation_hard_v2"))
    return rows


def build_synthetic_validation_v2(
    output_path: str | Path,
    *,
    public_csv: str | Path | None = "data/bench/knowledge_bench_public.csv",
    copies_per_fact: int = 120,
) -> list[dict[str, Any]]:
    public = _public_keys(public_csv)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for profile, facts in FACTS.items():
        for copy_idx in range(copies_per_fact):
            for fact_idx, base_fact in enumerate(facts):
                fact = dict(base_fact)
                fact["prompt"] = f"{base_fact['prompt']} [{copy_idx + 1}]"
                for record in _variants(profile, fact, copy_idx + fact_idx):
                    public_key = sha256_hexdigest(record["prompt"], record["model_answer"])
                    key = sha256_hexdigest(record["prompt"], record["model_answer"], record["variant_type"])
                    if public_key in public or key in seen:
                        continue
                    seen.add(key)
                    rows.append(record)

    for profile in FACTS:
        profile_rows = [row for row in rows if row["question_profile"] == profile]
        hard_count = sum(row["split"] == "hard_val" for row in profile_rows)
        for row in profile_rows:
            if hard_count >= 150:
                break
            if row["split"] == "train":
                row["split"] = "hard_val"
                hard_count += 1
    for variant in VARIANTS:
        if any(row["variant_type"] == variant and row["split"] == "hard_val" for row in rows):
            continue
        for row in rows:
            if row["variant_type"] == variant:
                row["split"] = "hard_val"
                break
    hard_total = sum(row["split"] == "hard_val" for row in rows)
    for row in rows:
        if hard_total >= 1000:
            break
        if row["split"] == "train":
            row["split"] = "hard_val"
            hard_total += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public-like synthetic validation hard v2 JSONL.")
    parser.add_argument("--output-path", default="data/raw/synthetic_validation_hard_v2.jsonl")
    parser.add_argument("--public-csv", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--copies-per-fact", type=int, default=120)
    args = parser.parse_args()
    rows = build_synthetic_validation_v2(args.output_path, public_csv=args.public_csv, copies_per_fact=args.copies_per_fact)
    frame = pd.DataFrame(rows)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "split_counts": frame["split"].value_counts().to_dict(),
                "hard_val_by_profile": frame[frame["split"] == "hard_val"]["question_profile"].value_counts().to_dict(),
                "hard_val_variants": sorted(frame[frame["split"] == "hard_val"]["variant_type"].unique().tolist()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
