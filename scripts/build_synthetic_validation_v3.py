from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.utils import sha256_hexdigest
from scripts.analysis_taxonomy import TAXONOMY_LABELS


PROFILES = ("who", "where", "when", "count", "generic")
VARIANTS = [
    "gate_overraises_correct",
    "gate_underraises_wrong",
    "fallback_overraises_correct",
    "fallback_misses_wrong",
    "long_supported_correct_penalized",
    "typed_short_wrong_missed",
    "where_who_when_regression",
    "audit_strong_risk_but_low_score",
    "audit_low_risk_but_high_score",
]


BASE_FACTS: dict[str, list[dict[str, str]]] = {
    "who": [
        {"prompt": "Кто написал пьесу Ревизор?", "reference_answer": "Николай Гоголь", "wrong": "Александр Островский"},
        {"prompt": "Who discovered radium?", "reference_answer": "Marie Curie and Pierre Curie", "wrong": "Rosalind Franklin"},
        {"prompt": "Кто был первым космонавтом?", "reference_answer": "Юрий Гагарин", "wrong": "Алексей Леонов"},
        {"prompt": "Who composed The Four Seasons?", "reference_answer": "Antonio Vivaldi", "wrong": "Johann Sebastian Bach"},
    ],
    "where": [
        {"prompt": "Где находится музей Прадо?", "reference_answer": "В Мадриде", "wrong": "В Лиссабоне"},
        {"prompt": "Where is the Uffizi Gallery located?", "reference_answer": "Florence, Italy", "wrong": "Vienna, Austria"},
        {"prompt": "В какой стране находится озеро Балхаш?", "reference_answer": "В Казахстане", "wrong": "В Монголии"},
        {"prompt": "Where is the Atacama Desert?", "reference_answer": "In Chile", "wrong": "In Morocco"},
    ],
    "when": [
        {"prompt": "Когда распался Советский Союз?", "reference_answer": "В 1991 году", "wrong": "В 1989 году"},
        {"prompt": "When was the Treaty of Versailles signed?", "reference_answer": "In 1919", "wrong": "In 1923"},
        {"prompt": "В каком году был запущен первый спутник Земли?", "reference_answer": "В 1957 году", "wrong": "В 1961 году"},
        {"prompt": "When did the Berlin Wall fall?", "reference_answer": "In 1989", "wrong": "In 1991"},
    ],
    "count": [
        {"prompt": "Сколько костей обычно в теле взрослого человека?", "reference_answer": "206", "wrong": "216"},
        {"prompt": "How many strings does a standard violin have?", "reference_answer": "4", "wrong": "5"},
        {"prompt": "Сколько камер сердца у человека?", "reference_answer": "4", "wrong": "3"},
        {"prompt": "How many teeth does a typical adult human have?", "reference_answer": "32", "wrong": "28"},
    ],
    "generic": [
        {"prompt": "Что такое вулкан?", "reference_answer": "Вулкан - геологическое образование, через которое магма и газы выходят на поверхность.", "wrong": "Вулкан - это искусственная плотина для хранения воды."},
        {"prompt": "What is an ecosystem?", "reference_answer": "An ecosystem is a community of organisms interacting with each other and their physical environment.", "wrong": "An ecosystem is a single chemical element found in soil."},
        {"prompt": "Что такое инфляция?", "reference_answer": "Инфляция - устойчивый рост общего уровня цен и снижение покупательной способности денег.", "wrong": "Инфляция - это снижение всех цен до нуля."},
        {"prompt": "What is plate tectonics?", "reference_answer": "Plate tectonics describes the movement of Earth's lithospheric plates.", "wrong": "Plate tectonics is a method for classifying clouds."},
    ],
}


UNSUPPORTED_TAILS = [
    "В дополнительных пояснениях часто указывают, что этот факт был официально закреплен международным комитетом в 2007 году.",
    "Some summaries add that the event was later renamed by a UNESCO panel, but that detail is not part of the verified answer.",
    "Также иногда добавляют связь с Нобелевской премией, хотя для данного вопроса это неподтвержденная подробность.",
    "A longer version may mention a later expedition and a museum archive, which is unnecessary and not supported by the prompt.",
]


def _split(prompt: str, answer: str, variant: str) -> str:
    value = int(sha256_hexdigest("hard-v3", prompt, answer, variant)[:8], 16) / 0xFFFFFFFF
    if value < 0.55:
        return "train"
    if value < 0.75:
        return "gate_val"
    return "hard_val"


def _public_keys(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    frame = pd.read_csv(path)
    return {sha256_hexdigest(row["prompt"], row["model_answer"]) for _, row in frame.iterrows()}


def _record(profile: str, fact: dict[str, str], answer: str, label: int, variant: str, ordinal: int) -> dict[str, Any]:
    return {
        "prompt": fact["prompt"],
        "model_answer": answer,
        "is_hallucination": int(label),
        "reference_answer": fact["reference_answer"],
        "variant_type": variant,
        "taxonomy_family": variant,
        "question_profile": profile,
        "split": _split(fact["prompt"], answer, variant),
        "generation_rule": f"hard_v3_{variant}",
        "source": "synthetic_validation_hard_v3",
        "synthetic_id": sha256_hexdigest("hard-v3-id", profile, fact["prompt"], answer, ordinal)[:16],
    }


def _fact_copy(profile: str, fact: dict[str, str], ordinal: int) -> dict[str, str]:
    topic = ["основной", "краткий", "проверочный", "энциклопедический", "архивный", "учебный", "справочный", "экзаменационный", "редакторский", "контрольный", "исторический"][ordinal % 11]
    context = ["для школьной справки", "для фактчекинга", "для карточки знаний", "для устного ответа", "для энциклопедической заметки", "для теста", "для краткого конспекта", "для проверочного списка", "для учебного примера", "для справочного блока"][(ordinal // 11) % 10]
    prompt = fact["prompt"]
    if profile == "generic":
        prompt = f"{prompt} Дайте {topic} ответ {context}."
    elif ordinal % 2:
        prompt = f"{prompt} Ответьте {topic} формулировкой {context}."
    else:
        prompt = f"{prompt} Нужен {topic} ответ {context}."
    return {**fact, "prompt": prompt}


def _variants(profile: str, fact: dict[str, str], ordinal: int) -> list[dict[str, Any]]:
    ref = fact["reference_answer"]
    wrong = fact["wrong"]
    supported_long = (
        f"{ref}. Это прямой ответ на вопрос. Дополнительное пояснение остается в рамках проверяемого факта "
        "и не добавляет отдельного спорного утверждения."
    )
    clean_short = ref
    wrong_core = f"{wrong}. Ответ дан кратко, без дополнительных оговорок."
    plausible_tail = f"{ref}. {UNSUPPORTED_TAILS[ordinal % len(UNSUPPORTED_TAILS)]}"
    typed_wrong = wrong
    records = [
        _record(profile, fact, supported_long, 0, "long_supported_correct_penalized", ordinal),
        _record(profile, fact, clean_short, 0, "audit_low_risk_but_high_score", ordinal),
        _record(profile, fact, wrong_core, 1, "gate_underraises_wrong", ordinal),
        _record(profile, fact, plausible_tail, 1, "audit_strong_risk_but_low_score", ordinal),
        _record(profile, fact, f"{wrong}. {UNSUPPORTED_TAILS[(ordinal + 1) % len(UNSUPPORTED_TAILS)]}", 1, "fallback_misses_wrong", ordinal),
        _record(profile, fact, f"{ref}. Подробное объяснение: ответ относится именно к запрошенному факту и не меняет его.", 0, "gate_overraises_correct", ordinal),
        _record(profile, fact, f"{ref}. Формулировка короткая и точная.", 0, "fallback_overraises_correct", ordinal),
    ]
    if profile in {"who", "where", "when", "count"}:
        records.append(_record(profile, fact, typed_wrong, 1, "typed_short_wrong_missed", ordinal))
    if profile in {"who", "where", "when"}:
        records.append(_record(profile, fact, f"{wrong}. Это выглядит как ответ нужного типа, но основной факт неверен.", 1, "where_who_when_regression", ordinal))
    else:
        records.append(_record(profile, fact, f"{wrong}. Основное утверждение неверно, хотя форма ответа правдоподобна.", 1, "where_who_when_regression", ordinal))
    return records


def build_synthetic_validation_v3(
    output_path: str | Path = "data/raw/synthetic_validation_hard_v3.jsonl",
    *,
    public_csv: str | Path | None = "data/bench/knowledge_bench_public.csv",
    copies_per_fact: int = 110,
) -> list[dict[str, Any]]:
    public = _public_keys(public_csv)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for profile, facts in BASE_FACTS.items():
        for ordinal in range(copies_per_fact):
            for fact in facts:
                copied = _fact_copy(profile, fact, ordinal)
                for record in _variants(profile, copied, ordinal):
                    public_key = sha256_hexdigest(record["prompt"], record["model_answer"])
                    stable_key = sha256_hexdigest(record["prompt"], record["model_answer"], record["variant_type"])
                    if public_key in public or stable_key in seen:
                        continue
                    seen.add(stable_key)
                    rows.append(record)

    for profile in PROFILES:
        profile_rows = [row for row in rows if row["question_profile"] == profile]
        hard_count = sum(row["split"] == "hard_val" for row in profile_rows)
        for row in profile_rows:
            if hard_count >= 200:
                break
            if row["split"] != "hard_val":
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
        if hard_total >= 1500:
            break
        if row["split"] != "hard_val":
            row["split"] = "hard_val"
            hard_total += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public-shaped synthetic validation hard v3 JSONL.")
    parser.add_argument("--output-path", default="data/raw/synthetic_validation_hard_v3.jsonl")
    parser.add_argument("--public-csv", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--copies-per-fact", type=int, default=110)
    args = parser.parse_args()
    rows = build_synthetic_validation_v3(args.output_path, public_csv=args.public_csv, copies_per_fact=args.copies_per_fact)
    frame = pd.DataFrame(rows)
    hard = frame[frame["split"] == "hard_val"]
    print(
        json.dumps(
            {
                "rows": int(len(frame)),
                "hard_val": int(len(hard)),
                "hard_val_by_profile": hard["question_profile"].value_counts().astype(int).to_dict(),
                "hard_val_taxonomy_families": sorted(hard["taxonomy_family"].unique().tolist()),
                "required_families": TAXONOMY_LABELS,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
