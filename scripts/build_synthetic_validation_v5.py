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

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.utils import sha256_hexdigest
from scripts.build_synthetic_validation_v4 import AUDIT_KEYS, FALSE_TAILS, audit_fixture as v4_audit_fixture


OLD_PROFILES = ("who", "where", "when", "count", "generic")
NEW_PROFILES = ("which_list", "what_property", "by_whom", "title_name", "definition")
PROFILES = OLD_PROFILES + NEW_PROFILES

OLD_AUDIT_FIXTURE_FAMILIES = (
    "correct_long_high_u",
    "correct_high_wrong_fields_x0",
    "correct_typed_short_high_wrong_fields",
    "wrong_short_exact_neutral_audit",
    "wrong_core_neutral_audit",
    "wrong_false_tail_low_wrong_fields",
    "wrong_strong_contradiction",
    "correct_low_risk_control",
    "wrong_high_risk_control",
    "fallback_bad_status_text_only",
    "public_replay_long_clean_audit",
)
NEW_AUDIT_FIXTURE_FAMILIES = (
    "wrong_long_clean_audit_public_like",
    "correct_long_clean_audit_public_like",
    "wrong_core_zero_wrong_fields",
    "correct_verbose_zero_wrong_fields",
    "list_question_wrong_clean_audit",
    "list_question_correct_clean_audit",
    "property_question_wrong_clean_audit",
    "property_question_correct_clean_audit",
    "by_whom_wrong_clean_audit",
    "title_name_wrong_clean_audit",
)
AUDIT_FIXTURE_FAMILIES = OLD_AUDIT_FIXTURE_FAMILIES + NEW_AUDIT_FIXTURE_FAMILIES


BASE_FACTS: dict[str, list[dict[str, str]]] = {
    "who": [
        {"prompt": "Кто написал роман Евгений Онегин?", "reference_answer": "Александр Пушкин", "wrong": "Михаил Лермонтов"},
        {"prompt": "Who painted The Starry Night?", "reference_answer": "Vincent van Gogh", "wrong": "Claude Monet"},
        {"prompt": "Кто открыл пенициллин?", "reference_answer": "Александр Флеминг", "wrong": "Луи Пастер"},
        {"prompt": "Who wrote Pride and Prejudice?", "reference_answer": "Jane Austen", "wrong": "Charlotte Bronte"},
    ],
    "where": [
        {"prompt": "Где находится Колизей?", "reference_answer": "В Риме", "wrong": "В Афинах"},
        {"prompt": "Where is Machu Picchu located?", "reference_answer": "In Peru", "wrong": "In Mexico"},
        {"prompt": "В какой стране находится город Киото?", "reference_answer": "В Японии", "wrong": "В Южной Корее"},
        {"prompt": "Where is the Louvre Museum?", "reference_answer": "In Paris, France", "wrong": "In Berlin, Germany"},
    ],
    "when": [
        {"prompt": "Когда началась Первая мировая война?", "reference_answer": "В 1914 году", "wrong": "В 1939 году"},
        {"prompt": "When did Apollo 11 land on the Moon?", "reference_answer": "In 1969", "wrong": "In 1972"},
        {"prompt": "В каком году закончилась Вторая мировая война?", "reference_answer": "В 1945 году", "wrong": "В 1941 году"},
        {"prompt": "When was the Magna Carta sealed?", "reference_answer": "In 1215", "wrong": "In 1066"},
    ],
    "count": [
        {"prompt": "Сколько планет в Солнечной системе?", "reference_answer": "8", "wrong": "9"},
        {"prompt": "How many players are on a soccer team on the field?", "reference_answer": "11", "wrong": "10"},
        {"prompt": "Сколько дней в високосном году?", "reference_answer": "366", "wrong": "365"},
        {"prompt": "How many chambers are in the human heart?", "reference_answer": "4", "wrong": "3"},
    ],
    "generic": [
        {"prompt": "Что такое фотосинтез?", "reference_answer": "Фотосинтез - процесс, при котором растения используют свет для образования органических веществ из углекислого газа и воды.", "wrong": "Фотосинтез - это распад горных пород под действием ветра."},
        {"prompt": "What is gravity?", "reference_answer": "Gravity is the attraction between masses.", "wrong": "Gravity is a chemical reaction that releases oxygen."},
        {"prompt": "Что такое демократия?", "reference_answer": "Демократия - форма правления, при которой власть принадлежит народу или осуществляется через избранных представителей.", "wrong": "Демократия - это система наследственной абсолютной власти."},
        {"prompt": "What is evaporation?", "reference_answer": "Evaporation is the change of a liquid into vapor at its surface.", "wrong": "Evaporation is the freezing of vapor into solid crystals."},
    ],
    "which_list": [
        {"prompt": "Какие три страны входят в Скандинавию?", "reference_answer": "Норвегия, Швеция и Дания", "wrong": "Финляндия, Исландия и Германия"},
        {"prompt": "Which colors are on the French flag?", "reference_answer": "Blue, white, and red", "wrong": "Green, white, and orange"},
        {"prompt": "Какие океаны омывают Россию?", "reference_answer": "Северный Ледовитый и Тихий океаны", "wrong": "Индийский и Атлантический океаны"},
        {"prompt": "Which planets are gas giants?", "reference_answer": "Jupiter, Saturn, Uranus, and Neptune", "wrong": "Mercury, Venus, Earth, and Mars"},
    ],
    "what_property": [
        {"prompt": "Какова частота переменного тока в бытовой сети России?", "reference_answer": "50 Гц", "wrong": "60 Гц"},
        {"prompt": "What is the value of pi rounded to two decimals?", "reference_answer": "3.14", "wrong": "2.71"},
        {"prompt": "Каково название химического символа Au?", "reference_answer": "Золото", "wrong": "Серебро"},
        {"prompt": "What is the name of Earth's natural satellite?", "reference_answer": "The Moon", "wrong": "Phobos"},
    ],
    "by_whom": [
        {"prompt": "Кем была разработана периодическая таблица элементов?", "reference_answer": "Дмитрием Менделеевым", "wrong": "Исааком Ньютоном"},
        {"prompt": "By whom was the telephone patented in 1876?", "reference_answer": "Alexander Graham Bell", "wrong": "Thomas Edison"},
        {"prompt": "Кем был спроектирован Эйфелева башня?", "reference_answer": "Гюставом Эйфелем и его инженерами", "wrong": "Ле Корбюзье"},
        {"prompt": "By whom was Frankenstein written?", "reference_answer": "Mary Shelley", "wrong": "Emily Bronte"},
    ],
    "title_name": [
        {"prompt": "Какое название носит столица Австралии?", "reference_answer": "Канберра", "wrong": "Сидней"},
        {"prompt": "What title is given to the head of state in Japan?", "reference_answer": "Emperor", "wrong": "President"},
        {"prompt": "Как называется самая высокая гора мира?", "reference_answer": "Эверест", "wrong": "Килиманджаро"},
        {"prompt": "What name is given to the process plants use to make food from light?", "reference_answer": "Photosynthesis", "wrong": "Respiration"},
    ],
    "definition": [
        {"prompt": "Что такое инерция?", "reference_answer": "Инерция - свойство тела сохранять состояние покоя или равномерного движения.", "wrong": "Инерция - это превращение света в электрический ток."},
        {"prompt": "What is osmosis?", "reference_answer": "Osmosis is the movement of solvent through a semipermeable membrane toward higher solute concentration.", "wrong": "Osmosis is the splitting of atoms into lighter nuclei."},
        {"prompt": "Что такое метафора?", "reference_answer": "Метафора - переносное употребление слова или выражения по сходству.", "wrong": "Метафора - это строгая математическая теорема."},
        {"prompt": "What is a glacier?", "reference_answer": "A glacier is a large, persistent mass of ice that moves slowly.", "wrong": "A glacier is a dry desert wind."},
    ],
}

CONTEXTS = (
    "для краткой справки",
    "для проверки фактов",
    "для учебной карточки",
    "для энциклопедической заметки",
    "для устного ответа",
    "для редакторской сверки",
    "для контрольного вопроса",
    "для справочного блока",
)
TONES = ("точной", "нейтральной", "короткой", "проверочной", "академической", "ясной", "сжатой")

FAMILY_CONTEXT = {
    **{family: f"Проверьте hard_v4 семейство {family}." for family in OLD_AUDIT_FIXTURE_FAMILIES},
    "wrong_long_clean_audit_public_like": "Проверьте длинный public-like ответ с чистым аудитом и неверным ядром.",
    "correct_long_clean_audit_public_like": "Проверьте длинный public-like ответ с чистым аудитом и верным ядром.",
    "wrong_core_zero_wrong_fields": "Проверьте неверное ядро при нулевых wrong-field сигналах.",
    "correct_verbose_zero_wrong_fields": "Проверьте верный подробный ответ при нулевых wrong-field сигналах.",
    "list_question_wrong_clean_audit": "Проверьте list-вопрос с неверным чистым ответом.",
    "list_question_correct_clean_audit": "Проверьте list-вопрос с верным чистым ответом.",
    "property_question_wrong_clean_audit": "Проверьте property-вопрос с неверным чистым ответом.",
    "property_question_correct_clean_audit": "Проверьте property-вопрос с верным чистым ответом.",
    "by_whom_wrong_clean_audit": "Проверьте by-whom вопрос с неверным чистым ответом.",
    "title_name_wrong_clean_audit": "Проверьте title/name вопрос с неверным чистым ответом.",
}


def audit_fixture(family: str) -> dict[str, Any]:
    if family in OLD_AUDIT_FIXTURE_FAMILIES:
        return v4_audit_fixture(family)
    values = v4_audit_fixture("public_replay_long_clean_audit")
    if family in {"wrong_core_zero_wrong_fields", "correct_verbose_zero_wrong_fields"}:
        values.update({"h": 0.0, "u": 0.0, "we": 0.0, "wn": 0.0, "ue": 0.0, "bt": 0.0, "conf": 0.78, "c": 1.0})
    if family in {
        "list_question_wrong_clean_audit",
        "list_question_correct_clean_audit",
        "property_question_wrong_clean_audit",
        "property_question_correct_clean_audit",
        "by_whom_wrong_clean_audit",
        "title_name_wrong_clean_audit",
    }:
        values.update({"conf": 0.80, "c": 1.0})
    payload = AuditPayload.model_validate(values)
    dumped = payload.model_dump(include=set(AUDIT_KEYS))
    return {key: dumped[key] for key in AUDIT_KEYS}


def _public_keys(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    frame = pd.read_csv(path)
    return {sha256_hexdigest(row["prompt"], row["model_answer"]) for _, row in frame.iterrows()}


def _prompt(profile: str, base: str, ordinal: int) -> str:
    context = CONTEXTS[ordinal % len(CONTEXTS)]
    tone = TONES[(ordinal // len(CONTEXTS)) % len(TONES)]
    return f"{base} Ответьте {tone} формулировкой {context}."


def _long_correct(ref: str, profile: str) -> str:
    return f"{ref}. Это развернутый ответ на вопрос профиля {profile}; он сохраняет основной проверяемый факт и не добавляет противоречащих деталей."


def _long_wrong(wrong: str, profile: str) -> str:
    return f"{wrong}. Это развернутый ответ на вопрос профиля {profile}; формулировка выглядит справочной, но центральный проверяемый факт неверен."


def _answer(family: str, ref: str, wrong: str, ordinal: int, profile: str) -> tuple[str, int]:
    if family == "correct_long_high_u":
        return (_long_correct(ref, profile), 0)
    if family in {"correct_high_wrong_fields_x0", "correct_low_risk_control", "correct_typed_short_high_wrong_fields"}:
        return (ref, 0)
    if family in {"wrong_short_exact_neutral_audit", "wrong_high_risk_control"}:
        return (wrong, 1)
    if family == "wrong_core_neutral_audit":
        return (f"{wrong}. Формулировка выглядит уверенной, но основной факт неверен.", 1)
    if family == "wrong_false_tail_low_wrong_fields":
        return (f"{ref}. {FALSE_TAILS[ordinal % len(FALSE_TAILS)]}", 1)
    if family == "wrong_strong_contradiction":
        return (f"{wrong}. Это противоречит проверяемому ответу.", 1)
    if family in {"public_replay_long_clean_audit", "correct_long_clean_audit_public_like", "correct_verbose_zero_wrong_fields"}:
        return (_long_correct(ref, profile), 0)
    if family in {"wrong_long_clean_audit_public_like", "wrong_core_zero_wrong_fields"}:
        return (_long_wrong(wrong, profile), 1)
    if family in {"list_question_correct_clean_audit", "property_question_correct_clean_audit"}:
        return (ref, 0)
    if family in {
        "list_question_wrong_clean_audit",
        "property_question_wrong_clean_audit",
        "by_whom_wrong_clean_audit",
        "title_name_wrong_clean_audit",
    }:
        return (wrong, 1)
    if family == "fallback_bad_status_text_only":
        return (f"{wrong}. Ответ получен без надежной проверки источника.", 1)
    return (wrong, 1)


def _record(profile: str, fact: dict[str, str], family: str, ordinal: int) -> dict[str, Any]:
    prompt = f"{_prompt(profile, fact['prompt'], ordinal)} {FAMILY_CONTEXT[family]}"
    answer, label = _answer(family, fact["reference_answer"], fact["wrong"], ordinal, profile)
    return {
        "prompt": prompt,
        "answer": answer,
        "model_answer": answer,
        "label": int(label),
        "is_hallucination": int(label),
        "reference_answer": fact["reference_answer"],
        "variant_type": family,
        "taxonomy_family": family,
        "audit_fixture_family": family,
        "audit_fixture": audit_fixture(family),
        "question_profile": profile,
        "split": "hard_val",
        "generation_rule": f"hard_v5_{family}",
        "source": "synthetic_validation_hard_v5",
        "synthetic_id": sha256_hexdigest("hard-v5-id", profile, fact["prompt"], answer, family, ordinal)[:16],
    }


def build_synthetic_validation_v5(
    output_path: str | Path = "data/raw/synthetic_validation_hard_v5.jsonl",
    *,
    public_csv: str | Path | None = "data/bench/knowledge_bench_public.csv",
    copies_per_fact: int = 20,
) -> list[dict[str, Any]]:
    public = _public_keys(public_csv)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for profile in PROFILES:
        for ordinal in range(copies_per_fact):
            for fact in BASE_FACTS[profile]:
                for family in AUDIT_FIXTURE_FAMILIES:
                    row = _record(profile, fact, family, ordinal)
                    public_key = sha256_hexdigest(row["prompt"], row["model_answer"])
                    stable_key = sha256_hexdigest(row["prompt"], row["model_answer"], row["audit_fixture_family"])
                    if public_key in public or stable_key in seen:
                        continue
                    seen.add(stable_key)
                    rows.append(row)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public-failure-aware hard_v5 synthetic validation JSONL.")
    parser.add_argument("--output-path", default="data/raw/synthetic_validation_hard_v5.jsonl")
    parser.add_argument("--public-csv", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--copies-per-fact", type=int, default=20)
    args = parser.parse_args()
    rows = build_synthetic_validation_v5(args.output_path, public_csv=args.public_csv, copies_per_fact=args.copies_per_fact)
    frame = pd.DataFrame(rows)
    hard = frame[frame["split"] == "hard_val"]
    print(
        json.dumps(
            {
                "rows": int(len(frame)),
                "hard_val": int(len(hard)),
                "hard_val_by_profile": hard["question_profile"].value_counts().astype(int).to_dict(),
                "hard_val_by_audit_fixture_family": hard["audit_fixture_family"].value_counts().astype(int).to_dict(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
