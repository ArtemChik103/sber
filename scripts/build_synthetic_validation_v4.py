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


PROFILES = ("who", "where", "when", "count", "generic")
AUDIT_FIXTURE_FAMILIES = (
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
AUDIT_KEYS = ("h", "n", "e", "r", "u", "c", "x", "q", "s", "m", "sem", "we", "wn", "ue", "bt", "conf", "status", "ok", "model_name", "mode")

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
FALSE_TAILS = (
    "В некоторых пересказах добавляют, что этот факт был подтвержден специальной комиссией в 2007 году, но это лишняя неподтвержденная деталь.",
    "A longer note sometimes says the answer was formally renamed by an international panel, which is not supported here.",
    "Также встречается утверждение о связи с Нобелевской премией, хотя для этого вопроса оно не подтверждено.",
    "Some summaries append a later expedition and archive number, but that added detail is not part of the verified answer.",
)


def audit_fixture(family: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "h": 0.04,
        "n": 0.02,
        "e": 0.02,
        "r": 0.92,
        "u": 0.03,
        "c": 1.0,
        "x": 0.0,
        "q": 0.90,
        "s": 0.90,
        "m": 0.02,
        "sem": 0.92,
        "we": 0.02,
        "wn": 0.02,
        "ue": 0.02,
        "bt": 0.02,
        "conf": 0.86,
        "status": "ok",
        "ok": True,
        "model_name": "hard_v4_fixture",
        "mode": "runtime",
    }
    overrides = {
        "correct_long_high_u": {"h": 0.72, "u": 0.88, "we": 0.62, "ue": 0.70, "bt": 0.66, "conf": 0.82, "c": 3.0},
        "correct_high_wrong_fields_x0": {"h": 0.35, "u": 0.76, "we": 0.86, "wn": 0.78, "ue": 0.74, "bt": 0.82, "x": 0.0, "conf": 0.88},
        "correct_typed_short_high_wrong_fields": {"h": 0.25, "u": 0.52, "we": 0.84, "wn": 0.82, "ue": 0.40, "bt": 0.58, "x": 0.0, "conf": 0.90},
        "wrong_short_exact_neutral_audit": {"h": 0.05, "u": 0.04, "we": 0.03, "wn": 0.03, "ue": 0.02, "bt": 0.02, "conf": 0.55},
        "wrong_core_neutral_audit": {"h": 0.10, "u": 0.08, "we": 0.08, "wn": 0.08, "ue": 0.06, "bt": 0.05, "conf": 0.58},
        "wrong_false_tail_low_wrong_fields": {"h": 0.18, "u": 0.22, "we": 0.10, "wn": 0.12, "ue": 0.18, "bt": 0.14, "conf": 0.68, "c": 3.0},
        "wrong_strong_contradiction": {"h": 0.93, "u": 0.88, "x": 1.0, "we": 0.94, "wn": 0.88, "ue": 0.78, "bt": 0.72, "conf": 0.94},
        "correct_low_risk_control": {},
        "wrong_high_risk_control": {"h": 0.88, "u": 0.82, "x": 1.0, "we": 0.88, "wn": 0.82, "ue": 0.70, "bt": 0.66, "conf": 0.90},
        "fallback_bad_status_text_only": {"status": "timeout", "ok": False, "h": 0.0, "u": 0.0, "we": 0.0, "wn": 0.0, "ue": 0.0, "bt": 0.0, "conf": 0.5},
        "public_replay_long_clean_audit": {"h": 0.0, "u": 0.0, "we": 0.0, "wn": 0.0, "ue": 0.0, "bt": 0.0, "conf": 0.72, "c": 3.0},
    }
    values.update(overrides[family])
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
    if profile == "generic":
        return f"{base} Дайте ответ {tone} формулировкой {context}."
    return f"{base} Ответьте {tone} формулировкой {context}."


def _answer(family: str, ref: str, wrong: str, ordinal: int, profile: str) -> tuple[str, int]:
    if family == "correct_long_high_u":
        return (
            f"{ref}. Это прямой ответ на вопрос; дополнительное пояснение остается в рамках проверенного факта и не меняет смысл.",
            0,
        )
    if family in {"correct_high_wrong_fields_x0", "correct_low_risk_control"}:
        return (ref, 0)
    if family == "correct_typed_short_high_wrong_fields":
        return (ref if profile != "generic" else f"{ref}.", 0)
    if family in {"wrong_short_exact_neutral_audit", "wrong_high_risk_control"}:
        return (wrong, 1)
    if family == "wrong_core_neutral_audit":
        return (f"{wrong}. Формулировка выглядит уверенной, но основной факт неверен.", 1)
    if family == "wrong_false_tail_low_wrong_fields":
        return (f"{ref}. {FALSE_TAILS[ordinal % len(FALSE_TAILS)]}", 1)
    if family == "wrong_strong_contradiction":
        return (f"{wrong}. Это противоречит проверяемому ответу.", 1)
    if family == "public_replay_long_clean_audit":
        if ordinal % 2 == 0:
            return (
                f"{ref}. Подробная формулировка повторяет основной факт, добавляет контекст вопроса и выглядит как длинная справочная заметка, но не вводит новый неверный факт.",
                0,
            )
        return (
            f"{wrong}. Подробная формулировка уверенно развивает ответ и выглядит как справочная заметка, но центральный факт остается неверным.",
            1,
        )
    return (f"{wrong}. Ответ получен без надежной проверки источника.", 1)


FAMILY_CONTEXT = {
    "correct_long_high_u": "Проверьте развернутую формулировку.",
    "correct_high_wrong_fields_x0": "Проверьте краткий ответ с возможным шумом аудита.",
    "correct_typed_short_high_wrong_fields": "Проверьте короткую typed-форму ответа.",
    "wrong_short_exact_neutral_audit": "Проверьте короткий вариант без пояснений.",
    "wrong_core_neutral_audit": "Проверьте уверенную формулировку основного факта.",
    "wrong_false_tail_low_wrong_fields": "Проверьте ответ с дополнительным хвостом.",
    "wrong_strong_contradiction": "Проверьте вариант с явным противоречием.",
    "correct_low_risk_control": "Проверьте контрольный правильный вариант.",
    "wrong_high_risk_control": "Проверьте контрольный неверный вариант.",
    "fallback_bad_status_text_only": "Проверьте вариант для текстового резервного пути.",
    "public_replay_long_clean_audit": "Проверьте длинный public-подобный ответ при чистом аудите.",
}


def _record(profile: str, fact: dict[str, str], family: str, ordinal: int) -> dict[str, Any]:
    prompt = f"{_prompt(profile, fact['prompt'], ordinal)} {FAMILY_CONTEXT[family]}"
    answer, label = _answer(family, fact["reference_answer"], fact["wrong"], ordinal, profile)
    split_hash = int(sha256_hexdigest("hard-v4", prompt, answer, family)[:8], 16) / 0xFFFFFFFF
    split = "hard_val" if split_hash >= 0.0 else "train"
    row = {
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
        "split": split,
        "generation_rule": f"hard_v4_{family}",
        "source": "synthetic_validation_hard_v4",
        "synthetic_id": sha256_hexdigest("hard-v4-id", profile, fact["prompt"], answer, family, ordinal)[:16],
    }
    return row


def build_synthetic_validation_v4(
    output_path: str | Path = "data/raw/synthetic_validation_hard_v4.jsonl",
    *,
    public_csv: str | Path | None = "data/bench/knowledge_bench_public.csv",
    copies_per_fact: int = 20,
) -> list[dict[str, Any]]:
    public = _public_keys(public_csv)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for profile, facts in BASE_FACTS.items():
        for ordinal in range(copies_per_fact):
            for fact in facts:
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
    parser = argparse.ArgumentParser(description="Build audit-fixture drift-hard synthetic validation v4 JSONL.")
    parser.add_argument("--output-path", default="data/raw/synthetic_validation_hard_v4.jsonl")
    parser.add_argument("--public-csv", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--copies-per-fact", type=int, default=20)
    args = parser.parse_args()
    rows = build_synthetic_validation_v4(args.output_path, public_csv=args.public_csv, copies_per_fact=args.copies_per_fact)
    frame = pd.DataFrame(rows)
    hard = frame[frame["split"] == "hard_val"]
    print(json.dumps({
        "rows": int(len(frame)),
        "hard_val": int(len(hard)),
        "hard_val_by_profile": hard["question_profile"].value_counts().astype(int).to_dict(),
        "hard_val_by_audit_fixture_family": hard["audit_fixture_family"].value_counts().astype(int).to_dict(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
