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
from scripts.build_synthetic_validation_v5 import audit_fixture, build_synthetic_validation_v5


DIAGNOSTIC_FAMILIES = (
    "guarded_count_property_title_clean_correct",
    "guarded_count_property_title_clean_wrong",
    "guarded_generic_long_clean_correct",
    "guarded_generic_long_clean_wrong",
)

DIAGNOSTIC_FACTS = {
    "count": [
        {"prompt": "Сколько спутников у Марса?", "reference_answer": "2", "wrong": "3"},
        {"prompt": "How many strings does a standard violin have?", "reference_answer": "4", "wrong": "6"},
    ],
    "what_property": [
        {"prompt": "Какова температура кипения воды при нормальном давлении?", "reference_answer": "100 градусов Цельсия", "wrong": "80 градусов Цельсия"},
        {"prompt": "What is the value of the speed of light in vacuum approximately?", "reference_answer": "299,792,458 meters per second", "wrong": "150,000,000 meters per second"},
    ],
    "title_name": [
        {"prompt": "Как называется столица Канады?", "reference_answer": "Оттава", "wrong": "Торонто"},
        {"prompt": "What name is given to Earth's largest ocean?", "reference_answer": "Pacific Ocean", "wrong": "Atlantic Ocean"},
    ],
    "generic": [
        {"prompt": "Что такое испарение?", "reference_answer": "Испарение - переход вещества из жидкого состояния в газообразное.", "wrong": "Испарение - переход вещества из газа в твердое состояние."},
        {"prompt": "What is a comet?", "reference_answer": "A comet is an icy small Solar System body that releases gas and dust near the Sun.", "wrong": "A comet is a permanent artificial satellite launched from Earth."},
    ],
}


def _record(profile: str, fact: dict[str, str], family: str, ordinal: int) -> dict[str, Any]:
    is_correct = family.endswith("_correct")
    answer = fact["reference_answer"] if is_correct else fact["wrong"]
    if "long" in family:
        answer = (
            f"{answer}. This is a longer clean-audit diagnostic answer with a public-like explanatory tail; "
            "the key fact should dominate the rank without broad score rewriting."
        )
    label = 0 if is_correct else 1
    contexts = (
        "для справочного ответа",
        "для редакторской сверки",
        "для учебной карточки",
        "для контрольного вопроса",
        "для энциклопедической заметки",
        "для краткой проверки",
        "для фактического аудита",
        "для public-like replay диагностики",
    )
    prompt = (
        f"{fact['prompt']} Ответьте кратко и точно {contexts[ordinal % len(contexts)]}; "
        f"вариант формулировки {ordinal // len(contexts) + 1}. Диагностика hard_v6 для guarded public replay regression."
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "model_answer": answer,
        "label": label,
        "is_hallucination": label,
        "reference_answer": fact["reference_answer"],
        "variant_type": family,
        "taxonomy_family": family,
        "audit_fixture_family": family,
        "audit_fixture": audit_fixture("public_replay_long_clean_audit"),
        "question_profile": profile,
        "split": "hard_val",
        "generation_rule": f"hard_v6_{family}",
        "source": "synthetic_validation_hard_v6",
        "synthetic_id": sha256_hexdigest("hard-v6-id", profile, fact["prompt"], answer, family, ordinal)[:16],
    }


def build_synthetic_validation_v6(
    output_path: str | Path = "data/raw/synthetic_validation_hard_v6.jsonl",
    *,
    public_csv: str | Path | None = "data/bench/knowledge_bench_public.csv",
    copies_per_fact: int = 20,
    diagnostic_copies: int = 80,
) -> list[dict[str, Any]]:
    rows = build_synthetic_validation_v5(output_path, public_csv=public_csv, copies_per_fact=copies_per_fact)
    public_keys: set[str] = set()
    if public_csv and Path(public_csv).exists():
        public = pd.read_csv(public_csv)
        public_keys = {sha256_hexdigest(row["prompt"], row["model_answer"]) for _, row in public.iterrows()}
    seen = {sha256_hexdigest(row["prompt"], row["model_answer"], row["audit_fixture_family"]) for row in rows}
    for profile, facts in DIAGNOSTIC_FACTS.items():
        families = (
            ("guarded_generic_long_clean_correct", "guarded_generic_long_clean_wrong")
            if profile == "generic"
            else ("guarded_count_property_title_clean_correct", "guarded_count_property_title_clean_wrong")
        )
        for ordinal in range(diagnostic_copies):
            for fact in facts:
                for family in families:
                    row = _record(profile, fact, family, ordinal)
                    public_key = sha256_hexdigest(row["prompt"], row["model_answer"])
                    stable_key = sha256_hexdigest(row["prompt"], row["model_answer"], row["audit_fixture_family"])
                    if public_key in public_keys or stable_key in seen:
                        continue
                    seen.add(stable_key)
                    rows.append(row)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard_v6 diagnostics for guarded public replay regressions.")
    parser.add_argument("--output-path", default="data/raw/synthetic_validation_hard_v6.jsonl")
    parser.add_argument("--public-csv", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--copies-per-fact", type=int, default=20)
    parser.add_argument("--diagnostic-copies", type=int, default=80)
    args = parser.parse_args()
    rows = build_synthetic_validation_v6(
        args.output_path,
        public_csv=args.public_csv,
        copies_per_fact=args.copies_per_fact,
        diagnostic_copies=args.diagnostic_copies,
    )
    frame = pd.DataFrame(rows)
    print(
        json.dumps(
            {
                "rows": int(len(frame)),
                "diagnostic_families": frame[frame["audit_fixture_family"].isin(DIAGNOSTIC_FAMILIES)]["audit_fixture_family"].value_counts().astype(int).to_dict(),
                "profile_counts": frame["question_profile"].value_counts().astype(int).to_dict(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
