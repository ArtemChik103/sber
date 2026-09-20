from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from guardian_of_truth.claims import extract_entities, extract_numbers, extract_years
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.utils import iter_jsonl, sha256_hexdigest, write_jsonl


WRONG_ENTITIES = ["Альберт Эйнштейн", "Мария Кюри", "Уильям Шекспир", "Никола Тесла", "Исаак Ньютон"]
WRONG_NUMBERS = ["7", "12", "42", "100", "365"]
WRONG_YEARS = ["1492", "1703", "1812", "1917", "1961", "2001"]


def _record_text(record: dict[str, Any]) -> tuple[str, str]:
    prompt = str(record.get("prompt") or record.get("question") or "")
    answer = str(record.get("answer") or record.get("model_answer") or "")
    return prompt, answer


def _wrong_entity(answer: str) -> str:
    entities = extract_entities(answer)
    replacement = next((candidate for candidate in WRONG_ENTITIES if candidate not in answer), WRONG_ENTITIES[0])
    if entities:
        return answer.replace(entities[0], replacement, 1)
    return f"{replacement}. {answer}"


def _wrong_number(answer: str) -> str:
    numbers = extract_numbers(answer)
    replacement = next((candidate for candidate in WRONG_NUMBERS if candidate not in numbers), WRONG_NUMBERS[0])
    if numbers:
        return re.sub(re.escape(numbers[0]), replacement, answer, count=1)
    return f"{answer} Это произошло {replacement} раз."


def _wrong_year(answer: str) -> str:
    years = extract_years(answer)
    replacement = next((candidate for candidate in WRONG_YEARS if candidate not in years), WRONG_YEARS[0])
    if years:
        return answer.replace(years[0], replacement, 1)
    return f"{answer} Это произошло в {replacement} году."


def _false_tail(answer: str) -> str:
    return f"{answer.rstrip('.')} . Дополнительно известно, что это произошло в 1492 году в Париже."


def generate(records: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    extractor = FeatureExtractor()
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for source_idx, record in enumerate(records):
        prompt, answer = _record_text(record)
        if not prompt or not answer:
            continue
        profile = extractor._question_profile(prompt)
        base_meta = {
            "prompt": prompt,
            "source": record.get("source", "hard_seed"),
            "source_idx": source_idx,
            "question_profile": profile,
        }
        variants = [
            ("hard_correct_short", answer.split(".")[0].strip() + ".", 0, "trusted_seed"),
            ("hard_long_supported_correct", answer, 0, "trusted_seed"),
            ("hard_markdown_supported_correct", f"## Ответ\n\n{answer}", 0, "trusted_seed_markdown"),
            ("hard_context_supported_correct", f"{answer} Это следует из базового факта, указанного в источнике.", 0, "trusted_seed_context"),
            ("hard_answer_only_correct", answer.split(",")[0].split(".")[0].strip(), 0, "trusted_seed_short"),
            ("hard_wrong_entity", _wrong_entity(answer), 1, "wrong_entity"),
            ("hard_wrong_year", _wrong_year(answer), 1, "wrong_year"),
            ("hard_wrong_number", _wrong_number(answer), 1, "wrong_number"),
            ("hard_correct_core_false_tail", _false_tail(answer), 1, "false_tail"),
            ("hard_unsupported_tail", f"{answer} Это широко считается доказанным фактом без дополнительных источников.", 1, "unsupported_tail"),
            ("hard_wrong_relation", f"Нет, правильный ответ связан с {WRONG_ENTITIES[source_idx % len(WRONG_ENTITIES)]}.", 1, "wrong_relation"),
            ("hard_vague_non_answer", "Точный ответ зависит от источника и контекста.", 1, "vague_non_answer"),
            ("hard_near_miss_entity", f"{_wrong_entity(answer)} Вероятно, это близкий вариант ответа.", 1, "near_miss_entity"),
        ]
        for variant_type, variant_answer, label, failure_mode in variants:
            item = {
                **base_meta,
                "answer": variant_answer,
                "label": label,
                "variant_type": variant_type,
                "generation_rule": variant_type,
                "expected_failure_mode": failure_mode,
                "example_id": sha256_hexdigest(prompt, variant_answer, variant_type),
            }
            if label == 0:
                positives.append(item)
            else:
                negatives.append(item)
    if limit is None:
        return positives + negatives
    target_pos = limit // 2
    target_neg = limit - target_pos
    return _round_robin_by_variant(positives, target_pos) + _round_robin_by_variant(negatives, target_neg)


def _round_robin_by_variant(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_variant.setdefault(str(item["variant_type"]), []).append(item)
    output: list[dict[str, Any]] = []
    variant_names = sorted(by_variant)
    offset = 0
    while len(output) < limit and any(by_variant.values()):
        name = variant_names[offset % len(variant_names)]
        if by_variant[name]:
            output.append(by_variant[name].pop(0))
        offset += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic hard-negative/positive factual examples.")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", default="data/raw/synthetic_validation_hard_evidence_v1.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.source:
        records.extend(iter_jsonl(path))
    generated = generate(records, limit=args.limit)
    write_jsonl(args.output, generated)
    counts: dict[str, int] = {}
    for item in generated:
        counts[item["variant_type"]] = counts.get(item["variant_type"], 0) + 1
    print(json.dumps({"output": args.output, "rows": len(generated), "variant_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
