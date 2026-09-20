from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from guardian_of_truth.utils import sha256_hexdigest


NUMBER_RE = re.compile(r"\b\d+\b")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _question_profile(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "кто" in prompt_lower or "who" in prompt_lower:
        return "who"
    if "когда" in prompt_lower or "в каком году" in prompt_lower or "when" in prompt_lower:
        return "when"
    if "где" in prompt_lower or "какая страна" in prompt_lower or "where" in prompt_lower:
        return "where"
    if "сколько" in prompt_lower or "how many" in prompt_lower or "how much" in prompt_lower:
        return "count"
    return "generic"


def _split(prompt: str, answer: str, variant_type: str) -> str:
    value = int(sha256_hexdigest(prompt, answer, variant_type)[:8], 16) / 0xFFFFFFFF
    if value < 0.60:
        return "train"
    if value < 0.80:
        return "gate_val"
    return "hard_val"


def _base_record(row: dict[str, Any], *, answer: str, label: int, variant_type: str, rule: str) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    reference_answer = str(row.get("reference_answer") or row.get("answer") or row.get("model_answer") or "")
    return {
        "prompt": prompt,
        "model_answer": answer,
        "answer": answer,
        "is_hallucination": int(label),
        "label": int(label),
        "reference_answer": reference_answer,
        "variant_type": variant_type,
        "question_profile": _question_profile(prompt),
        "split": _split(prompt, answer, variant_type),
        "generation_rule": rule,
        "source": row.get("source", "synthetic_validation_builder"),
    }


def _near_miss_number(answer: str) -> str | None:
    match = NUMBER_RE.search(answer)
    if not match:
        return None
    value = int(match.group(0))
    replacement = str(value + 1 if value < 2100 else value - 1)
    return answer[: match.start()] + replacement + answer[match.end() :]


def build_synthetic_validation(input_paths: list[str | Path], output_path: str | Path, *, min_hard_val: int = 500) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for path in input_paths:
        source_rows.extend(_read_jsonl(Path(path)))

    for row in source_rows:
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("model_answer") or row.get("answer") or "")
        if not prompt or not answer:
            continue
        label = int(row.get("is_hallucination", row.get("label", 0)))
        variant_type = str(row.get("variant_type") or ("positive" if label == 0 else "synthetic_negative"))
        candidates = [_base_record(row, answer=answer, label=label, variant_type=variant_type, rule="source_copy")]

        if label == 0:
            candidates.append(
                _base_record(
                    row,
                    answer=f"{answer} Дополнительно это произошло в 1899 году.",
                    label=1,
                    variant_type="false_tail_negative",
                    rule="append_false_tail_year",
                )
            )
            candidates.append(
                _base_record(
                    row,
                    answer=answer.split(".")[0].strip() or answer,
                    label=0,
                    variant_type="generated_exact_short_positive",
                    rule="first_sentence_positive",
                )
            )
            near_miss = _near_miss_number(answer)
            if near_miss and near_miss != answer:
                candidates.append(
                    _base_record(
                        row,
                        answer=near_miss,
                        label=1,
                        variant_type="near_miss_number_negative",
                        rule="increment_first_number",
                    )
                )
            candidates.append(
                _base_record(
                    row,
                    answer="Недостаточно информации для точного ответа.",
                    label=1,
                    variant_type="short_wrong_exact_negative",
                    rule="generic_refusal_negative",
                )
            )
        else:
            candidates.append(
                _base_record(
                    row,
                    answer=f"{answer} Также ответ включает неподтвержденную деталь.",
                    label=1,
                    variant_type="generated_hard_negative",
                    rule="preserve_negative_with_tail",
                )
            )

        for record in candidates:
            key = sha256_hexdigest(record["prompt"], record["model_answer"], record["variant_type"])
            if key not in seen:
                seen.add(key)
                records.append(record)

    if sum(record["split"] == "hard_val" for record in records) < min_hard_val:
        for record in records:
            if record["split"] == "train":
                record["split"] = "hard_val"
                if sum(item["split"] == "hard_val" for item in records) >= min_hard_val:
                    break

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic held-out adversarial validation JSONL.")
    parser.add_argument("--input-path", action="append", default=None)
    parser.add_argument("--output-path", default="data/raw/synthetic_validation_hard_v1.jsonl")
    parser.add_argument("--min-hard-val", type=int, default=500)
    args = parser.parse_args()
    input_paths = args.input_path or [
        "data/raw/synthetic_factual_data.jsonl",
        "data/raw/synthetic_factual_data_targeted_lite_v1.jsonl",
    ]
    records = build_synthetic_validation(input_paths, args.output_path, min_hard_val=args.min_hard_val)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["split"]] = counts.get(record["split"], 0) + 1
    print(json.dumps({"rows": len(records), "split_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
