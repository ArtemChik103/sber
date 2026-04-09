from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from guardian_of_truth.api_client import ApiSettings
from guardian_of_truth.utils import iter_jsonl


def _count_jsonl(path: str | Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def _minutes_for_requests(requests_count: int, rpm: int) -> float:
    if requests_count <= 0 or rpm <= 0:
        return 0.0
    return requests_count / rpm


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate pipeline wall time under current Groq free-plan settings.")
    parser.add_argument("--seed-path", default="data/raw/seed_qa.jsonl")
    parser.add_argument("--synthetic-path", default="data/raw/synthetic_factual_data.jsonl")
    parser.add_argument("--public-csv", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--planned-groq-negatives", type=int, default=300)
    args = parser.parse_args()

    settings = ApiSettings.from_yaml()
    seed_rows = _count_jsonl(args.seed_path) if Path(args.seed_path).exists() else 0
    synthetic_rows = _count_jsonl(args.synthetic_path) if Path(args.synthetic_path).exists() else 0
    public_rows = len(pd.read_csv(args.public_csv)) if Path(args.public_csv).exists() else 0

    projected_synthetic_rows = max(synthetic_rows, seed_rows * 2 + args.planned_groq_negatives)
    groq_neg_minutes = _minutes_for_requests(args.planned_groq_negatives, settings.target_rpm)
    api_feature_minutes = _minutes_for_requests(projected_synthetic_rows, settings.target_rpm)
    full_public_minutes = _minutes_for_requests(public_rows, settings.target_rpm)

    print(f"seed_rows={seed_rows}")
    print(f"synthetic_rows={synthetic_rows}")
    print(f"target_rpm={settings.target_rpm}")
    print(f"planned_groq_negatives={args.planned_groq_negatives}")
    print(f"groq_negative_generation_min_wall_time={groq_neg_minutes:.1f}")
    print(f"projected_synthetic_rows={projected_synthetic_rows}")
    print(f"api_feature_audit_min_wall_time={api_feature_minutes:.1f}")
    print("local_logreg_fit_estimate_min=0.1-2.0")
    print(f"public_rows={public_rows}")
    print(f"full_public_scoring_min_wall_time={full_public_minutes:.1f}")


if __name__ == "__main__":
    main()
