from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from guardian_of_truth.api_client import GroqVerifier
from guardian_of_truth.guardian import GuardianOfTruth
from guardian_of_truth.utils import sha256_hexdigest


def _latency_summary(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
    }


def _stable_dev_slice(df: pd.DataFrame, size: int) -> pd.DataFrame:
    keyed = df.copy()
    keyed["_slice_key"] = keyed.apply(
        lambda row: sha256_hexdigest(row.get("prompt"), row.get("model_answer"), row.get("is_hallucination")),
        axis=1,
    )
    if "is_hallucination" not in keyed.columns:
        return keyed.sort_values("_slice_key").head(size).drop(columns="_slice_key").reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    for _, chunk in keyed.groupby("is_hallucination"):
        part_size = max(1, round(size * len(chunk) / len(keyed)))
        sampled_parts.append(chunk.sort_values("_slice_key").head(min(part_size, len(chunk))))
    return pd.concat(sampled_parts, ignore_index=True).head(size).drop(columns="_slice_key").reset_index(drop=True)


def _prepare_frame(
    df: pd.DataFrame,
    limit: int | None = None,
    *,
    dev_slice_size: int | None = None,
) -> pd.DataFrame:
    if dev_slice_size is not None:
        return _stable_dev_slice(df, dev_slice_size)
    if limit is None or len(df) <= limit:
        return df.reset_index(drop=True)
    if "is_hallucination" not in df.columns:
        return df.head(limit).reset_index(drop=True)
    sampled_parts: list[pd.DataFrame] = []
    for _, chunk in df.groupby("is_hallucination"):
        part_size = max(1, round(limit * len(chunk) / len(df)))
        sampled_parts.append(chunk.sample(n=min(part_size, len(chunk)), random_state=42))
    return pd.concat(sampled_parts, ignore_index=True).head(limit).reset_index(drop=True)


def run_evaluation(
    csv_path: str | Path,
    *,
    output_path: str | Path,
    limit: int | None = None,
    dev_slice_size: int | None = None,
    resume_from_checkpoint: bool = False,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    source = pd.read_csv(csv_path)
    source = _prepare_frame(source, limit=limit, dev_slice_size=dev_slice_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if resume_from_checkpoint and output.exists():
        scored = pd.read_csv(output)
        start_idx = len(scored)
        rows: list[dict[str, object]] = scored.to_dict(orient="records")
    else:
        start_idx = 0
        rows = []

    guardian = GuardianOfTruth(verifier=GroqVerifier(allow_runtime_wait=True))
    for idx in tqdm(range(start_idx, len(source)), desc="score-public"):
        row = source.iloc[idx]
        result = guardian.score(str(row["prompt"]), str(row["model_answer"]))
        enriched = row.to_dict()
        enriched.update(
            {
                "is_hallucination_proba": result.is_hallucination_proba,
                "pred_is_hallucination": result.is_hallucination,
                "t_total_sec": result.t_total_sec,
                "t_model_sec": result.t_model_sec,
                "t_overhead_sec": result.t_overhead_sec,
            }
        )
        rows.append(enriched)
        if (idx + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(output, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential scorer for the public benchmark.")
    parser.add_argument("--csv-path", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--output-path", default="outputs/public_scored.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dev-slice-size", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    frame = run_evaluation(
        args.csv_path,
        output_path=args.output_path,
        limit=args.limit,
        dev_slice_size=args.dev_slice_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
        checkpoint_every=args.checkpoint_every,
    )

    if "is_hallucination" in frame.columns:
        pr_auc = average_precision_score(frame["is_hallucination"], frame["is_hallucination_proba"])
        print(f"PR-AUC: {pr_auc:.4f}")

    for metric_name, column in [
        ("total", "t_total_sec"),
        ("model", "t_model_sec"),
        ("overhead", "t_overhead_sec"),
    ]:
        stats = _latency_summary(frame[column])
        print(
            f"{metric_name} latency sec:"
            f" mean={stats['mean']:.4f}"
            f" p50={stats['p50']:.4f}"
            f" p95={stats['p95']:.4f}"
            f" p99={stats['p99']:.4f}"
        )


if __name__ == "__main__":
    main()
