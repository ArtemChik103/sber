from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _score_column(frame: pd.DataFrame) -> str:
    for column in ("is_hallucination_proba", "predict_proba"):
        if column in frame.columns:
            return column
    raise ValueError("CSV is missing a score column: expected is_hallucination_proba or predict_proba")


def _ap(frame: pd.DataFrame, score_column: str) -> float | None:
    if "is_hallucination" not in frame.columns:
        return None
    return float(average_precision_score(frame["is_hallucination"], frame[score_column]))


def verify_promotion(
    candidate_csv: str | Path,
    promoted_csv: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    candidate = pd.read_csv(candidate_csv)
    promoted = pd.read_csv(promoted_csv)
    candidate_score = _score_column(candidate)
    promoted_score = _score_column(promoted)

    if len(candidate) != len(promoted):
        raise ValueError(f"Row count mismatch: candidate={len(candidate)} promoted={len(promoted)}")

    candidate_values = candidate[candidate_score].to_numpy(dtype=np.float64)
    promoted_values = promoted[promoted_score].to_numpy(dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(candidate_values - promoted_values))) if len(candidate_values) else 0.0
    summary: dict[str, Any] = {
        "candidate_csv": str(candidate_csv),
        "promoted_csv": str(promoted_csv),
        "rows": int(len(candidate)),
        "candidate_score_column": candidate_score,
        "promoted_score_column": promoted_score,
        "candidate_pr_auc": _ap(candidate, candidate_score),
        "promoted_pr_auc": _ap(promoted, promoted_score),
        "max_abs_diff": max_abs_diff,
        "matches_exactly": bool(max_abs_diff == 0.0),
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify candidate/promoted score parity and public AP.")
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--promoted-csv", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    summary = verify_promotion(args.candidate_csv, args.promoted_csv, output_path=args.output_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["matches_exactly"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
