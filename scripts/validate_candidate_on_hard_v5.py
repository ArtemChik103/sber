from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.guardian import GuardianOfTruth
from guardian_of_truth.utils import sha256_hexdigest


class FixtureAuditVerifier:
    def __init__(self, audit_by_key: dict[str, AuditPayload], *, missing_policy: str = "error") -> None:
        self.audit_by_key = audit_by_key
        self.missing_policy = missing_policy
        self.settings = type("Settings", (), {"total_timeout_sec": 999.0})()
        self.last_audit: AuditPayload | None = None
        self.last_audit_source = "hard_v5_fixture"
        self.last_audit_latency_sec = 0.0
        self.last_source_audit_would_timeout = False

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        key = hard_v5_key(prompt, answer)
        audit = self.audit_by_key.get(key)
        if audit is None:
            if self.missing_policy == "neutral":
                audit = AuditPayload.neutral(status="fixture_missing", mode="runtime", model_name="hard_v5_fixture", ok=True)
            else:
                raise KeyError(f"Missing hard_v5 audit fixture for key {key}")
        self.last_audit = audit
        return audit


def hard_v5_key(prompt: Any, model_answer: Any) -> str:
    return sha256_hexdigest(prompt, model_answer)


def read_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def build_fixture_map(frame: pd.DataFrame) -> tuple[dict[str, AuditPayload], dict[str, Any]]:
    audit_by_key: dict[str, AuditPayload] = {}
    duplicates: list[str] = []
    missing = 0
    for _, row in frame.iterrows():
        key = hard_v5_key(row["prompt"], row["model_answer"])
        if key in audit_by_key:
            duplicates.append(key)
            continue
        fixture = row.get("audit_fixture")
        if not isinstance(fixture, dict):
            missing += 1
            continue
        audit_by_key[key] = AuditPayload.model_validate(fixture)
    return audit_by_key, {"duplicate_keys": len(duplicates), "missing_fixtures": missing, "unique_keys": len(audit_by_key)}


def _ap(y: pd.Series, scores: list[float]) -> float:
    if len(y) == 0 or y.nunique() < 2:
        return 0.0
    return float(average_precision_score(y.astype(int), scores))


def _group_ap(frame: pd.DataFrame, scores: list[float], column: str) -> dict[str, float]:
    y = frame["is_hallucination"].astype(int)
    values: dict[str, float] = {}
    for group, indices in frame.groupby(column).groups.items():
        idx = list(indices)
        values[str(group)] = _ap(y.iloc[idx], [scores[i] for i in idx])
    return values


def _mean_delta_by_group(frame: pd.DataFrame, scores: list[float], baseline_scores: list[float] | None, column: str) -> dict[str, float]:
    if baseline_scores is None:
        return {str(group): 0.0 for group in frame[column].dropna().unique()}
    deltas = np.asarray(scores, dtype=np.float64) - np.asarray(baseline_scores, dtype=np.float64)
    out: dict[str, float] = {}
    for group, indices in frame.groupby(column).groups.items():
        idx = list(indices)
        out[str(group)] = float(deltas[idx].mean()) if idx else 0.0
    return out


def score_hard_v5(
    dataset_path: str | Path,
    model_dir: str | Path,
    *,
    split: str = "hard_val",
    missing_policy: str = "error",
) -> tuple[pd.DataFrame, list[float], dict[str, Any]]:
    frame = read_jsonl(dataset_path)
    if split:
        frame = frame[frame["split"] == split].reset_index(drop=True)
    fixtures, fixture_summary = build_fixture_map(frame)
    if fixture_summary["duplicate_keys"]:
        raise ValueError(f"Duplicate hard_v5 fixture keys: {fixture_summary['duplicate_keys']}")
    if fixture_summary["missing_fixtures"] and missing_policy == "error":
        raise ValueError(f"Rows without audit_fixture: {fixture_summary['missing_fixtures']}")
    guardian = GuardianOfTruth(verifier=FixtureAuditVerifier(fixtures, missing_policy=missing_policy), model_dir=model_dir)
    scores = [guardian.score(str(row["prompt"]), str(row["model_answer"])).is_hallucination_proba for _, row in frame.iterrows()]
    return frame, scores, fixture_summary


def summarize_scores(
    frame: pd.DataFrame,
    scores: list[float],
    *,
    candidate_dir: str | Path,
    baseline_scores: list[float] | None = None,
    fixture_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y = frame["is_hallucination"].astype(int)
    score_array = np.asarray(scores, dtype=np.float64)
    noisy_correct = frame["audit_fixture_family"].isin(
        ["correct_long_high_u", "correct_high_wrong_fields_x0", "correct_typed_short_high_wrong_fields"]
    ) & (y == 0)
    neutral_wrong = frame["audit_fixture_family"].isin(
        ["wrong_short_exact_neutral_audit", "wrong_core_neutral_audit", "wrong_core_zero_wrong_fields"]
    ) & (y == 1)
    clean_families = {
        "public_replay_long_clean_audit",
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
    }
    clean = frame["audit_fixture_family"].isin(clean_families)
    clean_correct = clean & (y == 0)
    clean_wrong = clean & (y == 1)
    summary = {
        "candidate_dir": str(candidate_dir),
        "rows": int(len(frame)),
        "overall_ap": _ap(y, scores),
        "ap_by_profile": _group_ap(frame, scores, "question_profile"),
        "ap_by_taxonomy_family": _group_ap(frame, scores, "taxonomy_family"),
        "ap_by_audit_fixture_family": _group_ap(frame, scores, "audit_fixture_family"),
        "score_delta_by_label": _mean_delta_by_group(frame.assign(_label=y.astype(str)), scores, baseline_scores, "_label"),
        "score_delta_by_profile": _mean_delta_by_group(frame, scores, baseline_scores, "question_profile"),
        "clean_audit_correct_mean_score": float(score_array[clean_correct.to_numpy()].mean()) if clean_correct.any() else 0.0,
        "clean_audit_wrong_mean_score": float(score_array[clean_wrong.to_numpy()].mean()) if clean_wrong.any() else 0.0,
        "noisy_correct_mean_score": float(score_array[noisy_correct.to_numpy()].mean()) if noisy_correct.any() else 0.0,
        "neutral_wrong_mean_score": float(score_array[neutral_wrong.to_numpy()].mean()) if neutral_wrong.any() else 0.0,
        "score_min": float(score_array.min()) if len(score_array) else 0.0,
        "score_max": float(score_array.max()) if len(score_array) else 0.0,
        "fixture_summary": fixture_summary or {},
    }
    if baseline_scores is not None:
        summary["baseline_ap"] = _ap(y, baseline_scores)
        summary["delta_ap"] = summary["overall_ap"] - summary["baseline_ap"]
    return summary


def validate_candidate_on_hard_v5(
    dataset_path: str | Path = "data/raw/synthetic_validation_hard_v5.jsonl",
    *,
    candidate_dir: str | Path,
    output_summary: str | Path | None = None,
    baseline_dir: str | Path | None = None,
    missing_policy: str = "error",
) -> dict[str, Any]:
    frame, scores, fixture_summary = score_hard_v5(dataset_path, candidate_dir, missing_policy=missing_policy)
    baseline_scores = None
    if baseline_dir is not None:
        base_frame, baseline_scores, _ = score_hard_v5(dataset_path, baseline_dir, missing_policy=missing_policy)
        if len(base_frame) != len(frame):
            raise RuntimeError("Baseline and candidate hard_v5 frames differ.")
    summary = summarize_scores(
        frame,
        scores,
        candidate_dir=candidate_dir,
        baseline_scores=baseline_scores,
        fixture_summary=fixture_summary,
    )
    if output_summary is None:
        output_summary = Path("outputs") / f"{Path(candidate_dir).name}_hard_v5.summary.json"
    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(output_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a production-loadable candidate on hard_v5.")
    parser.add_argument("--dataset-path", default="data/raw/synthetic_validation_hard_v5.jsonl")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--missing-policy", choices=["error", "neutral"], default="error")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_candidate_on_hard_v5(
                args.dataset_path,
                candidate_dir=args.candidate_dir,
                baseline_dir=args.baseline_dir,
                output_summary=args.output_summary,
                missing_policy=args.missing_policy,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
