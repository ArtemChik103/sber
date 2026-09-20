from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.guardian import GuardianOfTruth
from guardian_of_truth.utils import sha256_hexdigest


class JsonlAuditVerifier:
    def __init__(self, audit_by_key: dict[str, AuditPayload]) -> None:
        self.audit_by_key = audit_by_key
        self.settings = type("Settings", (), {"total_timeout_sec": 999.0})()
        self.last_audit: AuditPayload | None = None
        self.last_audit_source = "hard_v3_jsonl"
        self.last_audit_latency_sec = 0.0
        self.last_source_audit_would_timeout = False

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        audit = self.audit_by_key.get(
            sha256_hexdigest(prompt, answer),
            AuditPayload.neutral(status="ok", mode="runtime", model_name="hard_v3_synthetic", ok=True),
        )
        self.last_audit = audit
        return audit


def _read_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def _audit_for_row(row: pd.Series) -> AuditPayload:
    family = str(row.get("taxonomy_family") or row.get("variant_type") or "")
    label = int(row.get("is_hallucination", 0))
    if family in {"audit_strong_risk_but_low_score", "gate_underraises_wrong", "typed_short_wrong_missed", "where_who_when_regression"}:
        return AuditPayload(h=0.9, u=0.85, x=1.0, we=0.9, wn=0.8, ue=0.7, bt=0.6, conf=0.9, status="ok", ok=True, model_name="hard_v3_synthetic")
    if label == 1:
        return AuditPayload(h=0.65, u=0.7, x=0.0, we=0.55, wn=0.55, ue=0.5, bt=0.4, conf=0.8, status="ok", ok=True, model_name="hard_v3_synthetic")
    return AuditPayload(h=0.05, u=0.05, x=0.0, we=0.0, wn=0.0, ue=0.0, bt=0.0, conf=0.9, status="ok", ok=True, model_name="hard_v3_synthetic")


def _ap(y: pd.Series, scores: list[float]) -> float:
    if y.nunique() < 2:
        return 0.0
    return float(average_precision_score(y.astype(int), scores))


def _score(frame: pd.DataFrame, model_dir: str | Path) -> list[float]:
    audits = {
        sha256_hexdigest(row["prompt"], row["model_answer"]): _audit_for_row(row)
        for _, row in frame.iterrows()
    }
    guardian = GuardianOfTruth(verifier=JsonlAuditVerifier(audits), model_dir=model_dir)
    return [guardian.score(str(row["prompt"]), str(row["model_answer"])).is_hallucination_proba for _, row in frame.iterrows()]


def validate_rescue_rules_candidate(
    dataset_path: str | Path = "data/raw/synthetic_validation_hard_v3.jsonl",
    *,
    baseline_dir: str | Path = "outputs/candidate_min_old_full_targeted_lite_v1",
    candidate_dir: str | Path = "outputs/candidate_rescue_rules_v1",
    output_summary: str | Path = "outputs/candidate_rescue_rules_v1_hard_v3.summary.json",
    max_profile_regression: float = 0.005,
    typed_regression_epsilon: float = 1e-4,
) -> dict[str, Any]:
    frame = _read_jsonl(dataset_path)
    hard = frame[frame["split"] == "hard_val"].reset_index(drop=True)
    baseline_scores = _score(hard, baseline_dir)
    candidate_scores = _score(hard, candidate_dir)
    y = hard["is_hallucination"].astype(int)
    profile_rows = []
    rejected_profiles = []
    for profile, indices in hard.groupby("question_profile").groups.items():
        idx = list(indices)
        base_ap = _ap(y.iloc[idx], [baseline_scores[i] for i in idx])
        cand_ap = _ap(y.iloc[idx], [candidate_scores[i] for i in idx])
        delta = cand_ap - base_ap
        profile_rows.append({"profile": str(profile), "baseline_ap": base_ap, "candidate_ap": cand_ap, "delta_ap": delta, "rows": len(idx)})
        if delta < -max_profile_regression or (profile in {"who", "where", "when"} and delta < -typed_regression_epsilon):
            rejected_profiles.append(str(profile))
    deltas = [candidate - base for candidate, base in zip(candidate_scores, baseline_scores, strict=True)]
    summary = {
        "dataset_path": str(dataset_path),
        "hard_val_rows": int(len(hard)),
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "baseline_ap": _ap(y, baseline_scores),
        "candidate_ap": _ap(y, candidate_scores),
        "profile_ap": profile_rows,
        "max_abs_decrease": float(abs(min(0.0, min(deltas)))) if deltas else 0.0,
        "never_decreases_base": bool(all(delta >= -1e-12 for delta in deltas)),
        "rejected_profiles": rejected_profiles,
        "accepted": not rejected_profiles and all(delta >= -1e-12 for delta in deltas),
    }
    Path(output_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rescue-rules candidate on hard_v3 before public replay.")
    parser.add_argument("--dataset-path", default="data/raw/synthetic_validation_hard_v3.jsonl")
    parser.add_argument("--baseline-dir", default="outputs/candidate_min_old_full_targeted_lite_v1")
    parser.add_argument("--candidate-dir", default="outputs/candidate_rescue_rules_v1")
    parser.add_argument("--output-summary", default="outputs/candidate_rescue_rules_v1_hard_v3.summary.json")
    args = parser.parse_args()
    print(json.dumps(validate_rescue_rules_candidate(args.dataset_path, baseline_dir=args.baseline_dir, candidate_dir=args.candidate_dir, output_summary=args.output_summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
