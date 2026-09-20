from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_multi_min_ensemble import build_multi_min_ensemble
from scripts.validate_candidate_on_hard_v4 import score_hard_v4, summarize_scores


DEFAULT_ROBUST_DIRS = (
    "outputs/candidate_audit_robust_hybrid_v1",
    "outputs/candidate_audit_robust_old_targeted_v1",
    "outputs/candidate_audit_robust_text_targeted_v1",
)


def _passes(summary: dict[str, Any], baseline: dict[str, Any], *, min_delta: float = 0.010) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["overall_ap"] < baseline["overall_ap"] + min_delta:
        reasons.append("overall_ap_delta")
    for profile, base_ap in baseline["ap_by_profile"].items():
        delta = summary["ap_by_profile"].get(profile, 0.0) - base_ap
        if delta < -0.005:
            reasons.append(f"profile_regression:{profile}")
        if profile in {"who", "where", "when"} and delta < -0.0001:
            reasons.append(f"typed_regression:{profile}")
    if summary["noisy_correct_mean_score"] > baseline["noisy_correct_mean_score"] + 0.005:
        reasons.append("noisy_correct_score_up")
    if summary["neutral_wrong_mean_score"] < baseline["neutral_wrong_mean_score"] - 0.005:
        reasons.append("neutral_wrong_score_down")
    family = "fallback_bad_status_text_only"
    if summary["ap_by_audit_fixture_family"].get(family, 0.0) < baseline["ap_by_audit_fixture_family"].get(family, 0.0) - 0.005:
        reasons.append("fallback_bad_status_regression")
    return not reasons, reasons


def _copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def select_audit_robust_candidate(
    dataset_path: str | Path = "data/raw/synthetic_validation_hard_v4.jsonl",
    *,
    baseline_dir: str | Path = "outputs/candidate_min_old_full_targeted_lite_v1",
    robust_dirs: list[str | Path] | None = None,
    output_dir: str | Path = "outputs/candidate_audit_robust_v1",
    work_root: str | Path = "outputs/audit_robust_pool",
) -> dict[str, Any]:
    robust_dirs = robust_dirs or list(DEFAULT_ROBUST_DIRS)
    baseline_frame, baseline_scores, fixture_summary = score_hard_v4(dataset_path, baseline_dir)
    baseline_summary = summarize_scores(baseline_frame, baseline_scores, candidate_dir=baseline_dir, fixture_summary=fixture_summary)
    pool_root = Path(work_root)
    pool_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for robust_dir in robust_dirs:
        robust_path = Path(robust_dir)
        if not robust_path.exists():
            continue
        pool_specs = [
            ("standalone", robust_path, None, robust_path),
            ("min_ablation_old_full_robust", pool_root / f"min_candidate_ablation_old_full_{robust_path.name}", "outputs/candidate_ablation_old_full", robust_path),
            ("min_promoted_robust", pool_root / f"min_promoted_{robust_path.name}", baseline_dir, robust_path),
        ]
        lite = Path("outputs/candidate_old_full_targeted_lite_v1")
        if lite.exists():
            pool_specs.append(("min_ablation_lite_robust", pool_root / f"min_ablation_lite_{robust_path.name}", "outputs/candidate_ablation_old_full", robust_path))
        for kind, candidate_path, primary, robust in pool_specs:
            if primary is not None:
                extra = lite if kind == "min_ablation_lite_robust" and lite.exists() else None
                build_multi_min_ensemble(primary, robust, candidate_path, extra_secondary_dir=extra)
            frame, scores, fs = score_hard_v4(dataset_path, candidate_path)
            summary = summarize_scores(frame, scores, candidate_dir=candidate_path, baseline_scores=baseline_scores, fixture_summary=fs)
            accepted, reasons = _passes(summary, baseline_summary)
            summary.update({"pool_kind": kind, "accepted": accepted, "rejection_reasons": reasons})
            candidates.append(summary)
    accepted = [item for item in candidates if item["accepted"]]
    if accepted:
        selected = max(accepted, key=lambda item: item["overall_ap"])
        _copytree_clean(Path(selected["candidate_dir"]), Path(output_dir))
    else:
        selected = dict(baseline_summary)
        selected.update({"candidate_dir": str(baseline_dir), "accepted": False, "rejection_reasons": ["no_candidate_passed"]})
    result = {
        "dataset_path": str(dataset_path),
        "baseline": baseline_summary,
        "candidates": candidates,
        "selected": selected,
        "output_dir": str(output_dir),
        "accepted": bool(accepted),
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/hard_v4_acceptance.summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if accepted:
        Path("outputs/candidate_audit_robust_v1_hard_v4.summary.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best audit-robust production-loadable candidate on hard_v4.")
    parser.add_argument("--dataset-path", default="data/raw/synthetic_validation_hard_v4.jsonl")
    parser.add_argument("--baseline-dir", default="outputs/candidate_min_old_full_targeted_lite_v1")
    parser.add_argument("--robust-dir", action="append", dest="robust_dirs")
    parser.add_argument("--output-dir", default="outputs/candidate_audit_robust_v1")
    args = parser.parse_args()
    print(json.dumps(select_audit_robust_candidate(args.dataset_path, baseline_dir=args.baseline_dir, robust_dirs=args.robust_dirs, output_dir=args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
