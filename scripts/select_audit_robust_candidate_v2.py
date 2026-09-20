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

from scripts.build_guarded_correction_candidate import build_guarded_correction_candidate
from scripts.build_multi_min_ensemble import build_multi_min_ensemble
from scripts.validate_candidate_on_hard_v5 import score_hard_v5, summarize_scores


DEFAULT_ROBUST_DIRS = (
    "outputs/candidate_audit_robust_hybrid_v2",
    "outputs/candidate_audit_robust_old_targeted_v2",
    "outputs/candidate_audit_robust_profile_v2",
)
OLD_PROFILES = {"who", "where", "when", "count", "generic"}
NEW_PROFILES = {"which_list", "what_property", "by_whom", "title_name", "definition"}


def _passes(summary: dict[str, Any], baseline: dict[str, Any], *, min_delta: float = 0.010) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["overall_ap"] < baseline["overall_ap"] + min_delta:
        reasons.append("overall_ap_delta")
    for profile, base_ap in baseline["ap_by_profile"].items():
        delta = summary["ap_by_profile"].get(profile, 0.0) - base_ap
        if profile in OLD_PROFILES and delta < -0.005:
            reasons.append(f"old_profile_regression:{profile}")
        if profile in NEW_PROFILES and delta < -0.005:
            reasons.append(f"new_profile_regression:{profile}")
        if profile in {"who", "where", "when"} and delta < -0.0001:
            reasons.append(f"typed_regression:{profile}")
    public_family = "public_replay_long_clean_audit"
    if (
        summary["ap_by_audit_fixture_family"].get(public_family, 0.0)
        < baseline["ap_by_audit_fixture_family"].get(public_family, 0.0) - 0.002
    ):
        reasons.append("public_replay_long_clean_audit_regression")
    if summary["clean_audit_correct_mean_score"] > baseline["clean_audit_correct_mean_score"] + 0.005:
        reasons.append("clean_audit_correct_score_up")
    if summary["clean_audit_wrong_mean_score"] < baseline["clean_audit_wrong_mean_score"] - 0.005:
        reasons.append("clean_audit_wrong_score_down")
    if summary["noisy_correct_mean_score"] > baseline["noisy_correct_mean_score"] + 0.000001:
        reasons.append("noisy_correct_not_improved")
    if summary["neutral_wrong_mean_score"] < baseline["neutral_wrong_mean_score"] - 0.000001:
        reasons.append("neutral_wrong_not_improved")
    family = "fallback_bad_status_text_only"
    if summary["ap_by_audit_fixture_family"].get(family, 0.0) < baseline["ap_by_audit_fixture_family"].get(family, 0.0) - 0.005:
        reasons.append("fallback_bad_status_regression")
    return not reasons, reasons


def _copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _write_diagnostics(result: dict[str, Any], output_md: str | Path) -> None:
    lines = [
        "# Candidate audit robust v2 diagnostics",
        "",
        f"- Baseline hard_v5 AP: {result['baseline']['overall_ap']:.6f}",
        f"- Accepted candidates: {sum(1 for item in result['candidates'] if item.get('accepted'))}",
        f"- Selected: `{result['selected']['candidate_dir']}`",
        f"- Selected AP: {result['selected']['overall_ap']:.6f}",
        "",
        "## Candidate Pool",
    ]
    for item in result["candidates"]:
        lines.append(
            f"- {Path(item['candidate_dir']).name}: AP {item['overall_ap']:.6f}, "
            f"accepted={item['accepted']}, reasons={item['rejection_reasons']}"
        )
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_audit_robust_candidate_v2(
    dataset_path: str | Path = "data/raw/synthetic_validation_hard_v5.jsonl",
    *,
    baseline_dir: str | Path = "outputs/candidate_min_old_full_targeted_lite_v1",
    robust_dirs: list[str | Path] | None = None,
    output_dir: str | Path = "outputs/candidate_audit_robust_v2",
    work_root: str | Path = "outputs/audit_robust_v2_pool",
) -> dict[str, Any]:
    robust_dirs = robust_dirs or list(DEFAULT_ROBUST_DIRS)
    baseline_frame, baseline_scores, fixture_summary = score_hard_v5(dataset_path, baseline_dir)
    baseline_summary = summarize_scores(baseline_frame, baseline_scores, candidate_dir=baseline_dir, fixture_summary=fixture_summary)
    pool_root = Path(work_root)
    pool_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for robust_dir in robust_dirs:
        robust_path = Path(robust_dir)
        if not robust_path.exists():
            continue
        guarded_path = pool_root / f"guarded_{robust_path.name}"
        build_guarded_correction_candidate(baseline_dir, robust_path, guarded_path)
        pool_specs = [("guarded_correction", guarded_path)]
        min_path = pool_root / f"min_diag_{robust_path.name}"
        build_multi_min_ensemble(baseline_dir, robust_path, min_path)
        pool_specs.append(("diagnostic_min", min_path))
        for kind, candidate_path in pool_specs:
            frame, scores, fs = score_hard_v5(dataset_path, candidate_path)
            summary = summarize_scores(frame, scores, candidate_dir=candidate_path, baseline_scores=baseline_scores, fixture_summary=fs)
            accepted, reasons = _passes(summary, baseline_summary)
            if kind == "diagnostic_min":
                accepted = False
                reasons = sorted(set(reasons + ["diagnostic_min_not_selectable"]))
            summary.update({"pool_kind": kind, "accepted": accepted, "rejection_reasons": reasons})
            candidates.append(summary)
    accepted = [item for item in candidates if item["accepted"]]
    if accepted:
        selected = max(accepted, key=lambda item: item["overall_ap"])
        _copytree_clean(Path(selected["candidate_dir"]), Path(output_dir))
    else:
        selected = dict(baseline_summary)
        selected.update({"candidate_dir": str(baseline_dir), "accepted": False, "rejection_reasons": ["no_candidate_passed"]})
        _copytree_clean(Path(baseline_dir), Path(output_dir))
    result = {
        "dataset_path": str(dataset_path),
        "baseline": baseline_summary,
        "candidates": candidates,
        "selected": selected,
        "output_dir": str(output_dir),
        "accepted": bool(accepted),
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/hard_v5_acceptance.summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("outputs/candidate_audit_robust_v2_hard_v5.summary.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_diagnostics(result, "outputs/diagnostics_candidate_audit_robust_v2.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best guarded audit-robust v2 candidate on hard_v5.")
    parser.add_argument("--dataset-path", default="data/raw/synthetic_validation_hard_v5.jsonl")
    parser.add_argument("--baseline-dir", default="outputs/candidate_min_old_full_targeted_lite_v1")
    parser.add_argument("--robust-dir", action="append", dest="robust_dirs")
    parser.add_argument("--output-dir", default="outputs/candidate_audit_robust_v2")
    args = parser.parse_args()
    print(
        json.dumps(
            select_audit_robust_candidate_v2(
                args.dataset_path,
                baseline_dir=args.baseline_dir,
                robust_dirs=args.robust_dirs,
                output_dir=args.output_dir,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
