from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.evaluate import AUDIT_COLUMNS, TrackingRuntimeVerifier, _audit_columns, _audit_source_key
from guardian_of_truth.guardian import GuardianOfTruth


DEFAULT_BAD_STATUSES = {
    "http_429",
    "timeout",
    "http_400",
    "connection_error",
    "invalid_json",
    "local_rate_limited",
    "cache_miss",
    "missing_api_key",
}

RATE_PROFILES = {
    "default": {
        "target_rpm": None,
        "target_tpm": None,
        "sleep_after_request_sec": 0.0,
        "sleep_after_429_sec": 65.0,
        "sleep_after_timeout_sec": 5.0,
        "max_attempts_per_row": None,
    },
    "slow-safe": {
        "target_rpm": 4,
        "target_tpm": 2500,
        "sleep_after_request_sec": 15.0,
        "sleep_after_429_sec": 90.0,
        "sleep_after_timeout_sec": 10.0,
        "max_attempts_per_row": 3,
    },
}


class PrefillRuntimeVerifier(TrackingRuntimeVerifier):
    def __init__(
        self,
        *args,
        retry_statuses: set[str],
        max_attempts_per_row: int,
        sleep_after_429_sec: float,
        sleep_after_timeout_sec: float,
        sleep_after_request_sec: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, allow_runtime_wait=True, **kwargs)
        self.retry_statuses = retry_statuses
        self.max_attempts_per_row = max_attempts_per_row
        self.sleep_after_429_sec = sleep_after_429_sec
        self.sleep_after_timeout_sec = sleep_after_timeout_sec
        self.sleep_after_request_sec = sleep_after_request_sec
        self.last_attempt_count = 0
        self.last_error: str | None = None

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        audit = AuditPayload.neutral(status="unknown", mode="runtime", model_name=self.settings.runtime_model, ok=False)
        for attempt in range(self.max_attempts_per_row):
            self.last_attempt_count = attempt + 1
            audit = super().verify(prompt, answer, mode="runtime")
            self.last_error = None if audit.ok or audit.status == "partial_json" else audit.status
            if audit.status in {"missing_api_key", "cache_miss"}:
                return audit
            if self.sleep_after_request_sec > 0:
                time.sleep(self.sleep_after_request_sec)
            if audit.status not in self.retry_statuses:
                return audit
            if attempt + 1 >= self.max_attempts_per_row:
                return audit
            if audit.status == "http_429":
                time.sleep(self.sleep_after_429_sec)
            elif audit.status == "timeout":
                time.sleep(self.sleep_after_timeout_sec)
        return audit


def _row_key(row: pd.Series) -> str:
    return _audit_source_key(row.get("prompt"), row.get("model_answer"))


def _good_statuses(frame: pd.DataFrame, bad_statuses: set[str]) -> pd.Series:
    if "audit_status" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return ~frame["audit_status"].fillna("unknown").astype(str).isin(bad_statuses)


def _merge_base_rows(source: pd.DataFrame, base: pd.DataFrame | None, bad_statuses: set[str]) -> dict[str, dict[str, Any]]:
    if base is None or base.empty:
        return {}
    base = base.copy()
    base["_audit_key"] = base.apply(_row_key, axis=1)
    good = base[_good_statuses(base, bad_statuses)]
    return {str(row["_audit_key"]): row.drop(labels=["_audit_key"]).to_dict() for _, row in good.iterrows()}


def _merge_existing_output_rows(existing: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if existing is None or existing.empty:
        return {}
    existing = existing.copy()
    existing["_audit_key"] = existing.apply(_row_key, axis=1)
    return {str(row["_audit_key"]): row.drop(labels=["_audit_key"]).to_dict() for _, row in existing.iterrows()}


def _attempt_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"total_attempts": 0, "retried_rows": 0}
    for row in rows:
        attempt_count = row.get("audit_attempt_count")
        try:
            attempt_count_int = int(float(attempt_count))
        except (TypeError, ValueError):
            attempt_count_int = 0
        counts["total_attempts"] += attempt_count_int
        if attempt_count_int > 1:
            counts["retried_rows"] += 1
    return counts


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        return {}
    counts = pd.Series([str(row.get("audit_status", "unknown")) for row in rows]).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _ok_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    ok_count = sum(str(row.get("audit_status")) in {"ok", "partial_json"} for row in rows)
    return ok_count / len(rows)


def _latency_quantiles(frame: pd.DataFrame) -> dict[str, float]:
    if "audit_latency_sec" not in frame.columns:
        return {}
    values = pd.to_numeric(frame["audit_latency_sec"], errors="coerce").dropna()
    if values.empty:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {name: float(values.quantile(q)) for name, q in {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}.items()}


def _write_checkpoint(output: Path, rows_by_key: dict[str, dict[str, Any]], source_keys: list[str]) -> None:
    rows = [rows_by_key[key] for key in source_keys if key in rows_by_key]
    pd.DataFrame(rows).to_csv(output, index=False)


def prefill_public_audit(
    csv_path: str | Path,
    *,
    output_path: str | Path,
    base_audit_csv: str | Path | None = None,
    model_dir: str | Path = "model",
    max_passes: int = 5,
    max_attempts_per_row: int = 2,
    target_ok_ratio: float = 0.90,
    checkpoint_every: int = 10,
    bad_statuses: set[str] | None = None,
    sleep_after_429_sec: float = 65.0,
    sleep_after_timeout_sec: float = 5.0,
    sleep_after_request_sec: float = 0.0,
    rate_profile: str = "default",
    max_minutes: float | None = None,
    runtime_prompt_version: str | None = None,
    limit: int | None = None,
    compare_slice_csv: str | Path | None = None,
) -> pd.DataFrame:
    bad_statuses = bad_statuses or set(DEFAULT_BAD_STATUSES)
    source = pd.read_csv(csv_path).reset_index(drop=True)
    if limit is not None:
        source = source.head(limit).reset_index(drop=True)
    source["_audit_key"] = source.apply(_row_key, axis=1)
    source_keys = [str(key) for key in source["_audit_key"]]
    base = pd.read_csv(base_audit_csv) if base_audit_csv and Path(base_audit_csv).exists() else None
    rows_by_key = _merge_base_rows(source, base, bad_statuses)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = pd.read_csv(output)
        rows_by_key.update(_merge_existing_output_rows(existing))
    verifier = PrefillRuntimeVerifier(
        retry_statuses=bad_statuses,
        max_attempts_per_row=max_attempts_per_row,
        sleep_after_429_sec=sleep_after_429_sec,
        sleep_after_timeout_sec=sleep_after_timeout_sec,
        sleep_after_request_sec=sleep_after_request_sec,
    )
    if rate_profile == "slow-safe":
        verifier.settings.target_rpm = 4
        verifier.settings.target_tpm = 2500
        verifier.rate_limiter.rpm = 4
        verifier.rate_limiter.tpm = 2500
    if runtime_prompt_version:
        verifier.settings.prompt_version = runtime_prompt_version
    started = time.monotonic()

    for pass_idx in range(max_passes):
        if max_minutes is not None and (time.monotonic() - started) / 60.0 >= max_minutes:
            break
        current_rows = [rows_by_key[key] for key in source_keys if key in rows_by_key]
        if len(current_rows) == len(source_keys) and _ok_ratio(current_rows) >= target_ok_ratio:
            break

        retry_rows = []
        for _, row in source.iterrows():
            key = str(row["_audit_key"])
            existing = rows_by_key.get(key)
            if existing is None or str(existing.get("audit_status", "unknown")) in bad_statuses:
                retry_rows.append(row)

        if not retry_rows:
            break

        for n, row in enumerate(tqdm(retry_rows, desc=f"audit-pass-{pass_idx + 1}"), start=1):
            if max_minutes is not None and (time.monotonic() - started) / 60.0 >= max_minutes:
                break
            key = str(row["_audit_key"])
            row_started = time.perf_counter()
            audit = verifier.verify(str(row["prompt"]), str(row["model_answer"]), mode="runtime")
            latency = time.perf_counter() - row_started
            enriched = row.drop(labels=["_audit_key"]).to_dict()
            enriched.update(
                _audit_columns(
                    audit,
                    getattr(verifier, "last_audit_source", "unknown"),
                    audit_latency_sec=latency,
                    audit_would_timeout=False,
                    score_path="audit_only",
                )
            )
            enriched.update(
                {
                    "audit_attempt_count": verifier.last_attempt_count,
                    "audit_last_error": verifier.last_error,
                    "audit_prefill_pass": pass_idx + 1,
                    "audit_collected_at": pd.Timestamp.utcnow().isoformat(),
                    "audit_rate_profile": rate_profile,
                }
            )
            rows_by_key[key] = enriched
            if n % checkpoint_every == 0:
                _write_checkpoint(output, rows_by_key, source_keys)

        _write_checkpoint(output, rows_by_key, source_keys)

    _write_checkpoint(output, rows_by_key, source_keys)
    frame = pd.read_csv(output)
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "unique_keys": int(frame.apply(_row_key, axis=1).nunique()) if {"prompt", "model_answer"}.issubset(frame.columns) else int(len(frame)),
        "status_counts": _status_counts(frame.to_dict(orient="records")),
        "ok_ratio": float(_ok_ratio(frame.to_dict(orient="records"))),
        "ok_partial_json_ratio": float(_ok_ratio(frame.to_dict(orient="records"))),
        "attempt_counts": _attempt_counts(frame.to_dict(orient="records")),
        "latency_quantiles": _latency_quantiles(frame),
        "source_counts": frame["audit_source"].fillna("unknown").value_counts().astype(int).to_dict()
        if "audit_source" in frame.columns
        else {},
        "bad_rows_remaining": int(frame["audit_status"].fillna("unknown").astype(str).isin(bad_statuses).sum())
        if "audit_status" in frame.columns
        else int(len(frame)),
        "rate_profile": rate_profile,
        "runtime_prompt_version": runtime_prompt_version,
        "limit": limit,
        "output_path": str(output),
        "base_audit_csv": str(base_audit_csv) if base_audit_csv else None,
    }
    if "is_hallucination" in frame.columns and "is_hallucination_proba" in frame.columns:
        scored = frame[["is_hallucination", "is_hallucination_proba"]].dropna()
        if len(scored) == len(frame) and scored["is_hallucination"].nunique() > 1:
            summary["pr_auc"] = float(average_precision_score(scored["is_hallucination"], scored["is_hallucination_proba"]))
    if compare_slice_csv and Path(compare_slice_csv).exists():
        compare = pd.read_csv(compare_slice_csv).head(len(frame))
        summary["compare_slice_csv"] = str(compare_slice_csv)
        summary["compare_status_counts"] = _status_counts(compare.to_dict(orient="records"))
        summary["compare_ok_partial_json_ratio"] = float(_ok_ratio(compare.to_dict(orient="records")))
        if "is_hallucination" in compare.columns and "is_hallucination_proba" in compare.columns:
            scored = compare[["is_hallucination", "is_hallucination_proba"]].dropna()
            if len(scored) == len(compare) and scored["is_hallucination"].nunique() > 1:
                summary["compare_pr_auc"] = float(average_precision_score(scored["is_hallucination"], scored["is_hallucination_proba"]))
        if "pr_auc" in summary and "compare_pr_auc" in summary:
            summary["delta_pr_auc_vs_compare"] = float(summary["pr_auc"] - summary["compare_pr_auc"])
        summary["delta_ok_partial_json_ratio_vs_compare"] = float(
            summary["ok_partial_json_ratio"] - summary["compare_ok_partial_json_ratio"]
        )
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a higher-quality runtime audit source with targeted retries.")
    parser.add_argument("--csv-path", default="data/bench/knowledge_bench_public.csv")
    parser.add_argument("--base-audit-csv", default=None)
    parser.add_argument("--output-path", default="outputs/public_scored_audit_rich_v4.csv")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--max-passes", type=int, default=5)
    parser.add_argument("--max-attempts-per-row", type=int, default=2)
    parser.add_argument("--target-ok-ratio", type=float, default=0.90)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sleep-after-429-sec", type=float, default=65.0)
    parser.add_argument("--sleep-after-timeout-sec", type=float, default=5.0)
    parser.add_argument("--sleep-after-request-sec", type=float, default=0.0)
    parser.add_argument("--rate-profile", choices=sorted(RATE_PROFILES), default="default")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--bad-statuses", nargs="*", default=sorted(DEFAULT_BAD_STATUSES))
    parser.add_argument("--runtime-prompt-version", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--compare-slice-csv",
        default=None,
        help="Optional same-slice v4/vbaseline probe CSV; summary will include useful-ratio and AP deltas.",
    )
    args = parser.parse_args()
    profile = RATE_PROFILES[args.rate_profile]
    max_attempts_per_row = (
        int(profile["max_attempts_per_row"])
        if profile["max_attempts_per_row"] is not None and args.max_attempts_per_row == parser.get_default("max_attempts_per_row")
        else args.max_attempts_per_row
    )
    sleep_after_request_sec = (
        float(profile["sleep_after_request_sec"])
        if args.sleep_after_request_sec == parser.get_default("sleep_after_request_sec")
        else args.sleep_after_request_sec
    )
    sleep_after_429_sec = (
        float(profile["sleep_after_429_sec"])
        if args.sleep_after_429_sec == parser.get_default("sleep_after_429_sec")
        else args.sleep_after_429_sec
    )
    sleep_after_timeout_sec = (
        float(profile["sleep_after_timeout_sec"])
        if args.sleep_after_timeout_sec == parser.get_default("sleep_after_timeout_sec")
        else args.sleep_after_timeout_sec
    )

    frame = prefill_public_audit(
        args.csv_path,
        output_path=args.output_path,
        base_audit_csv=args.base_audit_csv,
        model_dir=args.model_dir,
        max_passes=args.max_passes,
        max_attempts_per_row=max_attempts_per_row,
        target_ok_ratio=args.target_ok_ratio,
        checkpoint_every=args.checkpoint_every,
        bad_statuses=set(args.bad_statuses),
        sleep_after_429_sec=sleep_after_429_sec,
        sleep_after_timeout_sec=sleep_after_timeout_sec,
        sleep_after_request_sec=sleep_after_request_sec,
        rate_profile=args.rate_profile,
        max_minutes=args.max_minutes,
        runtime_prompt_version=args.runtime_prompt_version,
        limit=args.limit,
        compare_slice_csv=args.compare_slice_csv,
    )
    print(json.dumps({"rows": len(frame), "status_counts": _status_counts(frame.to_dict(orient="records"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
