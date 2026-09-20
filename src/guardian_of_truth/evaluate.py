from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from guardian_of_truth.api_client import AuditPayload, GroqVerifier
from guardian_of_truth.evidence import EvidenceRetriever
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.guardian import GuardianOfTruth
from guardian_of_truth.refute_overlay import SUPPORTED_REFUTE_OVERLAY_POLICIES
from guardian_of_truth.utils import sha256_hexdigest

WHO_HINTS = ("кто", "who")
WHEN_HINTS = ("когда", "в каком году", "what year", "when")
WHERE_HINTS = ("где", "в какой стране", "в каком городе", "where")
COUNT_HINTS = ("сколько", "how many", "how much")
AUDIT_VALUE_FIELDS = ("h", "n", "e", "r", "u", "c", "x", "q", "s", "m", "sem", "we", "wn", "ue", "bt", "conf")
AUDIT_COLUMNS = tuple(f"audit_{field}" for field in AUDIT_VALUE_FIELDS) + (
    "audit_status",
    "audit_ok",
    "audit_cached",
    "audit_model_name",
    "audit_mode",
    "audit_source",
    "audit_latency_sec",
    "audit_would_timeout",
    "score_path",
)
EVIDENCE_COLUMNS = (
    "evidence_status",
    "evidence_retrieval_latency_sec",
    "evidence_retrieval_hit_count",
    "evidence_top_bm25_score",
    "evidence_top_evidence_overlap",
    "evidence_core_supported",
    "evidence_core_refuted",
    "evidence_entity_refuted_count",
    "evidence_number_refuted_count",
    "evidence_year_refuted_count",
    "evidence_tail_unsupported_entity_count",
    "evidence_tail_unsupported_number_count",
    "evidence_answer_entity_not_in_evidence_ratio",
    "evidence_answer_number_not_in_evidence_ratio",
    "evidence_aligned_hit_count",
    "evidence_aligned_year_refuted_count",
    "evidence_aligned_number_refuted_count",
    "evidence_answer_year_in_prompt",
    "evidence_answer_number_in_prompt",
    "evidence_aligned_title_entity_match",
    "evidence_answer_core_entity_count",
    "evidence_answer_core_entity_in_aligned_evidence_count",
    "evidence_answer_core_entity_missing_from_aligned_evidence_count",
    "evidence_aligned_evidence_entity_count",
    "evidence_aligned_alternative_entity_count",
    "evidence_aligned_entity_mismatch_candidate",
    "evidence_aligned_entity_confidence",
    "evidence_claim_supported_count",
    "evidence_claim_refuted_count",
    "evidence_claim_unknown_count",
    "evidence_compact_json",
)
REFUTE_OVERLAY_COLUMNS = (
    "base_is_hallucination_proba",
    "refute_overlay_policy",
    "refute_overlay_reason",
    "refute_overlay_expected_kind",
    "refute_overlay_delta",
)


class TrackingRuntimeVerifier(GroqVerifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_audit: AuditPayload | None = None
        self.last_audit_status = "unknown"
        self.last_audit_source = "unknown"
        self.last_audit_latency_sec: float | None = None
        self.last_source_audit_would_timeout: bool | None = None

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        started = time.perf_counter()
        audit = super().verify(prompt, answer, mode=mode)
        self.last_audit_latency_sec = time.perf_counter() - started
        self.last_source_audit_would_timeout = None
        self.last_audit = audit
        self.last_audit_status = audit.status
        self.last_audit_source = "sqlite_cache" if audit.cached else "live"
        return audit


class CacheOnlyRuntimeVerifier(TrackingRuntimeVerifier):
    def __init__(self, *args, cache_miss_policy: str = "fallback", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache_miss_policy = cache_miss_policy

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        audit = self.cached_audit(prompt, answer, mode="runtime")
        self.last_audit_latency_sec = 0.0
        self.last_source_audit_would_timeout = None
        if audit is not None:
            self.last_audit = audit
            self.last_audit_status = audit.status
            self.last_audit_source = "sqlite_cache"
            return audit
        audit = AuditPayload.neutral(
            status="cache_miss",
            mode="runtime",
            model_name=self.settings.runtime_model,
            ok=self.cache_miss_policy == "neutral",
        )
        self.last_audit = audit
        self.last_audit_status = audit.status
        self.last_audit_source = "cache_miss"
        return audit


class SourceCsvRuntimeVerifier(TrackingRuntimeVerifier):
    def __init__(
        self,
        *args,
        audit_source_csv: str | Path,
        cache_miss_policy: str = "fallback",
        ignore_source_timeout: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, allow_runtime_wait=False, **kwargs)
        self.cache_miss_policy = cache_miss_policy
        self.ignore_source_timeout = ignore_source_timeout
        self.audit_by_key, self.audit_meta_by_key = _load_audit_source_csv(audit_source_csv)

    def verify(self, prompt: str, answer: str, mode: str = "runtime") -> AuditPayload:
        key = _audit_source_key(prompt, answer)
        audit = self.audit_by_key.get(key)
        meta = self.audit_meta_by_key.get(key, {})
        if audit is None:
            audit = AuditPayload.neutral(
                status="cache_miss",
                mode="runtime",
                model_name=self.settings.runtime_model,
                ok=self.cache_miss_policy == "neutral",
            )
            self.last_audit_source = "cache_miss"
            self.last_audit_latency_sec = 0.0
            self.last_source_audit_would_timeout = None
        else:
            self.last_audit_source = "source_csv"
            latency = _clean_csv_value(meta.get("audit_latency_sec"))
            self.last_audit_latency_sec = float(latency) if latency is not None else 0.0
            would_timeout = meta.get("audit_would_timeout")
            self.last_source_audit_would_timeout = (
                None if self.ignore_source_timeout or would_timeout is None else _parse_bool(would_timeout)
            )
        self.last_audit = audit
        self.last_audit_status = audit.status
        return audit


def _audit_source_key(prompt: Any, answer: Any) -> str:
    return sha256_hexdigest(prompt, answer)


def _clean_csv_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    value = _clean_csv_value(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _audit_from_row(row: pd.Series) -> AuditPayload:
    payload: dict[str, Any] = {}
    for field in AUDIT_VALUE_FIELDS:
        value = _clean_csv_value(row.get(f"audit_{field}"))
        if value is not None:
            payload[field] = value
    payload.update(
        {
            "status": _clean_csv_value(row.get("audit_status")) or "ok",
            "ok": _parse_bool(row.get("audit_ok"), default=True),
            "cached": _parse_bool(row.get("audit_cached"), default=False),
            "model_name": _clean_csv_value(row.get("audit_model_name")),
            "mode": _clean_csv_value(row.get("audit_mode")) or "runtime",
        }
    )
    return AuditPayload.model_validate(payload)


def _load_audit_source_csv(path: str | Path) -> tuple[dict[str, AuditPayload], dict[str, dict[str, Any]]]:
    frame = pd.read_csv(path)
    required = {"prompt", "model_answer", "audit_status"}
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Audit source CSV is missing required columns: {missing_text}")
    audits: dict[str, AuditPayload] = {}
    meta: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = _audit_source_key(row["prompt"], row["model_answer"])
        audits[key] = _audit_from_row(row)
        meta[key] = {
            "audit_latency_sec": row.get("audit_latency_sec"),
            "audit_would_timeout": row.get("audit_would_timeout"),
            "score_path": row.get("score_path"),
        }
    return audits, meta


def _audit_columns(
    audit: AuditPayload | None,
    source: str,
    *,
    audit_latency_sec: float | None = None,
    audit_would_timeout: bool | None = None,
    score_path: str | None = None,
) -> dict[str, Any]:
    if audit is None:
        values: dict[str, Any] = {column: None for column in AUDIT_COLUMNS}
        values["audit_status"] = "unknown"
        values["audit_source"] = source
        values["audit_latency_sec"] = audit_latency_sec
        values["audit_would_timeout"] = audit_would_timeout
        values["score_path"] = score_path
        return values
    values = {f"audit_{field}": getattr(audit, field) for field in AUDIT_VALUE_FIELDS}
    values.update(
        {
            "audit_status": audit.status,
            "audit_ok": audit.ok,
            "audit_cached": audit.cached,
            "audit_model_name": audit.model_name,
            "audit_mode": audit.mode,
            "audit_source": source,
            "audit_latency_sec": audit_latency_sec,
            "audit_would_timeout": audit_would_timeout,
            "score_path": score_path,
        }
    )
    return values


def _evidence_columns(evidence_audit: Any | None) -> dict[str, Any]:
    if evidence_audit is None:
        return {column: None for column in EVIDENCE_COLUMNS} | {"evidence_status": "missing"}
    return {
        "evidence_status": evidence_audit.status,
        "evidence_retrieval_latency_sec": evidence_audit.retrieval_latency_sec,
        "evidence_retrieval_hit_count": evidence_audit.retrieval_hit_count,
        "evidence_top_bm25_score": evidence_audit.top_bm25_score,
        "evidence_top_evidence_overlap": evidence_audit.top_evidence_overlap,
        "evidence_core_supported": evidence_audit.core_supported,
        "evidence_core_refuted": evidence_audit.core_refuted,
        "evidence_entity_refuted_count": evidence_audit.entity_refuted_count,
        "evidence_number_refuted_count": evidence_audit.number_refuted_count,
        "evidence_year_refuted_count": evidence_audit.year_refuted_count,
        "evidence_tail_unsupported_entity_count": evidence_audit.tail_unsupported_entity_count,
        "evidence_tail_unsupported_number_count": evidence_audit.tail_unsupported_number_count,
        "evidence_answer_entity_not_in_evidence_ratio": evidence_audit.answer_entity_not_in_evidence_ratio,
        "evidence_answer_number_not_in_evidence_ratio": evidence_audit.answer_number_not_in_evidence_ratio,
        "evidence_aligned_hit_count": evidence_audit.aligned_hit_count,
        "evidence_aligned_year_refuted_count": evidence_audit.aligned_year_refuted_count,
        "evidence_aligned_number_refuted_count": evidence_audit.aligned_number_refuted_count,
        "evidence_answer_year_in_prompt": evidence_audit.answer_year_in_prompt,
        "evidence_answer_number_in_prompt": evidence_audit.answer_number_in_prompt,
        "evidence_aligned_title_entity_match": evidence_audit.aligned_title_entity_match,
        "evidence_answer_core_entity_count": evidence_audit.answer_core_entity_count,
        "evidence_answer_core_entity_in_aligned_evidence_count": evidence_audit.answer_core_entity_in_aligned_evidence_count,
        "evidence_answer_core_entity_missing_from_aligned_evidence_count": evidence_audit.answer_core_entity_missing_from_aligned_evidence_count,
        "evidence_aligned_evidence_entity_count": evidence_audit.aligned_evidence_entity_count,
        "evidence_aligned_alternative_entity_count": evidence_audit.aligned_alternative_entity_count,
        "evidence_aligned_entity_mismatch_candidate": evidence_audit.aligned_entity_mismatch_candidate,
        "evidence_aligned_entity_confidence": evidence_audit.aligned_entity_confidence,
        "evidence_claim_supported_count": evidence_audit.claim_supported_count,
        "evidence_claim_refuted_count": evidence_audit.claim_refuted_count,
        "evidence_claim_unknown_count": evidence_audit.claim_unknown_count,
        "evidence_compact_json": evidence_audit.compact_json,
    }


def _refute_overlay_columns(guardian: GuardianOfTruth, proba: float) -> dict[str, Any]:
    base = getattr(guardian, "last_base_is_hallucination_proba", None)
    if base is None:
        base = proba
    return {
        "base_is_hallucination_proba": float(base),
        "refute_overlay_policy": getattr(guardian, "last_refute_overlay_policy", "disabled"),
        "refute_overlay_reason": getattr(guardian, "last_refute_overlay_reason", "disabled"),
        "refute_overlay_expected_kind": getattr(guardian, "last_refute_overlay_expected_kind", "none"),
        "refute_overlay_delta": float(getattr(guardian, "last_refute_overlay_delta", 0.0)),
    }


def _latency_summary(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
    }


def _question_profile(prompt: str) -> str:
    return FeatureExtractor()._question_profile(prompt)


def _stable_dev_slice(df: pd.DataFrame, size: int, *, salt: str = "balanced") -> pd.DataFrame:
    keyed = df.copy()
    keyed["_slice_key"] = keyed.apply(
        lambda row: sha256_hexdigest(salt, row.get("prompt"), row.get("model_answer"), row.get("is_hallucination")),
        axis=1,
    )
    if "is_hallucination" not in keyed.columns:
        return keyed.sort_values("_slice_key").head(size).drop(columns="_slice_key").reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    for _, chunk in keyed.groupby("is_hallucination"):
        part_size = max(1, round(size * len(chunk) / len(keyed)))
        sampled_parts.append(chunk.sort_values("_slice_key").head(min(part_size, len(chunk))))
    return pd.concat(sampled_parts, ignore_index=True).head(size).drop(columns="_slice_key").reset_index(drop=True)


def _typed_dev_slice(df: pd.DataFrame, size: int) -> pd.DataFrame:
    typed = df[df["prompt"].map(lambda prompt: _question_profile(str(prompt)) != "generic")].copy()
    fallback = df[df["prompt"].map(lambda prompt: _question_profile(str(prompt)) == "generic")].copy()
    if len(typed) >= size:
        return _stable_dev_slice(typed, size, salt="typed")
    if typed.empty:
        return _stable_dev_slice(df, size, salt="typed")
    fill_size = max(0, size - len(typed))
    filler = _stable_dev_slice(fallback, fill_size, salt="typed-fill") if fill_size else fallback.head(0)
    combined = pd.concat([typed, filler], ignore_index=True)
    return _stable_dev_slice(combined, min(size, len(combined)), salt="typed")


def _prepare_frame(
    df: pd.DataFrame,
    limit: int | None = None,
    *,
    dev_slice_size: int | None = None,
    slice_name: str = "balanced",
) -> pd.DataFrame:
    if dev_slice_size is not None:
        if slice_name == "typed":
            return _typed_dev_slice(df, dev_slice_size)
        return _stable_dev_slice(df, dev_slice_size, salt=slice_name)
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
    model_dir: str | Path | None = None,
    limit: int | None = None,
    dev_slice_size: int | None = None,
    slice_name: str = "balanced",
    resume_from_checkpoint: bool = False,
    checkpoint_every: int = 25,
    cache_only_api: bool = False,
    cache_miss_policy: str = "fallback",
    audit_source_csv: str | Path | None = None,
    ignore_source_timeout: bool = False,
    runtime_prompt_version: str | None = None,
    evidence_db_path: str | Path | None = None,
    retriever_version: str | None = None,
    refute_overlay_policy: str = "disabled",
) -> pd.DataFrame:
    source = pd.read_csv(csv_path)
    source = _prepare_frame(source, limit=limit, dev_slice_size=dev_slice_size, slice_name=slice_name)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_dir = Path(model_dir or "model")
    if not (resolved_model_dir / "detector.joblib").exists():
        raise FileNotFoundError(
            f"Model artifact is missing: {resolved_model_dir / 'detector.joblib'}. "
            "Run training first or pass --model-dir with promoted artifacts."
        )

    if resume_from_checkpoint and output.exists():
        scored = pd.read_csv(output)
        start_idx = len(scored)
        rows: list[dict[str, object]] = scored.to_dict(orient="records")
    else:
        start_idx = 0
        rows = []

    if audit_source_csv is not None:
        verifier = SourceCsvRuntimeVerifier(
            audit_source_csv=audit_source_csv,
            cache_miss_policy=cache_miss_policy,
            ignore_source_timeout=ignore_source_timeout,
        )
    elif cache_only_api:
        verifier = CacheOnlyRuntimeVerifier(allow_runtime_wait=False, cache_miss_policy=cache_miss_policy)
    else:
        verifier = TrackingRuntimeVerifier(allow_runtime_wait=True)
    if runtime_prompt_version:
        verifier.settings.prompt_version = runtime_prompt_version
    evidence_retriever = EvidenceRetriever(evidence_db_path, retriever_version=retriever_version) if evidence_db_path else None
    guardian = GuardianOfTruth(
        verifier=verifier,
        model_dir=resolved_model_dir,
        evidence_retriever=evidence_retriever,
        refute_overlay_policy=refute_overlay_policy,
    )
    for idx in tqdm(range(start_idx, len(source)), desc="score-public"):
        row = source.iloc[idx]
        result = guardian.score(str(row["prompt"]), str(row["model_answer"]))
        enriched = row.to_dict()
        enriched.update(
            {
                "is_hallucination_proba": result.is_hallucination_proba,
                "predict_proba": result.is_hallucination_proba,
                "pred_is_hallucination": result.is_hallucination,
                "t_total_sec": result.t_total_sec,
                "t_model_sec": result.t_model_sec,
                "t_overhead_sec": result.t_overhead_sec,
            }
        )
        enriched.update(
            _audit_columns(
                getattr(guardian.verifier, "last_audit", None),
                getattr(guardian.verifier, "last_audit_source", "unknown"),
                audit_latency_sec=getattr(guardian, "last_audit_latency_sec", None),
                audit_would_timeout=getattr(guardian, "last_audit_would_timeout", None),
                score_path=getattr(guardian, "last_score_path", None),
            )
        )
        enriched.update(_evidence_columns(getattr(guardian, "last_evidence_audit", None)))
        enriched.update(_refute_overlay_columns(guardian, result.is_hallucination_proba))
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
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dev-slice-size", type=int, default=None)
    parser.add_argument("--slice-name", choices=["balanced", "typed"], default="balanced")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--cache-only-api", action="store_true", help="Use only cached runtime audits; cache misses do not call Groq.")
    parser.add_argument(
        "--audit-source-csv",
        default=None,
        help="Read runtime audits from an audit-rich CSV keyed by sha256(prompt, model_answer); takes priority over SQLite/live mode.",
    )
    parser.add_argument(
        "--cache-miss-policy",
        choices=["fallback", "neutral"],
        default="fallback",
        help="With cache-only or audit-source CSV, score cache misses through fallback or through the main model with a neutral audit.",
    )
    parser.add_argument(
        "--ignore-source-timeout",
        action="store_true",
        help="With --audit-source-csv, ignore audit_would_timeout from the source CSV for diagnostics.",
    )
    parser.add_argument(
        "--runtime-prompt-version",
        default=None,
        help="Override runtime verifier prompt version for this run without editing configs/api.yaml.",
    )
    parser.add_argument("--evidence-db-path", default=None, help="Optional local SQLite FTS5 evidence KB.")
    parser.add_argument(
        "--retriever-version",
        default=None,
        help="Optional retriever experiment version. Defaults to promoted fts5_rules_v2.",
    )
    parser.add_argument(
        "--refute-overlay-policy",
        choices=sorted(SUPPORTED_REFUTE_OVERLAY_POLICIES),
        default="disabled",
        help="Optional strict refute overlay policy. Disabled by default for backward compatibility.",
    )
    args = parser.parse_args()

    frame = run_evaluation(
        args.csv_path,
        output_path=args.output_path,
        model_dir=args.model_dir,
        limit=args.limit,
        dev_slice_size=args.dev_slice_size,
        slice_name=args.slice_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        checkpoint_every=args.checkpoint_every,
        cache_only_api=args.cache_only_api,
        cache_miss_policy=args.cache_miss_policy,
        audit_source_csv=args.audit_source_csv,
        ignore_source_timeout=args.ignore_source_timeout,
        runtime_prompt_version=args.runtime_prompt_version,
        evidence_db_path=args.evidence_db_path,
        retriever_version=args.retriever_version,
        refute_overlay_policy=args.refute_overlay_policy,
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
