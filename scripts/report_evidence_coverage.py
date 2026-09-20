from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from guardian_of_truth.evidence import EvidenceRetriever, FORBIDDEN_SOURCE_COLUMNS
from guardian_of_truth.feature_extractor import FeatureExtractor


def _histogram(values: list[int]) -> dict[str, int]:
    buckets = Counter()
    for value in values:
        if value == 0:
            buckets["0"] += 1
        elif value == 1:
            buckets["1"] += 1
        elif value == 2:
            buckets["2"] += 1
        elif value <= 5:
            buckets["3-5"] += 1
        else:
            buckets["6+"] += 1
    return dict(buckets)


def build_coverage_report(csv_path: str | Path, db_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    forbidden = (FORBIDDEN_SOURCE_COLUMNS - {"is_hallucination"}) & set(frame.columns)
    if forbidden:
        raise ValueError(f"Coverage source contains forbidden columns: {sorted(forbidden)}")
    missing = {"prompt", "model_answer"} - set(frame.columns)
    if missing:
        raise ValueError(f"Coverage source is missing required columns: {sorted(missing)}")

    extractor = FeatureExtractor()
    retriever = EvidenceRetriever(db_path)
    rows: list[dict[str, Any]] = []
    hit_counts: list[int] = []
    latencies: list[float] = []
    for idx, row in frame.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["model_answer"])
        audit = retriever.audit(prompt, answer)
        profile = extractor._question_profile(prompt)
        hit_counts.append(int(audit.retrieval_hit_count))
        latencies.append(float(audit.retrieval_latency_sec))
        rows.append(
            {
                "row_index": int(idx),
                "row_key": row.get("row_key", ""),
                "prompt": prompt,
                "model_answer": answer,
                "profile": profile,
                "evidence_status": audit.status,
                "retrieval_hit_count": int(audit.retrieval_hit_count),
                "top_evidence_overlap": audit.top_evidence_overlap,
                "core_supported": audit.core_supported,
                "core_refuted": audit.core_refuted,
                "claim_refuted_count": audit.claim_refuted_count,
            }
        )

    audit_frame = pd.DataFrame(rows)
    total = int(len(audit_frame))
    with_evidence = int((audit_frame["retrieval_hit_count"] > 0).sum())
    by_profile: dict[str, Any] = {}
    for profile, chunk in audit_frame.groupby("profile"):
        profile_rows = int(len(chunk))
        profile_hits = int((chunk["retrieval_hit_count"] > 0).sum())
        by_profile[str(profile)] = {
            "rows": profile_rows,
            "rows_with_evidence": profile_hits,
            "coverage": float(profile_hits / max(1, profile_rows)),
        }

    typed_profiles = FeatureExtractor.TYPED_SHORT_PROFILES | {"which_list"}
    typed = audit_frame[audit_frame["profile"].isin(typed_profiles)]
    generic = audit_frame[~audit_frame["profile"].isin(typed_profiles)]
    report = {
        "csv_path": str(csv_path),
        "db_path": str(db_path),
        "rows": total,
        "rows_with_evidence": with_evidence,
        "coverage": float(with_evidence / max(1, total)),
        "typed_coverage": float((typed["retrieval_hit_count"] > 0).sum() / max(1, len(typed))),
        "generic_coverage": float((generic["retrieval_hit_count"] > 0).sum() / max(1, len(generic))),
        "by_profile": by_profile,
        "hit_count_distribution": _histogram(hit_counts),
        "core_supported": int((audit_frame["core_supported"] > 0).sum()),
        "core_refuted": int((audit_frame["core_refuted"] > 0).sum()),
        "claim_refuted_count": float(audit_frame["claim_refuted_count"].sum()),
        "p95_retrieval_latency_sec": float(pd.Series(latencies).quantile(0.95)) if latencies else 0.0,
        "top_zero_result_queries": audit_frame[audit_frame["retrieval_hit_count"] == 0]
        .groupby("profile")
        .head(20)[["row_index", "row_key", "profile", "prompt", "model_answer"]]
        .to_dict(orient="records"),
    }
    return report, audit_frame[audit_frame["retrieval_hit_count"] == 0].copy()


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence Coverage Report",
        "",
        f"- Rows: {report['rows']}",
        f"- Rows with evidence: {report['rows_with_evidence']}",
        f"- Coverage: {report['coverage']:.4f}",
        f"- Typed coverage: {report['typed_coverage']:.4f}",
        f"- Generic coverage: {report['generic_coverage']:.4f}",
        f"- p95 retrieval latency sec: {report['p95_retrieval_latency_sec']:.4f}",
        f"- Core supported: {report['core_supported']}",
        f"- Core refuted: {report['core_refuted']}",
        f"- Claim refuted count: {report['claim_refuted_count']}",
        "",
        "## Coverage by Profile",
    ]
    for profile, values in sorted(report["by_profile"].items()):
        lines.append(f"- {profile}: rows={values['rows']} hits={values['rows_with_evidence']} coverage={values['coverage']:.4f}")
    lines.extend(["", "## Hit Count Distribution"])
    for bucket, count in sorted(report["hit_count_distribution"].items()):
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "## Top Zero Result Queries"])
    for item in report["top_zero_result_queries"][:50]:
        lines.append(f"- row={item['row_index']} profile={item['profile']} prompt={item['prompt'][:160]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report offline evidence coverage before model training.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--db-path", default="model/evidence.db")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--md-output", required=True)
    parser.add_argument("--missing-output", required=True)
    args = parser.parse_args()

    report, missing = build_coverage_report(args.csv_path, args.db_path)
    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    missing_path = Path(args.missing_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    missing.to_csv(missing_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
