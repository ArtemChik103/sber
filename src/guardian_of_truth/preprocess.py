from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from guardian_of_truth.api_client import AuditPayload, GroqVerifier
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.utils import read_jsonl, sha256_hexdigest


def _selected_records(dataset_path: str | Path, limit: int | None = None) -> list[dict[str, object]]:
    records = read_jsonl(dataset_path)
    if limit is None or len(records) <= limit:
        return records

    frame = pd.DataFrame(
        [
            {
                "record_idx": idx,
                "label": int(record["label"]),
                "variant_type": str(record.get("variant_type", "unknown") or "unknown"),
                "sample_key": sha256_hexdigest(
                    record.get("variant_type"),
                    record.get("label"),
                    record.get("prompt"),
                    record.get("answer"),
                ),
            }
            for idx, record in enumerate(records)
        ]
    )
    frame["group_key"] = frame["variant_type"] + "__" + frame["label"].astype(str)
    sampled_parts: list[pd.DataFrame] = []

    for _, chunk in frame.groupby("group_key", sort=False):
        part_size = max(1, round(limit * len(chunk) / len(frame)))
        sampled_parts.append(chunk.sort_values("sample_key").head(min(part_size, len(chunk))))

    selected = pd.concat(sampled_parts, ignore_index=True).sort_values("sample_key").head(limit)
    return [records[int(row.record_idx)] for row in selected.itertuples(index=False)]


def build_feature_matrix(
    dataset_path: str | Path,
    verifier: GroqVerifier | None,
    extractor: FeatureExtractor,
    *,
    use_api: bool = True,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, object]] = []

    records = _selected_records(dataset_path, limit=limit)
    for record in tqdm(records, desc="preprocess"):
        prompt = str(record["prompt"])
        answer = str(record["answer"])
        label = int(record["label"])
        audit = (
            verifier.verify(prompt, answer, mode="dataset")
            if use_api and verifier is not None
            else AuditPayload.neutral(status="disabled", mode="dataset", model_name=None, ok=False)
        )
        feature_vector = extractor.extract(prompt, answer, audit)
        features.append(feature_vector)
        labels.append(label)
        rows.append(
            {
                "prompt": prompt,
                "answer": answer,
                "label": label,
                "variant_type": record.get("variant_type"),
                "source": record.get("source"),
                "audit_status": audit.status,
            }
        )

    X = np.stack(features).astype(np.float32) if features else np.empty((0, 14), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    meta = pd.DataFrame(rows)
    return X, y, meta


def build_text_only_matrix(dataset_path: str | Path, extractor: FeatureExtractor, *, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for record in _selected_records(dataset_path, limit=limit):
        features.append(extractor.extract_text_only(str(record["prompt"]), str(record["answer"])))
        labels.append(int(record["label"]))
    X = (
        np.stack(features).astype(np.float32)
        if features
        else np.empty((0, len(FeatureExtractor.text_feature_names)), dtype=np.float32)
    )
    y = np.array(labels, dtype=np.int32)
    return X, y
