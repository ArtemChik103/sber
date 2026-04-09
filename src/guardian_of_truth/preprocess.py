from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from guardian_of_truth.api_client import AuditPayload, GroqVerifier
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.utils import iter_jsonl


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

    for idx, record in enumerate(tqdm(iter_jsonl(dataset_path), desc="preprocess")):
        if limit is not None and idx >= limit:
            break

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
    for idx, record in enumerate(iter_jsonl(dataset_path)):
        if limit is not None and idx >= limit:
            break
        features.append(extractor.extract_text_only(str(record["prompt"]), str(record["answer"])))
        labels.append(int(record["label"]))
    X = (
        np.stack(features).astype(np.float32)
        if features
        else np.empty((0, len(FeatureExtractor.text_feature_names)), dtype=np.float32)
    )
    y = np.array(labels, dtype=np.int32)
    return X, y
