from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from guardian_of_truth.utils import CONFIG_DIR, load_yaml


def build_variant_stratify_labels(y: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    labels = pd.Series(y, dtype="int32").astype(str)
    variants = meta.get("variant_type", pd.Series(["unknown"] * len(meta))).fillna("unknown").astype(str)
    combined = variants + "__" + labels
    counts = combined.value_counts()
    stabilized = combined.where(combined.map(counts) >= 2, labels)
    return stabilized.to_numpy()


def compute_variant_weights(meta: pd.DataFrame, config_path: str | Path = CONFIG_DIR / "model.yaml") -> np.ndarray:
    training_cfg = load_yaml(config_path).get("training", {})
    variant_weights = training_cfg.get("variant_weights", {})
    if meta.empty:
        return np.empty((0,), dtype=np.float64)

    variants = meta.get("variant_type", pd.Series(["unknown"] * len(meta))).fillna("unknown").astype(str)
    weights = variants.map(lambda value: float(variant_weights.get(value, 1.0))).to_numpy(dtype=np.float64)
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    if mean_weight <= 0.0:
        return np.ones_like(weights, dtype=np.float64)
    return weights / mean_weight


def summarize_variant_counts(meta: pd.DataFrame) -> dict[str, int]:
    if meta.empty or "variant_type" not in meta.columns:
        return {}
    counts = meta["variant_type"].fillna("unknown").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}
