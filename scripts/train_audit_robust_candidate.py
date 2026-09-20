from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.api_client import AuditPayload, GroqVerifier
from guardian_of_truth.classifier import HallucinationClassifier, save_fallback_bundle, save_training_summary
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.utils import read_jsonl, sha256_hexdigest
from train import _feature_set_indices


ROBUST_TARGETS = {
    "candidate_audit_robust_hybrid_v1": "hybrid_targeted",
    "candidate_audit_robust_old_targeted_v1": "old_full_targeted",
    "candidate_audit_robust_text_targeted_v1": "text_targeted",
}


def split_original_rows(records: list[dict[str, Any]], *, val_ratio: float = 0.2) -> list[str]:
    splits: list[str] = []
    for record in records:
        value = int(sha256_hexdigest("audit-robust-split", record.get("prompt"), record.get("answer"), record.get("label"))[:8], 16) / 0xFFFFFFFF
        splits.append("val" if value < val_ratio else "train")
    if records and len(set(splits)) < 2:
        splits[-1] = "val"
    return splits


def _fixture(kind: str, base: AuditPayload | None = None) -> AuditPayload:
    if kind == "original" and base is not None:
        return base
    neutral = {
        "h": 0.0, "n": 0.0, "e": 0.0, "r": 0.5, "u": 0.0, "c": 1.0, "x": 0.0, "q": 0.5,
        "s": 0.5, "m": 0.0, "sem": 0.5, "we": 0.0, "wn": 0.0, "ue": 0.0, "bt": 0.0,
        "conf": 0.5, "ok": True, "status": "ok", "model_name": "audit_robust_aug", "mode": "dataset",
    }
    if kind == "correct_high_noisy_x0":
        neutral.update({"h": 0.34, "u": 0.80, "we": 0.88, "wn": 0.80, "ue": 0.75, "bt": 0.82, "x": 0.0, "conf": 0.88})
    elif kind == "wrong_neutral":
        neutral.update({"h": 0.04, "u": 0.03, "we": 0.02, "wn": 0.02, "ue": 0.02, "bt": 0.02, "conf": 0.55})
    elif kind == "wrong_strong_contradiction":
        neutral.update({"h": 0.92, "u": 0.86, "we": 0.90, "wn": 0.86, "ue": 0.72, "bt": 0.68, "x": 1.0, "conf": 0.92})
    return AuditPayload.model_validate(neutral)


def augmentation_plan(label: int) -> list[tuple[str, float]]:
    plan = [("original", 1.0), ("neutral_dropout", 0.30)]
    if int(label) == 0:
        plan.append(("correct_high_noisy_x0", 0.75))
    else:
        plan.extend([("wrong_neutral", 0.75), ("wrong_strong_contradiction", 0.40)])
    return plan


def load_original_audits(records: list[dict[str, Any]], *, cache_only_api: bool = True, disable_api: bool = False) -> list[AuditPayload]:
    verifier = None if disable_api else GroqVerifier()
    audits: list[AuditPayload] = []
    for record in records:
        if verifier is None:
            audits.append(AuditPayload.neutral(status="disabled", mode="dataset", model_name=None, ok=False))
            continue
        prompt = str(record["prompt"])
        answer = str(record.get("answer", record.get("model_answer")))
        audit = verifier.cached_audit(prompt, answer, mode="dataset") if cache_only_api else verifier.verify(prompt, answer, mode="dataset")
        if audit is None:
            audit = AuditPayload.neutral(status="cache_miss", mode="dataset", model_name=verifier.settings.experiment_model, ok=False)
        audits.append(audit)
    return audits


def build_augmented_rows(records: list[dict[str, Any]], audits: list[AuditPayload]) -> pd.DataFrame:
    splits = split_original_rows(records)
    rows: list[dict[str, Any]] = []
    for idx, (record, audit, split) in enumerate(zip(records, audits, splits, strict=True)):
        prompt = str(record["prompt"])
        answer = str(record.get("answer", record.get("model_answer")))
        label = int(record.get("label", record.get("is_hallucination")))
        original_key = sha256_hexdigest(prompt, answer, label, idx)
        for aug_kind, weight in augmentation_plan(label):
            aug_audit = _fixture("original", audit) if aug_kind == "original" else _fixture(aug_kind)
            rows.append({
                "prompt": prompt,
                "answer": answer,
                "label": label,
                "split": split,
                "original_key": original_key,
                "augmentation": aug_kind,
                "sample_weight": float(weight),
                "audit": aug_audit,
                "variant_type": record.get("variant_type", "unknown"),
            })
    return pd.DataFrame(rows)


def _matrices(frame: pd.DataFrame, extractor: FeatureExtractor) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X = np.stack([extractor.extract(row.prompt, row.answer, row.audit) for row in frame.itertuples(index=False)]).astype(np.float32)
    y = frame["label"].astype(int).to_numpy(dtype=np.int32)
    weights = frame["sample_weight"].astype(float).to_numpy(dtype=np.float64)
    return X, y, weights, frame


def train_one(frame: pd.DataFrame, output_dir: str | Path, feature_set: str) -> dict[str, Any]:
    extractor = FeatureExtractor()
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    X_train, y_train, w_train, _ = _matrices(train_frame, extractor)
    X_val, y_val, w_val, _ = _matrices(val_frame, extractor)
    feature_names = FeatureExtractor.api_feature_names + FeatureExtractor.text_feature_names
    indices = _feature_set_indices()[feature_set]
    classifier = HallucinationClassifier(
        feature_names=[feature_names[idx] for idx in indices],
        feature_indices=indices,
        model=LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=42),
    )
    classifier.fit(
        X_train[:, indices],
        y_train,
        X_val[:, indices],
        y_val,
        calibration="sigmoid",
        score_transform="sigmoid",
        sample_weight_train=w_train,
        sample_weight_val=w_val,
    )
    output = Path(output_dir)
    classifier.save(output)
    X_text_train = np.stack([extractor.extract_text_only(row.prompt, row.answer) for row in train_frame.itertuples(index=False)]).astype(np.float32)
    X_text_val = np.stack([extractor.extract_text_only(row.prompt, row.answer) for row in val_frame.itertuples(index=False)]).astype(np.float32)
    fallback = HallucinationClassifier(
        feature_names=FeatureExtractor.text_feature_names,
        feature_indices=list(range(len(FeatureExtractor.text_feature_names))),
        model=LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=42),
    )
    fallback.fit(X_text_train, y_train, X_text_val, y_val, calibration="sigmoid", score_transform="sigmoid", sample_weight_train=w_train, sample_weight_val=w_val)
    save_fallback_bundle(fallback, output)
    val_scores = classifier.predict_proba(X_val[:, indices])
    summary = {
        "model_family": "audit_robust",
        "feature_set": feature_set,
        "val_ap": float(average_precision_score(y_val, val_scores)) if len(set(y_val)) > 1 else 0.0,
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "augmentation_counts": frame["augmentation"].value_counts().astype(int).to_dict(),
        "augmentation_weights": {name: weight for label in (0, 1) for name, weight in augmentation_plan(label)},
    }
    save_training_summary(output, summary)
    return summary


def train_audit_robust_candidates(
    dataset_path: str | Path = "data/raw/synthetic_factual_data.jsonl",
    *,
    output_root: str | Path = "outputs",
    limit: int | None = None,
    cache_only_api: bool = True,
    disable_api: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]
    audits = load_original_audits(records, cache_only_api=cache_only_api, disable_api=disable_api)
    augmented = build_augmented_rows(records, audits)
    summaries: dict[str, Any] = {}
    for dirname, feature_set in ROBUST_TARGETS.items():
        summaries[dirname] = train_one(augmented, Path(output_root) / dirname, feature_set)
    overall = {
        "dataset_path": str(dataset_path),
        "original_rows": int(len(records)),
        "augmented_rows": int(len(augmented)),
        "targets": summaries,
    }
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "audit_robust_training.summary.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Train audit-robust candidate classifiers with deterministic audit augmentations.")
    parser.add_argument("--dataset-path", default="data/raw/synthetic_factual_data.jsonl")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only-api", action="store_true", default=True)
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--disable-api", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_audit_robust_candidates(
        args.dataset_path,
        output_root=args.output_root,
        limit=args.limit,
        cache_only_api=not args.live_api,
        disable_api=args.disable_api,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
