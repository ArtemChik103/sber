from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from guardian_of_truth.api_client import GroqVerifier
from guardian_of_truth.classifier import HallucinationClassifier, save_fallback_bundle, save_training_summary
from guardian_of_truth.evidence import EvidenceRetriever
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.preprocess import build_feature_matrix, build_text_only_matrix
from guardian_of_truth.training import build_variant_stratify_labels, compute_variant_weights, summarize_variant_counts
from guardian_of_truth.utils import CONFIG_DIR, load_yaml


OLD_API_FEATURE_COUNT = 10
BASE_TEXT_FEATURE_COUNT = 7


def _audit_summary(meta) -> dict[str, object]:
    if meta.empty or "audit_status" not in meta.columns:
        return {"audit_status_counts": {}, "audit_ok_ratio": 0.0}
    counts = meta["audit_status"].fillna("unknown").astype(str).value_counts()
    ok_count = int(counts.get("ok", 0) + counts.get("partial_json", 0))
    total = int(len(meta))
    return {
        "audit_status_counts": {str(key): int(value) for key, value in counts.items()},
        "audit_ok_ratio": float(ok_count / max(1, total)),
    }


def _make_model_candidates(random_state: int) -> dict[str, object]:
    return {
        "logreg": LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=random_state),
        "histgb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            l2_regularization=0.05,
            random_state=random_state,
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _feature_set_indices() -> dict[str, list[int]]:
    api_feature_count = len(FeatureExtractor.api_feature_names)
    text_feature_count = len(FeatureExtractor.text_feature_names)
    evidence_feature_count = len(FeatureExtractor.evidence_feature_names)
    text_start = api_feature_count
    evidence_start = api_feature_count + text_feature_count
    legacy_full_feature_count = api_feature_count + text_feature_count
    full_feature_count = legacy_full_feature_count + evidence_feature_count
    base_text_end = min(text_start + BASE_TEXT_FEATURE_COUNT, legacy_full_feature_count)
    base_text = list(range(text_start, base_text_end))
    old_text_4 = list(range(text_start, min(text_start + 4, base_text_end)))
    short_answer = list(range(base_text_end, legacy_full_feature_count))
    targeted_text = short_answer
    evidence = list(range(evidence_start, full_feature_count))

    return {
        "api_h_only": [0],
        "api": list(range(api_feature_count)),
        "full": list(range(api_feature_count)) + base_text,
        "old_full_14": list(range(OLD_API_FEATURE_COUNT)) + old_text_4,
        "old_full": list(range(OLD_API_FEATURE_COUNT)) + base_text,
        "new_api_only": list(range(OLD_API_FEATURE_COUNT, api_feature_count)),
        "text_only": base_text,
        "hybrid_old_new": list(range(api_feature_count)) + base_text,
        "old_full_plus_short_answer": list(range(OLD_API_FEATURE_COUNT)) + base_text + short_answer,
        "text_plus_short_answer": base_text + short_answer,
        "hybrid_plus_short_answer": list(range(legacy_full_feature_count)),
        "old_full_targeted": list(range(OLD_API_FEATURE_COUNT)) + base_text + targeted_text,
        "text_targeted": base_text + targeted_text,
        "hybrid_targeted": list(range(legacy_full_feature_count)),
        "audit_targeted": list(range(api_feature_count)),
        "evidence_only": evidence,
        "text_plus_evidence": list(range(text_start, text_start + text_feature_count)) + evidence,
        "hybrid_evidence_full": list(range(full_feature_count)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Guardian of Truth detector on synthetic data.")
    parser.add_argument("--dataset-path", default="data/raw/synthetic_factual_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-api", action="store_true")
    parser.add_argument("--cache-only-api", action="store_true")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--evidence-db-path", default=None)
    parser.add_argument(
        "--min-audit-ok-ratio",
        type=float,
        default=None,
        help="Override configs/model.yaml training.min_audit_ok_ratio for controlled cache-only experiments.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=sorted(_feature_set_indices()),
        help="Restrict candidate search to one named feature set. Can be passed more than once.",
    )
    args = parser.parse_args()

    config = load_yaml(CONFIG_DIR / "model.yaml")
    training_cfg = config.get("training", {})
    detector_cfg = config.get("detector", {})
    min_audit_ok_ratio = (
        float(args.min_audit_ok_ratio)
        if args.min_audit_ok_ratio is not None
        else float(training_cfg.get("min_audit_ok_ratio", 0.8))
    )
    random_state = int(detector_cfg.get("random_state", 42))

    verifier = None if args.disable_api else GroqVerifier()
    extractor = FeatureExtractor()
    evidence_retriever = EvidenceRetriever(args.evidence_db_path) if args.evidence_db_path else None
    X, y, meta = build_feature_matrix(
        args.dataset_path,
        verifier,
        extractor,
        use_api=not args.disable_api,
        cache_only_api=args.cache_only_api,
        limit=args.limit,
        evidence_retriever=evidence_retriever,
    )
    X_text, y_text = build_text_only_matrix(args.dataset_path, extractor, limit=args.limit)
    audit_summary = _audit_summary(meta)

    if not args.disable_api and audit_summary["audit_ok_ratio"] < min_audit_ok_ratio:
        raise RuntimeError(
            "Groq audit quality is too low for training: "
            f"ok_ratio={audit_summary['audit_ok_ratio']:.3f}, "
            f"required={min_audit_ok_ratio:.3f}, "
            f"status_counts={audit_summary['audit_status_counts']}"
        )

    if not np.array_equal(y, y_text):
        raise RuntimeError("Label order mismatch between full and text-only matrices.")

    indices = np.arange(len(y))
    stratify_labels = build_variant_stratify_labels(y, meta)
    if pd.Series(stratify_labels).value_counts().min() < 2:
        stratify_labels = y
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=stratify_labels,
        random_state=42,
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
    y_text_train, y_text_val = y_text[train_idx], y_text[val_idx]
    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_val = meta.iloc[val_idx].reset_index(drop=True)
    train_weights = compute_variant_weights(meta_train)
    val_weights = compute_variant_weights(meta_val)
    api_feature_count = len(FeatureExtractor.api_feature_names)
    full_feature_count = api_feature_count + len(FeatureExtractor.text_feature_names) + (
        len(FeatureExtractor.evidence_feature_names) if evidence_retriever is not None else 0
    )
    all_feature_sets = _feature_set_indices()
    selected_feature_set_names = args.feature_set or [
        "api_h_only",
        "api",
        "full",
        "old_full_14",
        "old_full",
        "new_api_only",
        "text_only",
        "hybrid_old_new",
        "hybrid_evidence_full",
    ]
    selected_feature_set_names = [name for name in selected_feature_set_names if name in all_feature_sets]
    feature_sets = {
        name: indices
        for name, indices in ((name, all_feature_sets[name]) for name in selected_feature_set_names)
        if indices and max(indices) < X.shape[1]
    }
    score_variants = [
        ("predict_proba", "none"),
        ("raw_margin_sigmoid", "none"),
        ("sigmoid", "sigmoid"),
        ("isotonic", "isotonic"),
    ]
    model_candidates = _make_model_candidates(random_state)

    best_name = ""
    best_ap = -1.0
    best_classifier: HallucinationClassifier | None = None
    metrics: dict[str, float] = {}

    feature_names = FeatureExtractor.api_feature_names + FeatureExtractor.text_feature_names + FeatureExtractor.evidence_feature_names

    for model_name, model in model_candidates.items():
        for feature_set_name, indices in feature_sets.items():
            for score_transform, calibration in score_variants:
                if model_name != "logreg" and score_transform == "raw_margin_sigmoid":
                    continue
                name = f"{model_name}_{feature_set_name}_{score_transform}"
                classifier = HallucinationClassifier(
                    feature_names=[feature_names[idx] for idx in indices],
                    feature_indices=indices,
                    model=model.__class__(**model.get_params()),
                )
                classifier.fit(
                    X_train[:, indices],
                    y_train,
                    X_val[:, indices],
                    y_val,
                    calibration=calibration,
                    score_transform=score_transform,
                    sample_weight_train=train_weights,
                    sample_weight_val=val_weights,
                )
                proba = classifier.predict_proba(X_val[:, indices])
                ap = float(average_precision_score(y_val, proba))
                metrics[name] = ap
                if ap > best_ap:
                    best_name = name
                    best_ap = ap
                    best_classifier = classifier

    if best_classifier is None:
        raise RuntimeError("No classifier variant was trained.")

    selected_indices = best_classifier.feature_indices
    best_classifier.save(args.model_dir)

    fallback = HallucinationClassifier(
        feature_names=FeatureExtractor.text_feature_names,
        feature_indices=list(range(len(FeatureExtractor.text_feature_names))),
    )
    fallback.fit(
        X_text_train,
        y_text_train,
        X_text_val,
        y_text_val,
        calibration="sigmoid",
        score_transform="sigmoid",
        sample_weight_train=train_weights,
        sample_weight_val=val_weights,
    )
    save_fallback_bundle(fallback, args.model_dir)

    summary = {
        "selected_variant": best_name,
        "selected_indices": selected_indices,
        "average_precision": metrics,
        "disable_api": args.disable_api,
        "dataset_path": args.dataset_path,
        "variant_counts": summarize_variant_counts(meta),
        "train_variant_counts": summarize_variant_counts(meta_train),
        "val_variant_counts": summarize_variant_counts(meta_val),
        "variant_weighting": "enabled",
        **audit_summary,
        "api_model": verifier.settings.runtime_model if verifier is not None else None,
        "dataset_api_model": verifier.settings.experiment_model if verifier is not None else None,
        "prompt_version": verifier.settings.prompt_version if verifier is not None else None,
        "dataset_prompt_version": verifier.settings.dataset_prompt_version if verifier is not None else None,
        "feature_count": int(full_feature_count),
        "evidence_db_path": args.evidence_db_path,
        "uses_offline_evidence": bool(evidence_retriever is not None and evidence_retriever.available),
        "feature_sets_evaluated": selected_feature_set_names,
        "selected_score_transform": best_classifier.score_transform,
        "selected_calibration": best_classifier.calibration_kind,
    }
    save_training_summary(args.model_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
