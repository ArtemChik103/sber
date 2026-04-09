from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from guardian_of_truth.api_client import GroqVerifier
from guardian_of_truth.classifier import HallucinationClassifier, save_fallback_bundle, save_training_summary
from guardian_of_truth.feature_extractor import FeatureExtractor
from guardian_of_truth.preprocess import build_feature_matrix, build_text_only_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Guardian of Truth detector on synthetic data.")
    parser.add_argument("--dataset-path", default="data/raw/synthetic_factual_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-api", action="store_true")
    parser.add_argument("--model-dir", default="model")
    args = parser.parse_args()

    verifier = None if args.disable_api else GroqVerifier()
    extractor = FeatureExtractor()
    X, y, _ = build_feature_matrix(args.dataset_path, verifier, extractor, use_api=not args.disable_api, limit=args.limit)
    X_text, y_text = build_text_only_matrix(args.dataset_path, extractor, limit=args.limit)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    X_text_train, X_text_val, y_text_train, y_text_val = train_test_split(
        X_text,
        y_text,
        test_size=0.2,
        stratify=y_text,
        random_state=42,
    )

    variants = {
        "V1": ([0], "none"),
        "V2": (list(range(7)), "none"),
        "V3": (list(range(14)), "none"),
        "V4": (list(range(14)), "isotonic"),
    }

    best_name = ""
    best_ap = -1.0
    best_classifier: HallucinationClassifier | None = None
    metrics: dict[str, float] = {}

    feature_names = FeatureExtractor.api_feature_names + FeatureExtractor.text_feature_names

    for name, (indices, calibration) in variants.items():
        classifier = HallucinationClassifier(
            feature_names=[feature_names[idx] for idx in indices],
            feature_indices=indices,
        )
        classifier.fit(
            X_train[:, indices],
            y_train,
            X_val[:, indices],
            y_val,
            calibration=calibration,
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

    selected_indices = variants[best_name][0]
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
        calibration="isotonic",
    )
    save_fallback_bundle(fallback, args.model_dir)

    summary = {
        "selected_variant": best_name,
        "selected_indices": selected_indices,
        "average_precision": metrics,
        "disable_api": args.disable_api,
        "dataset_path": args.dataset_path,
    }
    save_training_summary(args.model_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
