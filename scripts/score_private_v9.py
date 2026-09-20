import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.guardian import GuardianOfTruth
from guardian_of_truth.refute_overlay import (
    BAYESIAN_BLIND_RESCUE_POLICY,
    IsotonicRankCalibrator,
)
from guardian_of_truth.utils import MODEL_DIR

sys.stdout.reconfigure(encoding='utf-8')

guardian = GuardianOfTruth(
    refute_overlay_policy=BAYESIAN_BLIND_RESCUE_POLICY,
    model_dir="model",
)

input_path = "knowledge_bench_private_no_labels.csv"
output_path = "knowledge_bench_private_scores.csv"

df = pd.read_csv(input_path)
print(f"Loaded {len(df)} private samples.")

scores = []
latencies = []
reasons = []

t0 = time.perf_counter()
for idx, r in df.iterrows():
    p = r["prompt"]
    a = r["model_answer"]
    res = guardian.score(p, a)
    scores.append(res.is_hallucination_proba)
    latencies.append(res.t_total_sec)
    reasons.append(guardian.last_refute_overlay_reason)

elapsed = time.perf_counter() - t0
mean_latency = np.mean(latencies) * 1000

# Apply Isotonic Rank Calibration with Profile Priors
calibrator = IsotonicRankCalibrator(MODEL_DIR / "isotonic_calibrator.joblib")
prompts_list = df["prompt"].astype(str).tolist()
calibrated_scores = np.asarray(calibrator.predict(scores, prompts=prompts_list), dtype=float)


print(f"Scored {len(scores)} samples in {elapsed:.2f}s (mean: {mean_latency:.1f} ms)")
print(f"Raw Range: [{np.min(scores):.3f}, {np.max(scores):.3f}], Mean: {np.mean(scores):.3f}")
print(f"Calibrated Range: [{np.min(calibrated_scores):.3f}, {np.max(calibrated_scores):.3f}], Mean: {np.mean(calibrated_scores):.3f}")

# Update df
df["raw_predict_proba"] = scores
df["predict_proba"] = calibrated_scores
df["is_hallucination_proba"] = calibrated_scores
df["pred_is_hallucination"] = (calibrated_scores >= 0.5).astype(bool)
df["refute_overlay_policy"] = BAYESIAN_BLIND_RESCUE_POLICY
df["refute_overlay_reason"] = reasons

# Validation checks
assert len(df) == 1038, f"Expected 1038 rows, got {len(df)}"
assert df["predict_proba"].notna().all(), "Found NaN in predict_proba!"
assert (df["predict_proba"] >= 0.0).all() and (df["predict_proba"] <= 1.0).all(), "Scores out of bounds!"

df.to_csv(output_path, index=False)
print(f"Private submission successfully written to {output_path}!")

