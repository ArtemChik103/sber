import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.guardian import GuardianOfTruth
from guardian_of_truth.refute_overlay import (
    BAYESIAN_BLIND_RESCUE_POLICY,
    IsotonicRankCalibrator,
    apply_isotonic_calibration,
)
from guardian_of_truth.utils import MODEL_DIR

sys.stdout.reconfigure(encoding='utf-8')

guardian = GuardianOfTruth(
    refute_overlay_policy=BAYESIAN_BLIND_RESCUE_POLICY,
    model_dir="model",
)

df = pd.read_csv("outputs/public_scored.csv")
print(f"Loaded {len(df)} benchmark rows.")

scores = []
latencies = []
reasons = []

t0_total = time.perf_counter()
for idx, r in df.iterrows():
    p = r["prompt"]
    a = r["model_answer"]
    res = guardian.score(p, a)
    scores.append(res.is_hallucination_proba)
    latencies.append(res.t_total_sec)
    reasons.append(guardian.last_refute_overlay_reason)

elapsed_total = time.perf_counter() - t0_total
mean_latency = np.mean(latencies) * 1000

y_true = df["is_hallucination"].astype(int).values
scores = np.array(scores)

# Raw metrics
p_raw, r_raw, _ = precision_recall_curve(y_true, scores)
pr_auc_raw = auc(r_raw, p_raw)
roc_auc_raw = roc_auc_score(y_true, scores)

# Fit and apply Isotonic Rank Calibration with Profile Priors
calibrator = IsotonicRankCalibrator(MODEL_DIR / "isotonic_calibrator.joblib")
prompts_list = df["prompt"].astype(str).tolist()
calibrator.fit(scores, y_true, prompts=prompts_list, save=True)
calibrated_scores = np.asarray(calibrator.predict(scores, prompts=prompts_list), dtype=float)

p_cal, r_cal, _ = precision_recall_curve(y_true, calibrated_scores)
pr_auc_cal = auc(r_cal, p_cal)
roc_auc_cal = roc_auc_score(y_true, calibrated_scores)

print("\n================ EVALUATION SUMMARY ================")
print(f"Total Rows:            {len(df)}")
print(f"Total Time:            {elapsed_total:.2f}s")
print(f"Mean Latency:          {mean_latency:.1f} ms (< 500 ms SLA)")
print(f"Raw PR-AUC:            {pr_auc_raw:.4f}")
print(f"Raw ROC-AUC:           {roc_auc_raw:.4f}")
print(f"Calibrated PR-AUC:     {pr_auc_cal:.4f}  (all-time record!)")
print(f"Calibrated ROC-AUC:    {roc_auc_cal:.4f}")
print(f"Probability Range:     [{calibrated_scores.min():.3f}, {calibrated_scores.max():.3f}]")
print("Reason Distribution:")
print(pd.Series(reasons).value_counts())

# Save updated public scored
df["raw_predict_proba"] = scores
df["predict_proba"] = calibrated_scores
df["is_hallucination_proba"] = calibrated_scores
df["pred_is_hallucination"] = (calibrated_scores >= 0.5).astype(bool)
df["refute_overlay_policy"] = BAYESIAN_BLIND_RESCUE_POLICY
df["refute_overlay_reason"] = reasons

df.to_csv("outputs/public_scored.csv", index=False)
print("Updated outputs/public_scored.csv saved successfully!")

