from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from guardian_of_truth.api_client import GroqVerifier, AuditPayload
from guardian_of_truth.utils import load_local_env

sys.stdout.reconfigure(encoding='utf-8')
load_local_env()

bench_path = Path("data/bench/knowledge_bench_public.csv")
checkpoint_path = Path("outputs/public_scored_audit_rich_v6.csv")
final_output_path = Path("outputs/public_scored_audit_rich_v6_complete.csv")

df_full = pd.read_csv(bench_path)
total_count = len(df_full)

verifier = GroqVerifier(allow_runtime_wait=True)

# Загружаем уже имеющийся чекпоинт
existing_rows = {}
if checkpoint_path.exists():
    df_chk = pd.read_csv(checkpoint_path)
    for _, row in df_chk.iterrows():
        # Сохраняем если статус ok
        if row.get("audit_status") == "ok":
            key = (str(row.get("prompt", "")).strip(), str(row.get("model_answer", "")).strip())
            existing_rows[key] = row.to_dict()

print(f"Loaded {len(existing_rows)} successful rows from checkpoint.")

scored_records = []
pbar = tqdm(total=total_count, desc="Fast Qwen Prefill")

# Целевой интервал между запросами: ровно 2.05 секунды (29.2 RPM, строго ниже лимита 30 RPM)
TARGET_INTERVAL_SEC = 2.05
last_request_time = 0.0

for idx, row in df_full.iterrows():
    p = str(row.get("prompt", "")).strip()
    a = str(row.get("model_answer", "")).strip()
    key = (p, a)
    
    if key in existing_rows:
        rec = existing_rows[key]
        rec["is_hallucination"] = row["is_hallucination"]
        scored_records.append(rec)
        pbar.update(1)
        continue
    
    # Регулируем темп строго: не чаще 1 запроса в TARGET_INTERVAL_SEC
    elapsed_since_last = time.time() - last_request_time
    if elapsed_since_last < TARGET_INTERVAL_SEC:
        time.sleep(TARGET_INTERVAL_SEC - elapsed_since_last)
    
    last_request_time = time.time()
    
    # Вызов верификатора
    audit = verifier.verify(p, a, mode="runtime")
    
    # Если 429, делаем однократную паузу и повтор
    if audit.status == "http_429":
        time.sleep(30.0)
        last_request_time = time.time()
        audit = verifier.verify(p, a, mode="runtime")
        
    rec = row.to_dict()
    rec["audit_h"] = audit.h
    rec["audit_n"] = audit.n
    rec["audit_e"] = audit.e
    rec["audit_r"] = audit.r
    rec["audit_u"] = audit.u
    rec["audit_c"] = audit.c
    rec["audit_x"] = audit.x
    rec["audit_q"] = audit.q
    rec["audit_s"] = audit.s
    rec["audit_m"] = audit.m
    rec["audit_sem"] = audit.sem
    rec["audit_we"] = audit.we
    rec["audit_wn"] = audit.wn
    rec["audit_conf"] = audit.conf
    rec["audit_status"] = audit.status
    rec["audit_ok"] = audit.ok
    rec["audit_cached"] = audit.cached
    rec["audit_model_name"] = audit.model_name
    rec["is_hallucination_proba"] = audit.h
    
    scored_records.append(rec)
    existing_rows[key] = rec
    pbar.update(1)
    
    # Сохраняем промежуточный результат каждые 20 строк
    if len(scored_records) % 20 == 0:
        pd.DataFrame(scored_records).to_csv(final_output_path, index=False)

pbar.close()

# Сохраняем итоговый датафрейм
df_result = pd.DataFrame(scored_records)
df_result.to_csv(final_output_path, index=False)
df_result.to_csv(checkpoint_path, index=False)

print("\n================== PREFILL COMPLETED ==================")
print(f"Total rows: {len(df_result)}")
print(df_result["audit_status"].value_counts())

ok_mask = df_result["audit_status"] == "ok"
if ok_mask.sum() > 0:
    y_true = df_result.loc[ok_mask, "is_hallucination"].astype(int)
    y_pred = df_result.loc[ok_mask, "is_hallucination_proba"]
    print(f"\nOK rows: {ok_mask.sum()}/{len(df_result)}")
    print(f"FINAL PR-AUC: {average_precision_score(y_true, y_pred):.4f}")
    print(f"FINAL ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")
