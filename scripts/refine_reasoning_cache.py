from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from groq import AsyncGroq, RateLimitError, APIError

from guardian_of_truth.api_client import MultiKeyRateLimiter
from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import DATA_DIR, sha256_hexdigest, load_local_env

sys.stdout.reconfigure(encoding="utf-8")
load_local_env()

KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
VALID_KEYS = [k.strip() for k in KEYS if k and k.strip()]


SYSTEM_V7 = (
    "You are a master factual verification evaluator with expert encyclopedic knowledge. "
    "Examine whether the candidate answer contains any factual hallucination, incorrect entity, wrong date, wrong number, or false attribution.\n"
    "1. First, reason step-by-step through the question requirements and verify all claims in the candidate answer against historical facts.\n"
    "2. At the end, output a strict JSON block:\n"
    "```json\n"
    "{\n"
    '  "p": <float 0.0 for completely factual, 0.2 for minor rephrasing, 0.9 for hallucination/false fact>,\n'
    '  "h": <bool: true if hallucination, false if factually true>,\n'
    '  "error_type": <"none" | "number" | "date" | "name" | "entity" | "invented_fact" | "contradiction">,\n'
    '  "conf": <float 0.0 to 1.0>,\n'
    '  "reason": <short explanation of fact check>\n'
    "}\n"
    "```"
)


async def verify_item(
    client_pool: dict[str, AsyncGroq],
    limiter: MultiKeyRateLimiter,
    prompt: str,
    answer: str,
    cache: SQLiteCache,
    cache_key: str,
) -> bool:
    cached = cache.get(cache_key)
    if cached is not None:
        return True

    user_content = f"Question: {prompt}\nCandidate Answer: {answer}"
    estimated_tokens = max(64, (len(prompt) + len(answer)) // 4 + 600 + 32)

    for attempt in range(4):
        key, delay = limiter.reserve_key_and_delay(estimated_tokens)
        if delay > 0:
            await asyncio.sleep(min(delay, 5.0))
            continue

        client = client_pool[key]
        try:
            resp = await client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": SYSTEM_V7},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content or ""
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    payload = json.loads(raw[start : end + 1])
                    payload["raw_response"] = raw
                    payload["status"] = "ok"
                    payload["model_name"] = "qwen/qwen3.8-27b"
                    payload["prompt_version"] = "groq-verifier-v7-reasoned"
                    cache.set(cache_key, payload)
                    return True
                except json.JSONDecodeError:
                    pass
            # Fallback if json not cleanly parsed
            cache.set(cache_key, {"p": 0.5, "status": "partial_json", "raw_response": raw})
            return True

        except RateLimitError:
            limiter.report_429(key, 30.0)
            await asyncio.sleep(3.0)
        except Exception as e:
            if "429" in str(e):
                limiter.report_429(key, 30.0)
                await asyncio.sleep(3.0)
            else:
                await asyncio.sleep(1.0)

    return False


async def run_refinement(csv_path: str, min_prob: float = 0.32, max_prob: float = 0.68) -> None:
    df = pd.read_csv(csv_path)
    cache = SQLiteCache(DATA_DIR / "cache" / "groq_cache.sqlite")
    limiter = MultiKeyRateLimiter(VALID_KEYS, rpm=30, tpm=6000)
    client_pool = {k: AsyncGroq(api_key=k) for k in VALID_KEYS}

    # Отбираем пограничные строки
    target_indices = df[(df["predict_proba"] >= min_prob) & (df["predict_proba"] <= max_prob)].index.tolist()
    print(f"Total rows in {csv_path}: {len(df)}")
    print(f"Targeting {len(target_indices)} borderline rows in range [{min_prob:.2f}, {max_prob:.2f}]")

    tasks = []
    keys_to_process = []
    for idx in target_indices:
        r = df.loc[idx]
        p, a = str(r["prompt"]), str(r["model_answer"])
        ck = sha256_hexdigest("qwen/qwen3.8-27b", "groq-verifier-v7-reasoned", p, a, "runtime")
        if cache.get(ck) is None:
            keys_to_process.append((idx, p, a, ck))

    print(f"Rows already cached: {len(target_indices) - len(keys_to_process)}, need API: {len(keys_to_process)}")

    semaphore = asyncio.Semaphore(6)  # 6 parallel requests across 3 keys

    async def sem_task(idx, p, a, ck):
        async with semaphore:
            return await verify_item(client_pool, limiter, p, a, cache, ck)

    pbar = tqdm(total=len(keys_to_process), desc="Refining reasoning cache")
    batch_size = 15
    for i in range(0, len(keys_to_process), batch_size):
        batch = keys_to_process[i : i + batch_size]
        sub_tasks = [sem_task(idx, p, a, ck) for idx, p, a, ck in batch]
        results = await asyncio.gather(*sub_tasks)
        pbar.update(len(batch))
        await asyncio.sleep(0.5)

    pbar.close()
    print("Reasoning refinement completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="outputs/public_scored.csv")
    parser.add_argument("--min-prob", type=float, default=0.32)
    parser.add_argument("--max-prob", type=float, default=0.68)
    args = parser.parse_args()

    asyncio.run(run_refinement(args.csv_path, args.min_prob, args.max_prob))


if __name__ == "__main__":
    main()
