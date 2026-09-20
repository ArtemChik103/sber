import os
import sys
import time
import json
import asyncio
import pandas as pd
from pathlib import Path
from groq import AsyncGroq, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import sha256_hexdigest, load_local_env

sys.stdout.reconfigure(encoding='utf-8')
load_local_env()

KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]

cache = SQLiteCache("data/cache/groq_cache.sqlite")

df = pd.read_csv("knowledge_bench_private_scores.csv")
df["uncertainty"] = 1.0 - (df["predict_proba"] - 0.5).abs() * 2.0

# Take top 150 most uncertain private cases
top150 = df.sort_values("uncertainty", ascending=False).head(150)
prompts_to_fetch = []

for idx, r in top150.iterrows():
    p = r["prompt"]
    ckey = sha256_hexdigest("blind-fact-v1", p)
    if cache.get(ckey) is None:
        prompts_to_fetch.append(p)

print(f"Total candidate prompts: {len(top150)}, to fetch: {len(prompts_to_fetch)}")

if not prompts_to_fetch:
    print("All prompts already cached!")
    sys.exit(0)

clients = [AsyncGroq(api_key=k, max_retries=1) for k in KEYS]
sem = asyncio.Semaphore(4)

async def fetch_one(p: str, key_idx: int):
    ckey = sha256_hexdigest("blind-fact-v1", p)
    client = clients[key_idx % len(clients)]
    
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": "Ответь на фактический вопрос предельно кратко (только имя, фамилия, год, дата, число или ключевой факт в 1-3 словах, без лишних слов)."},
                        {"role": "user", "content": p}
                    ],
                    temperature=0.0,
                    max_tokens=25,
                )
                fact = resp.choices[0].message.content.strip()
                cache.set(ckey, {"blind_fact": fact, "model": "qwen/qwen3.8-27b", "version": "v1"})
                return fact
            except RateLimitError:
                await asyncio.sleep(4.0)
                client = clients[(key_idx + 1) % len(clients)]
            except Exception as e:
                if attempt == 2:
                    print(f"Error fetching for '{p[:30]}...': {e}", flush=True)
                    return ""
                await asyncio.sleep(2.0)
    return ""

async def main():
    tasks = [fetch_one(p, i) for i, p in enumerate(prompts_to_fetch)]
    batch_size = 30
    for i in range(0, len(tasks), batch_size):
        chunk = tasks[i:i + batch_size]
        print(f"Processing batch {i}..{i + len(chunk)} of {len(tasks)}...", flush=True)
        t0 = time.time()
        await asyncio.gather(*chunk)
        elapsed = time.time() - t0
        print(f"Batch completed in {elapsed:.1f}s. Total done: {min(i + batch_size, len(tasks))}", flush=True)
        if elapsed < 15.0 and (i + batch_size) < len(tasks):
            await asyncio.sleep(15.0 - elapsed)

if __name__ == "__main__":
    asyncio.run(main())
    print("Private blind recall prefill completed successfully!", flush=True)
