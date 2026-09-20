import sys
import time
import pandas as pd
from pathlib import Path
from groq import Groq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import sha256_hexdigest, DATA_DIR, load_local_env

sys.stdout.reconfigure(encoding='utf-8')
load_local_env()

KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]

cache = SQLiteCache(DATA_DIR / "cache" / "groq_cache.sqlite")
df = pd.read_csv("knowledge_bench_private_scores.csv")
p = df["predict_proba"]
uncertain = df[(p >= 0.45) & (p <= 0.75)]

to_fetch = []
for idx, r in uncertain.iterrows():
    p_text = str(r["prompt"])
    ckey = sha256_hexdigest("blind-fact-v1", p_text)
    if cache.get(ckey) is None:
        to_fetch.append((idx, p_text))

print(f"Found {len(to_fetch)} private uncertain prompts lacking blind facts.")

clients = [Groq(api_key=k) for k in KEYS]

for i, (idx, p_text) in enumerate(to_fetch):
    ckey = sha256_hexdigest("blind-fact-v1", p_text)
    client = clients[i % len(clients)]
    print(f"[{i+1}/{len(to_fetch)}] Private Row {idx}: {p_text[:60]}...")
    try:
        resp = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": "Ответь на фактический вопрос предельно кратко (только имя, фамилия, год, дата, число или ключевой факт в 1-3 словах, без лишних слов)."},
                {"role": "user", "content": p_text}
            ],
            temperature=0.0,
            max_tokens=25,
        )
        fact = resp.choices[0].message.content.strip()
        print(f"  -> Fact: {fact}")
        cache.set(ckey, {"blind_fact": fact, "model": "qwen/qwen3.8-27b", "version": "v1"})
        time.sleep(0.5)
    except Exception as e:
        print(f"  -> Error: {e}")

print("Done fetching private blind facts!")
