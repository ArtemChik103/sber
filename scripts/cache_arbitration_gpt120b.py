import os
import sys
import time
import json
import re
from pathlib import Path
import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guardian_of_truth.api_client import GroqVerifier
from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import sha256_hexdigest, DATA_DIR

sys.stdout.reconfigure(encoding='utf-8')

cache_path = DATA_DIR / "cache" / "groq_cache.sqlite"
cache = SQLiteCache(cache_path)
gv = GroqVerifier()

keys = [k for k in gv.api_keys if k]
print(f"Available Groq API keys: {len(keys)}")

def extract_json_from_llm(content: str) -> dict | None:
    m = re.search(r"\{[^{}]*\"is_hallucination\"[^{}]*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    is_h = None
    if re.search(r"\"is_hallucination\"\s*:\s*true", content, re.IGNORECASE):
        is_h = True
    elif re.search(r"\"is_hallucination\"\s*:\s*false", content, re.IGNORECASE):
        is_h = False
    if is_h is not None:
        conf = 0.90
        m_c = re.search(r"\"confidence\"\s*:\s*([0-9.]+)", content)
        if m_c:
            try:
                conf = float(m_c.group(1))
            except Exception:
                pass
        return {"is_hallucination": is_h, "confidence": conf, "explanation": "robust_parsed"}
    return None

def arbitrate_sample(prompt: str, answer: str, key_idx: int) -> tuple[dict | None, int]:
    ckey = sha256_hexdigest("gpt120b-arbitration-v1", prompt, answer)
    cached = cache.get(ckey)
    if cached is not None and "is_hallucination" in cached:
        return cached, key_idx

    url = "https://api.groq.com/openai/v1/chat/completions"
    
    sys_prompt = """Определи, содержит ли ответ фактическую галлюцинацию (ошибку).
Вопрос: """ + prompt + """
Ответ: """ + answer + """

Ответь строго в JSON формате: {"is_hallucination": true/false, "confidence": 0.0-1.0, "explanation": "краткое пояснение"}"""

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": sys_prompt}],
        "max_tokens": 1000,
        "temperature": 0.0
    }
    
    for attempt in range(len(keys)):
        k = keys[(key_idx + attempt) % len(keys)]
        headers = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                res = extract_json_from_llm(content)
                if res is not None:
                    cache.set(ckey, res)
                    return res, (key_idx + attempt)
                else:
                    print(f"  Failed to parse content: {content[:150]}")
            elif r.status_code == 429:
                print(f"  Key {attempt} hit rate limit (429), switching to next key...")
                continue
            else:
                print(f"  Error {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"  Exception on attempt {attempt}: {e}")
            time.sleep(1.0)
            
    return None, key_idx

def run():
    # 1. Public gray zone
    pub_df = pd.read_csv("outputs/public_scored.csv")
    pub_gray = pub_df[(pub_df["is_hallucination_proba"] >= 0.38) & (pub_df["is_hallucination_proba"] <= 0.62)]
    print(f"\nFound {len(pub_gray)} public gray-zone samples [0.38, 0.62].")
    
    k_idx = 1 # active keys
    success_pub = 0
    for i, (idx, r) in enumerate(pub_gray.iterrows()):
        p = str(r["prompt"])
        a = str(r["model_answer"])
        ckey = sha256_hexdigest("gpt120b-arbitration-v1", p, a)
        if cache.get(ckey) is not None:
            success_pub += 1
            continue
        print(f"[{i+1}/{len(pub_gray)}] Arbitrating public row {idx}...")
        res, k_idx = arbitrate_sample(p, a, k_idx)
        if res:
            success_pub += 1
            print(f"  -> Hallucination: {res.get('is_hallucination')}, Conf: {res.get('confidence')}")
        else:
            print("  -> Failed to arbitrate.")
        time.sleep(1.0)

    print(f"Public gray-zone arbitration completed: {success_pub}/{len(pub_gray)} cached.")

    # 2. Private gray zone
    priv_df = pd.read_csv("knowledge_bench_private_scores.csv")
    priv_gray = priv_df[(priv_df["predict_proba"] >= 0.38) & (priv_df["predict_proba"] <= 0.62)]
    print(f"\nFound {len(priv_gray)} private gray-zone samples [0.38, 0.62].")
    
    success_priv = 0
    for i, (idx, r) in enumerate(priv_gray.iterrows()):
        p = str(r["prompt"])
        a = str(r["model_answer"])
        ckey = sha256_hexdigest("gpt120b-arbitration-v1", p, a)
        if cache.get(ckey) is not None:
            success_priv += 1
            continue
        print(f"[{i+1}/{len(priv_gray)}] Arbitrating private row {idx}...")
        res, k_idx = arbitrate_sample(p, a, k_idx)
        if res:
            success_priv += 1
            print(f"  -> Hallucination: {res.get('is_hallucination')}, Conf: {res.get('confidence')}")
        else:
            print("  -> Failed to arbitrate.")
        time.sleep(1.0)

    print(f"Private gray-zone arbitration completed: {success_priv}/{len(priv_gray)} cached.")

if __name__ == "__main__":
    run()
