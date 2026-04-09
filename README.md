# Guardian of Truth

Production-like factual hallucination detector with a baseline-compatible `GuardianOfTruth.score(prompt, answer)` interface, Groq runtime verification, compact feature extraction, and a lightweight local classifier.

The implementation is API-only. It does not use local LLM inference and does not train on `data/bench/knowledge_bench_public.csv`.

## Setup

Python `3.11` is required.

```bash
cp .env.example .env
export GROQ_API_KEY=your_new_key_here
./scripts/install.sh
```

The Groq token shared in the handoff should be treated as compromised and rotated before real use.

## Main Commands

Generate or extend the synthetic dataset in small batches:

```bash
./scripts/generate_dataset.sh --stage seed-harvest --resume --limit 300
./scripts/generate_dataset.sh --stage rule-negatives --resume
./scripts/generate_dataset.sh --stage groq-negatives --resume --limit 100
```

Estimate wall time under the configured free-plan limits:

```bash
./scripts/estimate_pipeline.sh --planned-groq-negatives 300
```

Train on synthetic data only:

```bash
./scripts/train.sh --dataset-path data/raw/synthetic_factual_data.jsonl
```

Score the public benchmark sequentially with checkpointing:

```bash
./scripts/score_public.sh --csv-path data/bench/knowledge_bench_public.csv --limit 150
./scripts/score_public.sh --csv-path data/bench/knowledge_bench_public.csv
```

Run smoke checks:

```bash
./scripts/smoke.sh
```

## Runtime Design

`GuardianOfTruth.score(prompt, answer)` does one short Groq audit call in runtime mode, extracts `7` API features and `7` text features, then applies a local calibrated classifier. If the API path fails with timeout, `429`, invalid JSON, missing key, or local rate-limit pressure, the runtime falls back to a deterministic text-only classifier.

`t_model_sec` measures the Groq API segment. `t_overhead_sec` measures local feature extraction, classifier inference, and fallback logic. Cached API hits report near-zero model time.

## Compliance Notes

- `train.py` reads only synthetic JSONL data.
- `data/bench/knowledge_bench_public.csv` is reserved for offline evaluation and error analysis.
- `data/cache/groq_cache.sqlite` prevents duplicate token spend for repeated `(prompt, answer, mode)` requests.
- Public scoring is sequential by design and supports resume from a checkpoint file.
