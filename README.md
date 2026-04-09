# Guardian of Truth

Production-like factual hallucination detector with a baseline-compatible `GuardianOfTruth.score(prompt, answer)` interface, Groq runtime verification, compact feature extraction, and a lightweight local classifier.

The implementation is API-only. It does not use local LLM inference and does not train on `data/bench/knowledge_bench_public.csv`.
