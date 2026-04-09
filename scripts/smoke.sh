#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from guardian_of_truth.guardian import GuardianOfTruth

guard = GuardianOfTruth()
sample = guard.score("В каком году был основан Санкт-Петербург?", "Санкт-Петербург был основан в 1703 году.")
print("local_sample", sample)

fallback_guard = GuardianOfTruth()
fallback_guard.verifier.verify = lambda prompt, answer, mode="runtime": __import__("guardian_of_truth.api_client", fromlist=["AuditPayload"]).AuditPayload.neutral(
    status="forced_fallback", mode="runtime", model_name="llama-3.1-8b-instant", ok=False
)
fallback = fallback_guard.score("Кто был первым президентом США?", "Первым президентом США был Джордж Вашингтон.")
print("forced_fallback", fallback)
PY

if [[ -n "${GROQ_API_KEY:-}" ]]; then
  python - <<'PY'
from guardian_of_truth.guardian import GuardianOfTruth

guard = GuardianOfTruth()
sample = guard.score("В каком году Юрий Гагарин совершил первый полёт в космос?", "Юрий Гагарин совершил первый полёт в космос в 1964 году.")
print("api_sample", sample)
PY
else
  echo "api_sample skipped: GROQ_API_KEY is not set"
fi
