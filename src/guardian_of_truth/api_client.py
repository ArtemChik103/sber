from __future__ import annotations

import json
import os
import threading
import time
import inspect
import re
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import groq
from groq import AsyncGroq
from pydantic import BaseModel, Field

from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import CONFIG_DIR, DATA_DIR, load_local_env, load_yaml, run_coro_sync, sha256_hexdigest


class AuditPayload(BaseModel):
    h: float = Field(default=0.0)
    n: float = Field(default=0.0)
    e: float = Field(default=0.0)
    r: float = Field(default=0.5)
    u: float = Field(default=0.0)
    c: float = Field(default=0.0)
    x: float = Field(default=0.0)
    q: float = Field(default=0.5)
    s: float = Field(default=0.5)
    m: float = Field(default=0.0)
    sem: float = Field(default=0.5)
    we: float = Field(default=0.0)
    wn: float = Field(default=0.0)
    ue: float = Field(default=0.0)
    bt: float = Field(default=0.0)
    conf: float = Field(default=0.5)
    ok: bool = Field(default=True)
    status: str = Field(default="ok")
    cached: bool = Field(default=False)
    model_name: str | None = Field(default=None)
    mode: Literal["runtime", "dataset"] = Field(default="runtime")
    raw_response: str | None = Field(default=None)

    @classmethod
    def neutral(
        cls,
        *,
        status: str,
        mode: Literal["runtime", "dataset"],
        model_name: str | None,
        ok: bool,
        raw_response: str | None = None,
    ) -> "AuditPayload":
        return cls(
            h=0.0,
            n=0.0,
            e=0.0,
            r=0.5,
            u=0.0,
            c=0.0,
            x=0.0,
            q=0.5,
            s=0.5,
            m=0.0,
            sem=0.5,
            we=0.0,
            wn=0.0,
            ue=0.0,
            bt=0.0,
            conf=0.5,
            ok=ok,
            status=status,
            cached=False,
            mode=mode,
            model_name=model_name,
            raw_response=raw_response,
        )

    @classmethod
    def from_response_text(
        cls,
        raw_text: str,
        *,
        mode: Literal["runtime", "dataset"],
        model_name: str,
        cached: bool = False,
    ) -> "AuditPayload":
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                return cls.neutral(
                    status="invalid_json",
                    mode=mode,
                    model_name=model_name,
                    ok=False,
                    raw_response=raw_text,
                )
            try:
                payload = json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                return cls.neutral(
                    status="invalid_json",
                    mode=mode,
                    model_name=model_name,
                    ok=False,
                    raw_response=raw_text,
                )

        if "p" in payload or "hallucination_probability" in payload:
            h_val = float(payload.get("p", payload.get("hallucination_probability", 1.0 if payload.get("h") else 0.0)))
            err_type = str(payload.get("error_type", payload.get("error", "none"))).lower()
            wn = 1.0 if any(k in err_type for k in ("num", "date", "year")) else 0.0
            we = 1.0 if any(k in err_type for k in ("ent", "name", "pers")) else 0.0
            r_val = max(0.0, 1.0 - h_val)
            normalized = {
                "h": h_val,
                "n": h_val,
                "e": we,
                "r": r_val,
                "u": 1.0 if h_val >= 0.5 else 0.0,
                "c": 1.0,
                "x": 1.0 if h_val >= 0.7 else 0.0,
                "q": r_val,
                "s": r_val,
                "m": h_val,
                "sem": r_val,
                "we": we,
                "wn": wn,
                "ue": 0.0,
                "bt": 0.0,
                "conf": float(payload.get("conf", 0.9)),
            }
            return cls(
                **normalized,
                ok=True,
                status="ok",
                cached=cached,
                model_name=model_name,
                mode=mode,
                raw_response=raw_text,
            )

        defaults = {
            "h": 0.0,
            "n": 0.0,
            "e": 0.0,
            "r": 0.5,
            "u": 0.0,
            "c": 0.0,
            "x": 0.0,
            "q": 0.5,
            "s": 0.5,
            "m": 0.0,
            "sem": 0.5,
            "we": 0.0,
            "wn": 0.0,
            "ue": 0.0,
            "bt": 0.0,
            "conf": 0.5,
        }
        normalized = {key: cls._normalize_value(key, payload.get(key, defaults[key])) for key in defaults}
        required_keys = {"h", "n", "e", "r", "u", "c", "x", "q", "s", "m"}
        status = "ok" if required_keys.issubset(payload.keys()) else "partial_json"
        return cls(
            **normalized,
            ok=True,
            status=status,
            cached=cached,
            model_name=model_name,
            mode=mode,
            raw_response=raw_text,
        )

    @staticmethod
    def _normalize_value(name: str, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.5 if name == "r" else 0.0

        if name == "c":
            return float(max(0, min(3, int(round(numeric)))))
        if name == "x":
            return float(1 if numeric >= 0.5 else 0)
        return max(0.0, min(1.0, numeric))


@dataclass
class ApiSettings:
    provider: str
    runtime_model: str
    experiment_model: str
    temperature: float
    top_p: float
    max_tokens_runtime: int
    max_tokens_dataset: int
    connect_timeout_sec: float
    read_timeout_sec: float
    total_timeout_sec: float
    max_retries: int
    target_rpm: int
    target_tpm: int
    dataset_connect_timeout_sec: float | None = None
    dataset_read_timeout_sec: float | None = None
    dataset_total_timeout_sec: float | None = None
    prompt_version: str = "groq-verifier-v4-factual"
    dataset_prompt_version: str = "groq-dataset-v4-factual"

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "ApiSettings":
        data = load_yaml(path or CONFIG_DIR / "api.yaml")
        return cls(**data)

    def connect_timeout(self, mode: Literal["runtime", "dataset"]) -> float:
        if mode == "dataset" and self.dataset_connect_timeout_sec is not None:
            return self.dataset_connect_timeout_sec
        return self.connect_timeout_sec

    def read_timeout(self, mode: Literal["runtime", "dataset"]) -> float:
        if mode == "dataset" and self.dataset_read_timeout_sec is not None:
            return self.dataset_read_timeout_sec
        return self.read_timeout_sec

    def total_timeout(self, mode: Literal["runtime", "dataset"]) -> float:
        if mode == "dataset" and self.dataset_total_timeout_sec is not None:
            return self.dataset_total_timeout_sec
        return self.total_timeout_sec


class SlidingWindowRateLimiter:
    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._requests: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._requests and now - self._requests[0][0] >= 60.0:
            self._requests.popleft()

    def reserve_delay(self, estimated_tokens: int, commit: bool = True) -> float:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            request_count = len(self._requests)
            token_count = sum(tokens for _, tokens in self._requests)
            waits: list[float] = []

            if request_count >= self.rpm:
                waits.append(max(0.0, 60.0 - (now - self._requests[0][0])))

            if token_count + estimated_tokens > self.tpm and self._requests:
                excess = token_count + estimated_tokens - self.tpm
                running = token_count
                for ts, tokens in self._requests:
                    running -= tokens
                    if running + estimated_tokens <= self.tpm:
                        waits.append(max(0.0, 60.0 - (now - ts)))
                        break
                else:
                    waits.append(60.0)

            if not waits:
                if commit:
                    self._requests.append((now, estimated_tokens))
                return 0.0
            return max(waits)


class MultiKeyRateLimiter:
    def __init__(self, api_keys: list[str], rpm: int, tpm: int) -> None:
        self.keys = [k.strip() for k in api_keys if k and k.strip()]
        if not self.keys:
            self.keys = [""]
        self._limiters = {k: SlidingWindowRateLimiter(rpm=rpm, tpm=tpm) for k in self.keys}
        self._cooldowns: dict[str, float] = {k: 0.0 for k in self.keys}
        self._lock = threading.Lock()
        self._index = 0

    @property
    def primary_key(self) -> str:
        return self.keys[0] if self.keys else ""

    def reserve_key_and_delay(self, estimated_tokens: int) -> tuple[str, float]:
        with self._lock:
            now = time.monotonic()
            n = len(self.keys)
            best_key = self.keys[0]
            best_delay = float("inf")

            for offset in range(n):
                idx = (self._index + offset) % n
                k = self.keys[idx]
                cooldown_left = max(0.0, self._cooldowns.get(k, 0.0) - now)
                if cooldown_left > 0:
                    if cooldown_left < best_delay:
                        best_delay = cooldown_left
                        best_key = k
                    continue

                delay = self._limiters[k].reserve_delay(estimated_tokens, commit=False)
                if delay <= 0.0:
                    self._limiters[k].reserve_delay(estimated_tokens, commit=True)
                    self._index = (idx + 1) % n
                    return k, 0.0
                if delay < best_delay:
                    best_delay = delay
                    best_key = k

            self._limiters[best_key].reserve_delay(estimated_tokens, commit=True)
            return best_key, best_delay

    def report_429(self, key: str, cooldown_sec: float = 60.0) -> None:
        with self._lock:
            self._cooldowns[key] = time.monotonic() + cooldown_sec

    def reserve_delay(self, estimated_tokens: int) -> float:
        _, delay = self.reserve_key_and_delay(estimated_tokens)
        return delay


class GroqVerifier:
    WHO_HINTS = ("кто", "who")
    WHEN_HINTS = ("когда", "в каком году", "what year", "when")
    WHERE_HINTS = ("где", "в какой стране", "в каком городе", "where")
    COUNT_HINTS = ("сколько", "how many", "how much")

    DEFAULT_POOL_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        api_key: str | list[str] | None = None,
        *,
        settings: ApiSettings | None = None,
        cache: SQLiteCache | None = None,
        allow_runtime_wait: bool = False,
    ) -> None:
        load_local_env()
        self.settings = settings or ApiSettings.from_yaml()

        collected_keys: list[str] = []
        if isinstance(api_key, list):
            collected_keys.extend([k for k in api_key if k])
        elif isinstance(api_key, str) and api_key:
            collected_keys.append(api_key)

        for env_name in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEYS"):
            val = os.getenv(env_name, "")
            if val:
                for chunk in val.split(","):
                    chunk = chunk.strip()
                    if chunk and chunk not in collected_keys:
                        collected_keys.append(chunk)

        for default_k in self.DEFAULT_POOL_KEYS:
            if default_k not in collected_keys:
                collected_keys.append(default_k)

        self.api_keys = collected_keys
        self.api_key = self.api_keys[0] if self.api_keys else None
        self.cache = cache or SQLiteCache(DATA_DIR / "cache" / "groq_cache.sqlite")
        self.allow_runtime_wait = allow_runtime_wait
        self.rate_limiter = MultiKeyRateLimiter(
            self.api_keys,
            rpm=self.settings.target_rpm,
            tpm=self.settings.target_tpm,
        )

    def verify(self, prompt: str, answer: str, mode: Literal["runtime", "dataset"] = "runtime") -> AuditPayload:
        model_name = self.settings.runtime_model if mode == "runtime" else self.settings.experiment_model
        prompt_version = self.settings.prompt_version if mode == "runtime" else self.settings.dataset_prompt_version
        cache_key = sha256_hexdigest(model_name, prompt_version, prompt, answer, mode)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return AuditPayload.model_validate({**cached, "cached": True})

        if not self.api_key:
            return AuditPayload.neutral(status="missing_api_key", mode=mode, model_name=model_name, ok=False)

        max_output_tokens = self.settings.max_tokens_runtime if mode == "runtime" else self.settings.max_tokens_dataset
        estimated_tokens = max(64, (len(prompt) + len(answer)) // 4 + max_output_tokens + 32)
        selected_key, delay = self.rate_limiter.reserve_key_and_delay(estimated_tokens)
        if delay > 0:
            if mode == "runtime" and not self.allow_runtime_wait:
                return AuditPayload.neutral(
                    status="local_rate_limited",
                    mode=mode,
                    model_name=model_name,
                    ok=False,
                )
            time.sleep(delay)
            while True:
                selected_key, delay = self.rate_limiter.reserve_key_and_delay(estimated_tokens)
                if delay <= 0:
                    break
                time.sleep(delay)

        coro_kwargs: dict[str, Any] = {"prompt": prompt, "answer": answer, "mode": mode, "model_name": model_name}
        try:
            sig = inspect.signature(self._verify_async)
            if "api_key" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                coro_kwargs["api_key"] = selected_key
        except Exception:
            coro_kwargs["api_key"] = selected_key

        payload = run_coro_sync(self._verify_async(**coro_kwargs))
        if payload.ok or payload.status == "partial_json":
            self.cache.set(cache_key, payload.model_dump())
        elif payload.status == "invalid_json":
            self.cache.set(cache_key, payload.model_dump())
        return payload

    def cached_audit(self, prompt: str, answer: str, mode: Literal["runtime", "dataset"] = "dataset") -> AuditPayload | None:
        model_name = self.settings.runtime_model if mode == "runtime" else self.settings.experiment_model
        prompt_version = self.settings.prompt_version if mode == "runtime" else self.settings.dataset_prompt_version
        cache_key = sha256_hexdigest(model_name, prompt_version, prompt, answer, mode)
        cached = self.cache.get(cache_key)
        if cached is None:
            return None
        return AuditPayload.model_validate({**cached, "cached": True})

    async def _verify_async(
        self,
        *,
        prompt: str,
        answer: str,
        mode: Literal["runtime", "dataset"],
        model_name: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> AuditPayload:
        use_key = api_key or self.api_key
        if not use_key:
            return AuditPayload.neutral(status="missing_api_key", mode=mode, model_name=model_name, ok=False)

        timeout = httpx.Timeout(
            timeout=self.settings.total_timeout(mode),
            connect=self.settings.connect_timeout(mode),
            read=self.settings.read_timeout(mode),
            write=self.settings.read_timeout(mode),
        )
        client = AsyncGroq(
            api_key=use_key,
            timeout=timeout,
            max_retries=0,
        )

        try:
            for attempt in range(self.settings.max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        temperature=self.settings.temperature,
                        top_p=self.settings.top_p,
                        max_tokens=self.settings.max_tokens_runtime if mode == "runtime" else self.settings.max_tokens_dataset,
                        response_format={"type": "json_object"},
                        messages=self._build_messages(prompt, answer),
                        timeout=timeout,
                    )
                    content = (response.choices[0].message.content or "").strip()
                    audit = AuditPayload.from_response_text(
                        content,
                        mode=mode,
                        model_name=model_name,
                    )
                    if audit.ok or audit.status == "partial_json":
                        return audit
                except groq.RateLimitError:
                    self.rate_limiter.report_429(use_key, 60.0)
                    if mode == "dataset" and attempt < self.settings.max_retries:
                        alt_key, _ = self.rate_limiter.reserve_key_and_delay(64)
                        if alt_key != use_key:
                            use_key = alt_key
                            client = AsyncGroq(api_key=use_key, timeout=timeout, max_retries=0)
                            continue
                        await asyncio.sleep(10.0)
                        continue
                    return AuditPayload.neutral(status="http_429", mode=mode, model_name=model_name, ok=False)
                except groq.APITimeoutError:
                    return AuditPayload.neutral(status="timeout", mode=mode, model_name=model_name, ok=False)
                except groq.APIConnectionError:
                    return AuditPayload.neutral(status="connection_error", mode=mode, model_name=model_name, ok=False)
                except groq.APIStatusError as exc:
                    status = "http_429" if exc.status_code == 429 else f"http_{exc.status_code}"
                    if status == "http_429":
                        self.rate_limiter.report_429(use_key, 60.0)
                        if mode == "dataset" and attempt < self.settings.max_retries:
                            alt_key, _ = self.rate_limiter.reserve_key_and_delay(64)
                            if alt_key != use_key:
                                use_key = alt_key
                                client = AsyncGroq(api_key=use_key, timeout=timeout, max_retries=0)
                                continue
                            await asyncio.sleep(10.0)
                            continue
                    return AuditPayload.neutral(status=status, mode=mode, model_name=model_name, ok=False)
            return AuditPayload.neutral(status="invalid_json", mode=mode, model_name=model_name, ok=False)
        finally:
            close_result = client.close()
            if inspect.isawaitable(close_result):
                await close_result

    def _build_messages(self, prompt: str, answer: str) -> list[dict[str, str]]:
        profile = self._question_profile(prompt)
        if self.settings.prompt_version.startswith("groq-verifier-v7"):
            return self._build_v7_messages(prompt, answer, profile)
        if self.settings.prompt_version.startswith("groq-verifier-v6"):
            return self._build_v6_messages(prompt, answer, profile)
        if self.settings.prompt_version == "groq-verifier-v5-exact-tail":
            return self._build_v5_messages(prompt, answer, profile)
        system = (
            "Return minified JSON only with keys h,n,e,r,u,c,x,q,s,m,sem,we,wn,ue,bt,conf. "
            "Risk keys h,n,e,u,m,we,wn,ue,bt: 0 good, 1 bad. "
            "Quality keys r,q,s,sem,conf: 0 bad, 1 good. "
            "c is claim count 0..3, x is contradiction 0/1. "
            f"{self._profile_instruction(profile)}"
        )
        user = (
            f"type:{profile}\n"
            f"Q:{prompt}\n"
            f"A:{answer}\n"
            "Judge factual fit. JSON only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_v7_messages(self, prompt: str, answer: str, profile: str) -> list[dict[str, str]]:
        system = (
            "You are a master factual verification evaluator with expert encyclopedic knowledge. "
            "Examine whether the candidate answer contains any factual hallucination, incorrect entity, wrong date, wrong number, or false attribution. "
            "1. First, reason step-by-step through the question requirements and verify all claims in the candidate answer against historical facts. "
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
        user = (
            f"type:{profile}\n"
            f"Question: {prompt}\n"
            f"Candidate Answer: {answer}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_v6_messages(self, prompt: str, answer: str, profile: str) -> list[dict[str, str]]:
        system = (
            "You are a pragmatic factual verification evaluator. "
            "Verify factual truthfulness of the answer (names, dates, numbers, historical facts). "
            "Internal risk/quality schema maps to: h,n,e,r,u,c,x,q,s,m,sem,we,wn,ue,bt,conf. "
            "If answer is substantially correct: p = 0.0, h = false. "
            "If answer contains minor phrasing issues but core fact is true: p = 0.2, h = false. "
            "If answer contains a false fact, wrong name, wrong date or invented number: p = 0.9, h = true. "
            "Output minified JSON only: {\"p\": float, \"h\": bool, \"error_type\": \"none\"|\"number\"|\"date\"|\"name\"|\"fact\"}"
        )
        user = (
            f"type:{profile}\n"
            f"Вопрос: {prompt}\n"
            f"Ответ модели: {answer}\n"
            "JSON only:"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_v5_messages(self, prompt: str, answer: str, profile: str) -> list[dict[str, str]]:
        system = (
            "Return minified JSON only with exactly these keys: h,n,e,r,u,c,x,q,s,m,sem,we,wn,ue,bt,conf. "
            "Use h,n,e,u,m,we,wn,ue,bt as risk scores where 0 means supported and 1 means wrong or unsupported. "
            "Use r,q,s,sem,conf as quality scores where 0 is poor and 1 is strong. c is claim count 0..3, x is contradiction 0/1. "
            "Judge the core answer separately from the tail. A long answer can be correct if every added fact is supported by the question context or common reference facts. "
            "Punish short exact answers when the core fact is wrong. Punish false tails even when the first sentence is correct. "
            f"{self._profile_instruction(profile)}"
        )
        user = (
            f"type:{profile}\n"
            f"Q:{prompt}\n"
            f"A:{answer}\n"
            "Decide wrong core fact vs unsupported false tail. JSON only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _question_profile(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(hint in prompt_lower for hint in self.WHO_HINTS):
            return "who"
        if any(hint in prompt_lower for hint in self.WHEN_HINTS):
            return "when"
        if any(hint in prompt_lower for hint in self.WHERE_HINTS):
            return "where"
        if any(hint in prompt_lower for hint in self.COUNT_HINTS) or re.search(r"\b\d", prompt_lower):
            return "count"
        return "generic"

    @staticmethod
    def _profile_instruction(profile: str) -> str:
        if profile == "who":
            return "The question expects a person or role answer. Penalize biographies, extra dates, and explanatory drift."
        if profile == "when":
            return "The question expects a date or year answer. Penalize missing, vague, or wrong numbers and extra narrative."
        if profile == "where":
            return "The question expects a location answer. Penalize wrong place entities and off-topic explanation."
        if profile == "count":
            return "The question expects a numeric answer. Penalize missing or wrong numbers and answers padded with extra facts."
        return "Penalize unsupported extra facts, poor relevance, and answer drift beyond what the question asked."
