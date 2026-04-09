from __future__ import annotations

import json
import os
import threading
import time
import inspect
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import groq
from groq import AsyncGroq
from pydantic import BaseModel, Field

from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import CONFIG_DIR, DATA_DIR, load_yaml, run_coro_sync, sha256_hexdigest


class AuditPayload(BaseModel):
    h: float = Field(default=0.0)
    n: float = Field(default=0.0)
    e: float = Field(default=0.0)
    r: float = Field(default=0.5)
    u: float = Field(default=0.0)
    c: float = Field(default=0.0)
    x: float = Field(default=0.0)
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

        defaults = {"h": 0.0, "n": 0.0, "e": 0.0, "r": 0.5, "u": 0.0, "c": 0.0, "x": 0.0}
        normalized = {key: cls._normalize_value(key, payload.get(key, defaults[key])) for key in defaults}
        status = "ok" if set(defaults).issubset(payload.keys()) else "partial_json"
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
    prompt_version: str = "groq-verifier-v1"
    dataset_prompt_version: str = "groq-dataset-v1"

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "ApiSettings":
        data = load_yaml(path or CONFIG_DIR / "api.yaml")
        return cls(**data)


class SlidingWindowRateLimiter:
    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._requests: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._requests and now - self._requests[0][0] >= 60.0:
            self._requests.popleft()

    def reserve_delay(self, estimated_tokens: int) -> float:
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
                self._requests.append((now, estimated_tokens))
                return 0.0
            return max(waits)


class GroqVerifier:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        settings: ApiSettings | None = None,
        cache: SQLiteCache | None = None,
        allow_runtime_wait: bool = False,
    ) -> None:
        self.settings = settings or ApiSettings.from_yaml()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.cache = cache or SQLiteCache(DATA_DIR / "cache" / "groq_cache.sqlite")
        self.allow_runtime_wait = allow_runtime_wait
        self.rate_limiter = SlidingWindowRateLimiter(
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

        estimated_tokens = max(32, (len(prompt) + len(answer)) // 4 + 32)
        delay = self.rate_limiter.reserve_delay(estimated_tokens)
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
                delay = self.rate_limiter.reserve_delay(estimated_tokens)
                if delay <= 0:
                    break
                time.sleep(delay)

        payload = run_coro_sync(self._verify_async(prompt=prompt, answer=answer, mode=mode, model_name=model_name))
        if payload.ok or payload.status == "partial_json":
            self.cache.set(cache_key, payload.model_dump())
        elif payload.status == "invalid_json":
            self.cache.set(cache_key, payload.model_dump())
        return payload

    async def _verify_async(
        self,
        *,
        prompt: str,
        answer: str,
        mode: Literal["runtime", "dataset"],
        model_name: str,
    ) -> AuditPayload:
        timeout = httpx.Timeout(
            timeout=self.settings.total_timeout_sec,
            connect=self.settings.connect_timeout_sec,
            read=self.settings.read_timeout_sec,
            write=self.settings.read_timeout_sec,
        )
        client = AsyncGroq(
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,
        )

        try:
            for _ in range(self.settings.max_retries + 1):
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
                    return AuditPayload.neutral(status="http_429", mode=mode, model_name=model_name, ok=False)
                except groq.APITimeoutError:
                    return AuditPayload.neutral(status="timeout", mode=mode, model_name=model_name, ok=False)
                except groq.APIConnectionError:
                    return AuditPayload.neutral(status="connection_error", mode=mode, model_name=model_name, ok=False)
                except groq.APIStatusError as exc:
                    status = "http_429" if exc.status_code == 429 else f"http_{exc.status_code}"
                    return AuditPayload.neutral(status=status, mode=mode, model_name=model_name, ok=False)
            return AuditPayload.neutral(status="invalid_json", mode=mode, model_name=model_name, ok=False)
        finally:
            close_result = client.close()
            if inspect.isawaitable(close_result):
                await close_result

    def _build_messages(self, prompt: str, answer: str) -> list[dict[str, str]]:
        system = (
            "Return only JSON: {h,n,e,r,u,c,x}. "
            "h,n,e,u,r are 0..1 floats; c is 0..3; x is 0 or 1."
        )
        user = (
            f"Q:{prompt}\n"
            f"A:{answer}\n"
            "Score factual risk only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
