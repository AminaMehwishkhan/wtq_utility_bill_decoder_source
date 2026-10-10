"""Thin, dependency-light client for the competition-approved LLMs only.

Providers are called over plain HTTPS with ``requests`` (no vendor SDK needed).
Any model outside PERMITTED_MODELS is refused at construction time, so an
unapproved model can never be called by mistake.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

PERMITTED_MODELS: dict[str, tuple[str, ...]] = {
    "google": ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"),
    "anthropic": ("claude-haiku-4-5",),
    "openai": ("gpt-4.1-mini", "gpt-5-mini"),
}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any failure talking to a model (message is safe to log: keys are redacted)."""


@dataclass
class LLMConfig:
    provider: str = "google"
    model: str = "gemini-3.5-flash-lite"
    api_key: str = field(default="", repr=False)
    fallback_model: str = ""          # optional, must be permitted for the same provider
    timeout: float = 120.0
    max_retries: int = 3
    min_interval: float = 4.5         # seconds between calls (free tier: 15 requests/min)
    temperature: float | None = None  # unset -> provider default

    def validate(self) -> None:
        if self.provider not in PERMITTED_MODELS:
            raise LLMError(f"MODEL_PROVIDER must be one of {sorted(PERMITTED_MODELS)}, got {self.provider!r}")
        for m in filter(None, (self.model, self.fallback_model)):
            if m not in PERMITTED_MODELS[self.provider]:
                raise LLMError(f"model {m!r} is not a permitted {self.provider} model "
                               f"({', '.join(PERMITTED_MODELS[self.provider])})")
        if not self.api_key or self.api_key.startswith("your-api-key"):
            raise LLMError("API_KEY is not set. Copy .env.example to .env and add your key.")


def load_config() -> LLMConfig:
    """Read configuration from the environment (call load_dotenv() first)."""
    env = os.environ.get
    temp = env("LLM_TEMPERATURE", "").strip()
    return LLMConfig(
        provider=env("MODEL_PROVIDER", "google").strip().lower(),
        model=env("MODEL_NAME", "gemini-3.5-flash-lite").strip(),
        api_key=env("API_KEY", "").strip(),
        fallback_model=env("FALLBACK_MODEL_NAME", "").strip(),
        timeout=float(env("LLM_TIMEOUT", "120")),
        max_retries=int(env("LLM_MAX_RETRIES", "3")),
        min_interval=float(env("LLM_MIN_INTERVAL", "4.5")),
        temperature=float(temp) if temp else None,
    )


class LLMClient:
    def __init__(self, cfg: LLMConfig, *, sleep=time.sleep, clock=time.monotonic):
        cfg.validate()
        self.cfg = cfg
        self.model = cfg.model
        self._sleep, self._clock = sleep, clock
        self._last_call = float("-inf")
        self.calls = 0

    # ------------------------------------------------------------------ public
    def generate(self, system: str, user_text: str, images: list[bytes] | None = None,
                 max_tokens: int = 8192) -> str:
        """Send one request (with bounded retries) and return the model's text."""
        images = images or []
        switched = False
        while True:
            try:
                return self._with_retries(system, user_text, images, max_tokens)
            except _ModelNotFound as e:
                fb = self.cfg.fallback_model
                if fb and not switched and fb != self.model:
                    self.model, switched = fb, True       # still a permitted model
                    continue
                raise LLMError(str(e)) from None

    # ------------------------------------------------------------------ internals
    def _with_retries(self, system, user_text, images, max_tokens) -> str:
        last = "unknown error"
        for attempt in range(self.cfg.max_retries + 1):
            self._pace()
            try:
                url, headers, body = self._build(system, user_text, images, max_tokens)
                resp = requests.post(url, headers=headers, json=body, timeout=self.cfg.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                last = f"network error: {type(e).__name__}"
            else:
                self.calls += 1
                if resp.status_code == 200:
                    return self._parse(resp.json())
                last = f"HTTP {resp.status_code}: {self._redact(resp.text)[:300]}"
                if resp.status_code == 404:
                    raise _ModelNotFound(f"model {self.model!r} not found/available ({last})")
                if resp.status_code not in RETRYABLE_STATUS:
                    raise LLMError(last)
                retry_after = _retry_after(resp)
                if retry_after:
                    self._sleep(min(retry_after, 60))
            if attempt < self.cfg.max_retries:
                self._sleep(min(2 ** attempt * 3, 30))
        raise LLMError(f"giving up after {self.cfg.max_retries + 1} attempts ({last})")

    def _pace(self) -> None:
        wait = self.cfg.min_interval - (self._clock() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = self._clock()

    def _redact(self, text: str) -> str:
        return text.replace(self.cfg.api_key, "***") if self.cfg.api_key else text

    def _build(self, system, user_text, images, max_tokens):
        p, m, key = self.cfg.provider, self.model, self.cfg.api_key
        t = self.cfg.temperature
        b64 = [base64.b64encode(i).decode() for i in images]
        if p == "google":
            parts = [{"inlineData": {"mimeType": "image/jpeg", "data": d}} for d in b64]
            parts.append({"text": user_text})
            gen: dict[str, Any] = {"responseMimeType": "application/json", "maxOutputTokens": max_tokens}
            if t is not None:
                gen["temperature"] = t
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
            return (f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                    {"x-goog-api-key": key, "Content-Type": "application/json"}, body)
        if p == "anthropic":
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}}
                       for d in b64]
            content.append({"type": "text", "text": user_text})
            body = {"model": m, "max_tokens": max_tokens, "system": system,
                    "messages": [{"role": "user", "content": content}]}
            if t is not None:
                body["temperature"] = t
            return ("https://api.anthropic.com/v1/messages",
                    {"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"}, body)
        # openai
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}} for d in b64]
        content.append({"type": "text", "text": user_text})
        body = {"model": m, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content}]}
        if m.startswith("gpt-5"):
            body["max_completion_tokens"] = max_tokens      # reasoning model: no temperature
        else:
            body["max_tokens"] = max_tokens
            if t is not None:
                body["temperature"] = t
        return ("https://api.openai.com/v1/chat/completions",
                {"Authorization": "Bearer " + key, "Content-Type": "application/json"}, body)

    def _parse(self, data: dict[str, Any]) -> str:
        p = self.cfg.provider
        try:
            if p == "google":
                cands = data.get("candidates") or []
                if not cands:
                    raise LLMError(f"no candidates returned ({self._redact(str(data.get('promptFeedback', '')))[:200]})")
                cand = cands[0]
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(x.get("text", "") for x in parts if not x.get("thought"))
                if not text:
                    raise LLMError(f"empty response (finishReason={cand.get('finishReason')})")
                if cand.get("finishReason") == "MAX_TOKENS":
                    raise LLMError("response truncated (MAX_TOKENS)")
                return text
            if p == "anthropic":
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                if data.get("stop_reason") == "max_tokens":
                    raise LLMError("response truncated (max_tokens)")
            else:
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
                if choice.get("finish_reason") == "length":
                    raise LLMError("response truncated (length)")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError(f"unexpected response shape: {type(e).__name__}") from None
        if not text.strip():
            raise LLMError("empty response")
        return text


class _ModelNotFound(Exception):
    pass


def _retry_after(resp: requests.Response) -> float:
    try:
        return float(resp.headers.get("Retry-After", 0))
    except (TypeError, ValueError):
        return 0.0
