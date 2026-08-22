"""LLM backends.

Default experimental generator: Gemini 3.6 Flash via the Gemini API.
Local Hugging Face (FLAN-T5) and OpenAI-compatible endpoints remain available.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from config import CONFIG, Config

# Free-tier GenerateRequestsPerMinutePerProject limit is 5 RPM.
# Space every generate_content call by at least 20s so a 15-question run
# stays below 5 requests/minute instead of retrying into 429s.
GEMINI_FREE_TIER_MIN_INTERVAL_SECONDS = 20.0
GEMINI_RATE_LIMIT_SAFETY_MARGIN_SECONDS = 5.0
GEMINI_RATE_LIMIT_MAX_RETRIES = 12
GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS = 180.0

# One clock for the whole process so pacing is not reset per question/client.
_LAST_GENERATE_CONTENT_MONO: float | None = None
_PACE_LOCK = threading.Lock()

_RETRY_DELAY_RE = re.compile(
    r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


@dataclass
class LLMOutput:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    raw: dict[str, Any]


class HuggingFaceLocalLLM:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        self._model = None
        self._tokenizer = None
        self._is_encoder_decoder = True

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

        model_name = self.cfg.llm_model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
            self._is_encoder_decoder = True
        except (ValueError, OSError):
            self._model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self._is_encoder_decoder = False
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.to(self.cfg.llm_device)
        self._model.eval()

    def _format_prompt(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        if (
            not self._is_encoder_decoder
            and hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template
        ):
            return tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You answer financial questions using only the provided evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompt: str) -> LLMOutput:
        import time

        import torch

        self._load()
        tokenizer = self._tokenizer
        formatted = self._format_prompt(prompt)
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.llm_max_input_tokens,
        )
        encoded = {k: v.to(self.cfg.llm_device) for k, v in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[1])

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.cfg.llm_max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
        }
        if self.cfg.llm_temperature and self.cfg.llm_temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.cfg.llm_temperature
        else:
            generate_kwargs["do_sample"] = False

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(**encoded, **generate_kwargs)
        latency = time.perf_counter() - t0

        if self._is_encoder_decoder:
            generated = output_ids[0]
        else:
            generated = output_ids[0][input_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        output_tokens = int(generated.shape[0])
        return LLMOutput(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            raw={"model": self.cfg.llm_model, "provider": "huggingface_local"},
        )


class OpenAICompatibleLLM:
    """Optional OpenAI-compatible endpoint (vLLM, Ollama, Groq, etc.)."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")

    def generate(self, prompt: str) -> LLMOutput:
        import time

        import requests

        t0 = time.perf_counter()
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.cfg.llm_model,
                "temperature": self.cfg.llm_temperature,
                "max_tokens": self.cfg.llm_max_new_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer financial questions using only provided evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        latency = time.perf_counter() - t0
        text = payload["choices"][0]["message"]["content"].strip()
        usage = payload.get("usage") or {}
        return LLMOutput(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_seconds=latency,
            raw={"provider": "openai_compatible", "model": self.cfg.llm_model},
        )


def _walk_retry_delay(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("retryDelay") or value.get("retry_delay")
        if raw is not None:
            parsed = _parse_delay_value(raw)
            if parsed is not None:
                return parsed
        for nested in value.values():
            parsed = _walk_retry_delay(nested)
            if parsed is not None:
                return parsed
    elif isinstance(value, (list, tuple)):
        for item in value:
            parsed = _walk_retry_delay(item)
            if parsed is not None:
                return parsed
    elif isinstance(value, str):
        match = _RETRY_DELAY_RE.search(value)
        if match:
            return float(match.group(1))
    return None


def _parse_delay_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        if "seconds" in raw:
            nanos = float(raw.get("nanos") or 0) / 1e9
            return float(raw["seconds"]) + nanos
        return _walk_retry_delay(raw)
    text = str(raw).strip()
    if text.endswith("s") and text[:-1].replace(".", "", 1).isdigit():
        return float(text[:-1])
    try:
        return float(text)
    except ValueError:
        match = _RETRY_DELAY_RE.search(text)
        return float(match.group(1)) if match else None


def is_gemini_rate_limit_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "")
    blob = f"{status} {exc}"
    if code == 429:
        return True
    return "RESOURCE_EXHAUSTED" in blob or "GenerateRequestsPerMinute" in blob


def gemini_retry_delay_seconds(exc: BaseException) -> float:
    """Wait after a 429: max(API retryDelay, 20s) + 5s safety margin."""
    reported = _walk_retry_delay(getattr(exc, "details", None))
    if reported is None:
        reported = _walk_retry_delay(getattr(exc, "message", None))
    if reported is None:
        reported = _walk_retry_delay(str(exc))
    required = GEMINI_FREE_TIER_MIN_INTERVAL_SECONDS
    if reported is not None and reported > required:
        required = reported
    delay = required + GEMINI_RATE_LIMIT_SAFETY_MARGIN_SECONDS
    return min(delay, GEMINI_RATE_LIMIT_MAX_WAIT_SECONDS)


class GeminiLLM:
    """Gemini API backend. The API key is read from the environment only."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        env_name = getattr(self.cfg, "llm_api_key_env", None) or "GEMINI_API_KEY"
        self.api_key = os.environ.get(env_name) or os.environ.get("GOOGLE_API_KEY") or ""
        self._client = None
        self.rate_limit_retries = 0
        self.rate_limit_waits = 0
        self.rate_limit_wait_seconds = 0.0

    def _client_or_raise(self):
        if not self.api_key:
            env_name = getattr(self.cfg, "llm_api_key_env", None) or "GEMINI_API_KEY"
            raise RuntimeError(
                f"Missing Gemini API key. Set the {env_name} environment variable."
            )
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _extract_text(self, response) -> str:
        try:
            text = response.text
            if text:
                return text.strip()
        except Exception:
            pass
        candidates = getattr(response, "candidates", None) or []
        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                value = getattr(part, "text", None)
                if value:
                    parts.append(value)
        return "\n".join(parts).strip()

    def _usage(self, response) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0, 0
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
        candidates = int(getattr(usage, "candidates_token_count", 0) or 0)
        total = getattr(usage, "total_token_count", None)
        output_tokens = candidates + thoughts
        if total is not None and input_tokens:
            output_tokens = max(output_tokens, int(total) - input_tokens)
        return input_tokens, output_tokens

    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        time.sleep(seconds)
        self.rate_limit_waits += 1
        self.rate_limit_wait_seconds += seconds

    def _pace(self) -> None:
        """Wait until 20s have passed since the last generate_content in this process."""
        global _LAST_GENERATE_CONTENT_MONO
        with _PACE_LOCK:
            last = _LAST_GENERATE_CONTENT_MONO
            if last is None:
                return
            remaining = GEMINI_FREE_TIER_MIN_INTERVAL_SECONDS - (time.monotonic() - last)
            if remaining > 0:
                print(
                    f"Gemini free-tier pacing: waiting {remaining:.1f}s "
                    f"(minimum {GEMINI_FREE_TIER_MIN_INTERVAL_SECONDS:.0f}s between generate_content calls)"
                )
                self._sleep(remaining)

    def _mark_request(self) -> None:
        global _LAST_GENERATE_CONTENT_MONO
        with _PACE_LOCK:
            _LAST_GENERATE_CONTENT_MONO = time.monotonic()

    def generate(self, prompt: str) -> LLMOutput:
        from google.genai import types

        client = self._client_or_raise()
        gen_config = types.GenerateContentConfig(
            temperature=self.cfg.llm_temperature,
            max_output_tokens=self.cfg.llm_max_new_tokens,
            candidate_count=1,
            seed=0,
        )
        retries_before = self.rate_limit_retries
        waits_before = self.rate_limit_waits
        wait_seconds_before = self.rate_limit_wait_seconds
        attempt = 0
        response = None
        api_latency = 0.0
        while True:
            self._pace()
            self._mark_request()
            t0 = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=self.cfg.llm_model,
                    contents=prompt,
                    config=gen_config,
                )
                api_latency = time.perf_counter() - t0
                break
            except Exception as exc:
                if not is_gemini_rate_limit_error(exc):
                    raise
                attempt += 1
                self.rate_limit_retries += 1
                if attempt > GEMINI_RATE_LIMIT_MAX_RETRIES:
                    raise
                delay = gemini_retry_delay_seconds(exc)
                print(
                    f"Gemini 429 RESOURCE_EXHAUSTED; retrying the same request after {delay:.1f}s "
                    f"(attempt {attempt}/{GEMINI_RATE_LIMIT_MAX_RETRIES})"
                )
                self._sleep(delay)
                continue
        input_tokens, output_tokens = self._usage(response)
        return LLMOutput(
            text=self._extract_text(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=api_latency,
            raw={
                "provider": "gemini",
                "model": self.cfg.llm_model,
                "finish_reason": str(
                    getattr((getattr(response, "candidates", None) or [None])[0], "finish_reason", "")
                    or ""
                ),
                "rate_limit_retries": self.rate_limit_retries - retries_before,
                "rate_limit_waits": self.rate_limit_waits - waits_before,
                "rate_limit_wait_seconds": round(
                    self.rate_limit_wait_seconds - wait_seconds_before, 4
                ),
            },
        )


def build_llm(cfg: Config | None = None):
    cfg = cfg or CONFIG
    if cfg.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(cfg)
    if cfg.llm_provider == "huggingface_local":
        return HuggingFaceLocalLLM(cfg)
    if cfg.llm_provider == "gemini":
        return GeminiLLM(cfg)
    raise ValueError(f"Unknown llm_provider: {cfg.llm_provider}")
