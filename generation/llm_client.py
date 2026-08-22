"""LLM backends.

Default experimental generator: Gemini 2.5 Flash via the Gemini API.
Local Hugging Face (FLAN-T5) and OpenAI-compatible endpoints remain available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from config import CONFIG, Config


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


class GeminiLLM:
    """Gemini API backend. The API key is read from the environment only."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or CONFIG
        env_name = getattr(self.cfg, "llm_api_key_env", None) or "GEMINI_API_KEY"
        self.api_key = os.environ.get(env_name) or os.environ.get("GOOGLE_API_KEY") or ""
        self._client = None

    def _client_or_raise(self):
        if not self.api_key:
            env_name = getattr(self.cfg, "llm_api_key_env", None) or "GEMINI_API_KEY"
            raise RuntimeError(
                f"Missing Gemini API key. Set the {env_name} environment variable."
            )
        if self._client is None:
            from google import genai

            # Pass the key only when present. The SDK also reads GEMINI_API_KEY
            # / GOOGLE_API_KEY from the process environment.
            if self.api_key:
                self._client = genai.Client(api_key=self.api_key)
            else:
                self._client = genai.Client()
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

    def generate(self, prompt: str) -> LLMOutput:
        import time

        from google.genai import types

        client = self._client_or_raise()
        gen_config = types.GenerateContentConfig(
            temperature=self.cfg.llm_temperature,
            max_output_tokens=self.cfg.llm_max_new_tokens,
            candidate_count=1,
            seed=0,
        )
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=self.cfg.llm_model,
            contents=prompt,
            config=gen_config,
        )
        latency = time.perf_counter() - t0
        input_tokens, output_tokens = self._usage(response)
        return LLMOutput(
            text=self._extract_text(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            raw={
                "provider": "gemini",
                "model": self.cfg.llm_model,
                "finish_reason": str(
                    getattr((getattr(response, "candidates", None) or [None])[0], "finish_reason", "")
                    or ""
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
