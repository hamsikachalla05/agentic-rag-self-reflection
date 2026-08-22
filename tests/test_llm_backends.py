from dataclasses import replace

from config import CONFIG
from generation.llm_client import (
    GeminiLLM,
    HuggingFaceLocalLLM,
    LLMOutput,
    build_llm,
    gemini_retry_delay_seconds,
    is_gemini_rate_limit_error,
)


def test_default_backend_is_gemini():
    llm = build_llm(CONFIG)
    assert isinstance(llm, GeminiLLM)
    assert CONFIG.llm_model == "gemini-3.6-flash"


def test_flan_t5_backend_still_available():
    cfg = replace(
        CONFIG,
        llm_provider="huggingface_local",
        llm_model="google/flan-t5-base",
    )
    llm = build_llm(cfg)
    assert isinstance(llm, HuggingFaceLocalLLM)


def test_gemini_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    llm = GeminiLLM(CONFIG)
    assert llm.api_key == "test-not-a-real-key"


def test_gemini_generate_uses_prompt_only(monkeypatch):
    captured = {}

    class _Models:
        def generate_content(self, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["temperature"] = config.temperature

            class _Usage:
                prompt_token_count = 11
                candidates_token_count = 4
                thoughts_token_count = 0
                total_token_count = 15

            class _Candidate:
                finish_reason = "STOP"

            class _Response:
                text = "94.0"
                usage_metadata = _Usage()
                candidates = [_Candidate()]

            return _Response()

    class _Client:
        models = _Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    llm = GeminiLLM(CONFIG)
    llm._client = _Client()
    prompt = "Question: net change?\nEvidence: 2014 5735; 2015 5829\nAnswer:"
    out = llm.generate(prompt)
    assert isinstance(out, LLMOutput)
    assert captured["model"] == "gemini-3.6-flash"
    assert captured["contents"] == prompt
    assert captured["temperature"] == 0.0
    assert "gold_answer" not in captured["contents"]
    assert "program_answer" not in captured["contents"]
    assert out.text == "94.0"
    assert out.input_tokens == 11
    assert out.output_tokens == 4
    assert out.raw["provider"] == "gemini"


def test_llm_client_source_has_no_hardcoded_key():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "generation" / "llm_client.py").read_text(
        encoding="utf-8"
    )
    assert "AIza" not in source
    assert "GEMINI_API_KEY" in source


def test_gemini_rate_limit_error_and_retry_delay():
    class _Err(Exception):
        def __init__(self):
            super().__init__("429 RESOURCE_EXHAUSTED retryDelay: 46s")
            self.code = 429
            self.status = "RESOURCE_EXHAUSTED"
            self.details = {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{"retryDelay": "46s"}],
                }
            }
            self.message = "GenerateRequestsPerMinutePerProject-FreeTier"

    err = _Err()
    assert is_gemini_rate_limit_error(err)
    assert gemini_retry_delay_seconds(err) == 48.0  # 46s + 2s margin


def test_gemini_retries_same_prompt_after_429(monkeypatch):
    calls = {"n": 0}
    slept = []

    class _Err(Exception):
        def __init__(self):
            super().__init__("429 RESOURCE_EXHAUSTED retryDelay: 1s")
            self.code = 429
            self.status = "RESOURCE_EXHAUSTED"
            self.details = {"error": {"details": [{"retryDelay": "1s"}]}}
            self.message = "RESOURCE_EXHAUSTED"

    class _Models:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            calls["contents"] = contents
            if calls["n"] == 1:
                raise _Err()

            class _Usage:
                prompt_token_count = 3
                candidates_token_count = 2
                thoughts_token_count = 0
                total_token_count = 5

            class _Candidate:
                finish_reason = "STOP"

            class _Response:
                text = "94.0"
                usage_metadata = _Usage()
                candidates = [_Candidate()]

            return _Response()

    class _Client:
        models = _Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr("generation.llm_client.time.sleep", lambda s: slept.append(s))
    llm = GeminiLLM(CONFIG)
    llm._client = _Client()
    llm.min_request_interval_seconds = 0
    out = llm.generate("Question: x\nEvidence: y\nAnswer:")
    assert calls["n"] == 2
    assert calls["contents"] == "Question: x\nEvidence: y\nAnswer:"
    assert out.text == "94.0"
    assert slept and slept[0] >= 3.0  # 1s retryDelay + 2s margin
    assert out.raw["rate_limit_retries"] == 1
    assert out.latency_seconds >= 0

