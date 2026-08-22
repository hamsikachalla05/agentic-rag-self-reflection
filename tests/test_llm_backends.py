from dataclasses import replace
import time

from config import CONFIG
from generation.llm_client import (
    GeminiLLM,
    HuggingFaceLocalLLM,
    LLMOutput,
    OpenAICompatibleLLM,
    build_llm,
    gemini_retry_delay_seconds,
    is_gemini_rate_limit_error,
)


def test_default_backend_is_local_ollama():
    llm = build_llm(CONFIG)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert CONFIG.llm_provider == "openai_compatible"
    assert CONFIG.llm_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert CONFIG.llm_base_url == "http://localhost:11434/v1"
    assert CONFIG.llm_temperature == 0.0
    assert CONFIG.llm_num_ctx == 2048
    assert CONFIG.llm_think is False
    assert llm.base_url == "http://localhost:11434/v1"
    assert llm.api_key == "ollama"


def test_ollama_request_targets_localhost_and_disables_thinking(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "94.0"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }

    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("requests.post", _post)
    llm = OpenAICompatibleLLM(CONFIG)
    prompt = "Question: net change?\nEvidence: 2014 5735; 2015 5829\nAnswer:"
    out = llm.generate(prompt)
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["num_ctx"] == 2048
    assert captured["json"]["options"]["num_batch"] == 1
    assert captured["json"]["messages"][1]["content"] == prompt
    assert "gold_answer" not in prompt
    assert "program_answer" not in captured["json"]["messages"][1]["content"]
    assert captured["headers"]["Authorization"] == "Bearer ollama"
    assert out.text == "94.0"
    assert out.raw["provider"] == "openai_compatible"
    assert out.raw["base_url"] == "http://localhost:11434/v1"


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
    assert captured["model"] == CONFIG.llm_model
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
    assert gemini_retry_delay_seconds(err) == 51.0  # max(46, 20) + 5s margin


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
    monkeypatch.setattr("generation.llm_client._LAST_GENERATE_CONTENT_MONO", None)
    llm = GeminiLLM(CONFIG)
    llm._client = _Client()
    out = llm.generate("Question: x\nEvidence: y\nAnswer:")
    assert calls["n"] == 2
    assert calls["contents"] == "Question: x\nEvidence: y\nAnswer:"
    assert out.text == "94.0"
    assert slept and slept[0] == 25.0  # max(1s retryDelay, 20s) + 5s margin
    assert out.raw["rate_limit_retries"] == 1
    assert out.latency_seconds >= 0


def test_gemini_pacing_is_global_across_clients(monkeypatch):
    slept = []
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr("generation.llm_client.time.sleep", lambda s: slept.append(s))
    monkeypatch.setattr("generation.llm_client._LAST_GENERATE_CONTENT_MONO", time.monotonic())
    first = GeminiLLM(CONFIG)
    second = GeminiLLM(CONFIG)
    first._pace()
    second._pace()
    assert slept
    assert slept[0] >= 19.0
    assert slept[1] >= 19.0

