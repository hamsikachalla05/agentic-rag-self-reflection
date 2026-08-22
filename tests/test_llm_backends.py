from dataclasses import replace

from config import CONFIG
from generation.llm_client import GeminiLLM, HuggingFaceLocalLLM, LLMOutput, build_llm


def test_default_backend_is_gemini():
    llm = build_llm(CONFIG)
    assert isinstance(llm, GeminiLLM)
    assert CONFIG.llm_model == "gemini-2.5-flash"


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
    assert captured["model"] == "gemini-2.5-flash"
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
