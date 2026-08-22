from config import CONFIG


def test_frozen_baseline_knobs_unchanged():
    assert CONFIG.chunk_size_tokens == 512
    assert CONFIG.chunk_overlap_tokens == 50
    assert CONFIG.embedding_model == "BAAI/bge-small-en-v1.5"
    assert CONFIG.dense_top_k == 20
    assert CONFIG.bm25_top_k == 20
    assert CONFIG.final_top_k == 5
    assert CONFIG.hybrid_dense_weight == 0.6
    assert CONFIG.hybrid_bm25_weight == 0.4
    assert CONFIG.hybrid_method == "weighted_minmax"
    assert CONFIG.llm_temperature == 0.0
    assert CONFIG.llm_provider == "gemini"
    assert CONFIG.llm_model == "gemini-3.6-flash"
    assert CONFIG.llm_api_key_env == "GEMINI_API_KEY"
