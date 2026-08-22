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
    assert CONFIG.llm_provider == "openai_compatible"
    assert CONFIG.llm_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert CONFIG.llm_base_url == "http://localhost:11434/v1"
    assert CONFIG.llm_num_ctx == 2048
    assert CONFIG.llm_think is False
