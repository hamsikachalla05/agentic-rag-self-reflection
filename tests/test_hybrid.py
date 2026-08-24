from retrieval.hybrid_retriever import HybridRetriever


def test_weighted_minmax_combination():
    retriever = HybridRetriever.__new__(HybridRetriever)
    from config import CONFIG

    retriever.cfg = CONFIG
    dense = {"a": 10.0, "b": 0.0, "c": 5.0}
    bm25 = {"a": 1.0, "b": 5.0, "d": 3.0}
    scores = retriever._weighted_minmax_scores(dense, bm25)
    # a is best on dense, weak on bm25; b is worst dense, best bm25
    assert scores["a"] > 0
    assert scores["b"] > 0
    assert "c" in scores and "d" in scores
    # missing channel contributes 0 after min-max on the other list
    assert scores["c"] == CONFIG.hybrid_dense_weight * 0.5
    assert scores["d"] == CONFIG.hybrid_bm25_weight * ((3.0 - 1.0) / (5.0 - 1.0))


def test_rrf_scores_are_rank_based():
    from dataclasses import replace
    from config import CONFIG

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.cfg = replace(CONFIG, hybrid_method="rrf", rrf_k=60)
    dense = [("a", 0.9), ("b", 0.8)]
    bm25 = [("b", 12.0), ("c", 4.0)]
    scores = retriever._rrf_scores(dense, bm25)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]
