import inspect

from agent.agent import BaselineAgent
from config import CONFIG
from tests.test_frozen_config import test_frozen_baseline_knobs_unchanged


def test_baseline_config_has_no_self_rag_flags():
    assert not hasattr(CONFIG, "retrieval_decision")
    assert not hasattr(CONFIG, "retrieval_relevance")
    assert not hasattr(CONFIG, "evidence_support")
    assert not hasattr(CONFIG, "answer_usefulness")
    test_frozen_baseline_knobs_unchanged()


def test_baseline_agent_answer_signature_unchanged():
    params = list(inspect.signature(BaselineAgent.answer).parameters)
    assert params == ["self", "question", "question_id"]
