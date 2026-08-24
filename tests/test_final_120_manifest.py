from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "final_120_questions.json"
DEV = ROOT / "data" / "t2_ragbench" / "questions" / "t2_small_15.json"


def test_final_120_manifest_frozen_shape():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    questions = payload["questions"]
    assert len(questions) == 120
    by_ds = {}
    ids = []
    for item in questions:
        by_ds[item["dataset"]] = by_ds.get(item["dataset"], 0) + 1
        ids.append(item["question_id"])
        assert item["question"].strip()
        assert item["gold_answer"] not in (None, "")
        assert item["gold"]["document_id"]
        assert item["gold"]["answer"] == item["gold_answer"]
        assert (item["gold"].get("evidence") or "").strip()
    assert by_ds == {"FinQA": 40, "ConvFinQA": 40, "TAT-DQA": 40}
    assert len(set(ids)) == 120
    assert payload["selection"]["used_model_outputs"] is False
    assert payload["selection"]["seed"] == 20260822


def test_final_120_excludes_development_15():
    final_ids = {
        q["question_id"]
        for q in json.loads(MANIFEST.read_text(encoding="utf-8"))["questions"]
    }
    dev_ids = {
        q["question_id"]
        for q in json.loads(DEV.read_text(encoding="utf-8"))["questions"]
    }
    assert len(dev_ids) == 15
    assert not (final_ids & dev_ids)
