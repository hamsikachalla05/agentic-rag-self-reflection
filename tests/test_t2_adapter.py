from data_loader.t2_ragbench_loader import normalize_row, resolve_hf_pdf_path


def test_normalize_finqa_row():
    row = {
        "id": "finqa_test_0",
        "context_id": "finqa_test_ctx_1",
        "split": "test",
        "question": "What was revenue?",
        "program_answer": "12.4",
        "original_answer": "12.40",
        "context": "Total revenues were 12.4 billion.",
        "file_name": "pdf/ADI/2009/page_49.pdf",
        "company_name": "Analog Devices",
        "report_year": 2009,
        "page_number": 49,
    }
    example = normalize_row("FinQA", row, split="test")
    record = example.to_question_record()
    assert record["question"] == "What was revenue?"
    assert record["gold_answer"] == "12.4"
    assert record["gold"]["document_id"] == "finqa_test_ctx_1"
    assert record["gold"]["evidence"].startswith("Total revenues")
    assert "gold" in record


def test_hf_pdf_path_resolution():
    files = {
        "data/FinQA/test/pdf/ETR/2016/page_23.pdf",
        "data/ConvFinQA/pdf/JKHY/2009/page_28.pdf",
        "data/TAT-DQA/test/raw/abc.pdf",
    }
    assert (
        resolve_hf_pdf_path("FinQA", "test", "pdf/ETR/2016/page_23.pdf", files)
        == "data/FinQA/test/pdf/ETR/2016/page_23.pdf"
    )
    assert (
        resolve_hf_pdf_path("ConvFinQA", "turn_0", "pdf/JKHY/2009/page_28.pdf", files)
        == "data/ConvFinQA/pdf/JKHY/2009/page_28.pdf"
    )
    assert (
        resolve_hf_pdf_path("TAT-DQA", "test", "raw/abc.pdf", files)
        == "data/TAT-DQA/test/raw/abc.pdf"
    )
    assert resolve_hf_pdf_path("FinQA", "test", "pdf/missing.pdf", files) is None
