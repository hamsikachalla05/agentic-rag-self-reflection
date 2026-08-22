from evaluation.answer_metrics import is_answer_correct, parse_number


def test_numeric_and_percent_answers():
    assert is_answer_correct("$12.40 billion", "12.4")
    assert is_answer_correct("The rate was 93.5%.", "0.935", "93.5%")
    assert is_answer_correct("£414 million", "414")
    assert is_answer_correct("18,420 people", "18420")
    assert not is_answer_correct("insufficient evidence", "12.4")


def test_tat_list_original_answer():
    assert is_answer_correct("216.0", "216.0", "['$216.0']")
    assert parse_number("($66)") == -66.0
