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


def test_unit_and_scale_equivalents():
    assert is_answer_correct("$9.36 billion", "9362.2")
    assert is_answer_correct("9362 million", "9362.2")
    assert is_answer_correct("9,362,000,000", "9362.2")
    assert is_answer_correct("14%", "0.14464285714285713", "14%")
    assert is_answer_correct("3.6%", "3.6", "['+3.6%']")
    assert is_answer_correct("2019", "2019.0", "['2019']")


def test_calculation_answer_is_not_a_nearby_table_number():
    # Entergy net change is 94 (5829-5735), not an unrelated $5.4 million figure.
    assert is_answer_correct("94", "94.0", "94")
    assert is_answer_correct("$94 million", "94.0", "94")
    assert not is_answer_correct("$5.4 million", "94.0", "94")
    assert not is_answer_correct("2018", "2019.0", "['2019']")
    assert not is_answer_correct("-24.05", "24.05")


def test_yes_no_program_answers():
    assert is_answer_correct("yes", "1", "yes")
    assert is_answer_correct("no", "0", "no")
