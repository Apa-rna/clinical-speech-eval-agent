from src.evaluation.metrics import calculate_wer, calculate_cer, exact_match, classify_result


def test_wer_perfect_match():
    assert calculate_wer("patient in NICU", "patient in NICU") == 0.0


def test_wer_complete_mismatch():
    wer = calculate_wer("patient in NICU", "NICO NICK YOU NEEKU")
    assert wer > 0.5


def test_wer_partial_error():
    wer = calculate_wer("patient transferred from NICU", "patient transferred from NICO")
    assert 0.0 < wer < 0.5


def test_cer_perfect_match():
    assert calculate_cer("NICU team performed rounds", "NICU team performed rounds") == 0.0


def test_cer_single_char_error():
    cer = calculate_cer("NICU", "NICO")
    assert 0.0 < cer <= 0.3


def test_exact_match_identical():
    assert exact_match("patient in NICU", "patient in NICU") is True


def test_exact_match_case_insensitive():
    assert exact_match("Patient in NICU", "patient in nicu") is True


def test_exact_match_punctuation():
    assert exact_match("NICU, team.", "nicu team") is True


def test_exact_match_different():
    assert exact_match("patient in NICU", "patient in NICO") is False


def test_classify_clean():
    assert classify_result(0.10) == "clean"


def test_classify_borderline():
    assert classify_result(0.35) == "borderline"


def test_classify_failed():
    assert classify_result(0.75) == "failed"


def test_classify_boundary_clean():
    assert classify_result(0.20) == "clean"


def test_classify_boundary_borderline():
    assert classify_result(0.50) == "borderline"


def test_classify_just_over_borderline():
    assert classify_result(0.51) == "failed"
