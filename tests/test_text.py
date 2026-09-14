from intent_classifier.text import extract_tokens, normalize_text


def test_normalize_text():
    raw = "  QuEro mUdAr de tUrNo!   "
    expected = "quero mudar de turno"
    assert normalize_text(raw) == expected


def test_normalize_text_removes_accents():
    raw = "Declaração de Matrícula & Atestado"
    expected = "declaracao de matricula atestado"
    assert normalize_text(raw) == expected


def test_extract_tokens():
    text = "trocar trocar de turno"
    tokens = extract_tokens(text)
    assert tokens == {"trocar", "de", "turno"}
