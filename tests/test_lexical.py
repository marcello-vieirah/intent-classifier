from intent_classifier.lexical import LexicalEngine
from intent_classifier.models import Intent


def test_lexical_engine_matching():
    intents = [
        Intent(
            id="mudanca_turno",
            description="",
            keywords=["trocar", "turno"],
            examples=[],
        ),
        Intent(
            id="trancamento",
            description="",
            keywords=["trancar", "curso"],
            examples=[],
        ),
    ]

    engine = LexicalEngine()
    engine.fit(intents)

    scores = engine.predict("quero trocar de turno por favor")

    assert scores["mudanca_turno"] > 0.0
    assert scores["trancamento"] == 0.0
