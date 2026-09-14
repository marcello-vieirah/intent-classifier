from intent_classifier.classifier import IntentClassifier
from intent_classifier.models import Intent


def get_mock_intents():
    return [
        Intent(
            id="mudanca_turno",
            description="",
            keywords=["turno", "horario"],
            examples=["Quero trocar o horario das minhas aulas"],
        ),
        Intent(
            id="trancamento_curso",
            description="",
            keywords=["trancar", "pausar"],
            examples=["Preciso trancar o curso neste semestre"],
        ),
    ]


def test_classifier_clear_match():
    """1. Pergunta claramente pertencente a uma intenção."""
    classifier = IntentClassifier(
        semantic_weight=0.75, lexical_weight=0.25, min_score_threshold=0.40
    )
    classifier.fit(get_mock_intents())

    result = classifier.classify("gostaria de trocar meu turno de aula")

    assert result.predicted_intent == "mudanca_turno"
    assert result.score >= 0.40
    assert len(result.candidates) == 2


def test_classifier_out_of_domain_unknown():
    """2. Pergunta completamente fora das intenções existentes."""
    classifier = IntentClassifier(
        semantic_weight=0.75, lexical_weight=0.25, min_score_threshold=0.40
    )
    classifier.fit(get_mock_intents())

    # Pergunta sobre receitas/culinária não tem relação com as intenções cadastradas
    result = classifier.classify("como fazer um bolo de cenoura?")

    assert result.predicted_intent == "unknown"
    assert result.score < 0.40
    # Candidatos ranqueados continuam sendo retornados normalmente
    assert len(result.candidates) == 2
    assert result.candidates[0].score == result.score


def test_classifier_borderline_threshold():
    """3. Pergunta ambígua/limítrofe ajustando o threshold."""
    intents = get_mock_intents()
    query = "como solicitar uma alteracao?"

    # Com threshold permissivo (0.20), deve classificar no candidato de maior score
    lenient_classifier = IntentClassifier(min_score_threshold=0.20)
    lenient_classifier.fit(intents)
    result_lenient = lenient_classifier.classify(query)

    # Com threshold rigoroso (0.90), deve se abster e retornar unknown
    strict_classifier = IntentClassifier(min_score_threshold=0.90)
    strict_classifier.fit(intents)
    result_strict = strict_classifier.classify(query)

    assert result_lenient.predicted_intent != "unknown"
    assert result_strict.predicted_intent == "unknown"
    # O score absoluto calculado deve ser exatamente o mesmo nos dois casos
    assert result_lenient.score == result_strict.score
