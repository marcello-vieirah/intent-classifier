from datetime import datetime, timezone
from typing import List, Optional
from intent_classifier.lexical import LexicalEngine
from intent_classifier.models import Candidate, ClassificationResult, Intent
from intent_classifier.semantic import SemanticEngine


class IntentClassifier:
    """
    Orquestrador híbrido que unifica pontuações semânticas e lexicais.
    """

    def __init__(
        self,
        semantic_weight: float = 0.75,
        lexical_weight: float = 0.25,
        min_score_threshold: float = 0.40,
        unknown_label: str = "unknown",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        if not (0.0 <= semantic_weight <= 1.0 and 0.0 <= lexical_weight <= 1.0):
            raise ValueError("Os pesos devem estar no intervalo [0.0, 1.0].")

        if not abs((semantic_weight + lexical_weight) - 1.0) < 1e-5:
            raise ValueError("A soma dos pesos deve ser exatamente 1.0.")

        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.min_score_threshold = min_score_threshold
        self.unknown_label = unknown_label
        self.model_name = model_name

        self.semantic_engine = SemanticEngine(model_name=self.model_name)
        self.lexical_engine = LexicalEngine()
        self.intents: List[Intent] = []

    def fit(self, intents: List[Intent]) -> None:
        """Carrega e indexa as intenções em ambos os motores."""
        self.intents = intents
        self.semantic_engine.fit(intents)
        self.lexical_engine.fit(intents)

    def classify(
        self, query: str, expected_intent: Optional[str] = None
    ) -> ClassificationResult:
        """Classifica uma pergunta e retorna os candidatos ordenados por score."""
        semantic_scores = self.semantic_engine.predict(query)
        lexical_scores = self.lexical_engine.predict(query)

        candidates: List[Candidate] = []

        for intent in self.intents:
            intent_id = intent.id
            sem_score = semantic_scores.get(intent_id, 0.0)
            lex_score = lexical_scores.get(intent_id, 0.0)

            # Combinação ponderada das notas
            final_score = (
                self.semantic_weight * sem_score
            ) + (self.lexical_weight * lex_score)

            candidates.append(
                Candidate(
                    intent_id=intent_id,
                    score=final_score,
                    semantic_score=sem_score,
                    lexical_score=lex_score,
                )
            )

        # Ordena candidatos do maior score para o menor
        candidates.sort(key=lambda c: c.score, reverse=True)

        top_score = candidates[0].score if candidates else 0.0

        # Regra de abstenção: se o maior score não atingir o threshold, classifica como unknown
        if candidates and top_score >= self.min_score_threshold:
            predicted = candidates[0].intent_id
        else:
            predicted = self.unknown_label

        is_correct = (
            (predicted == expected_intent)
            if expected_intent is not None
            else None
        )

        return ClassificationResult(
            query=query,
            predicted_intent=predicted,
            score=top_score,
            candidates=candidates,
            expected_intent=expected_intent,
            is_correct=is_correct,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config={
                "semantic_weight": self.semantic_weight,
                "lexical_weight": self.lexical_weight,
                "min_score_threshold": self.min_score_threshold,
                "model_name": self.model_name,
            },
        )
