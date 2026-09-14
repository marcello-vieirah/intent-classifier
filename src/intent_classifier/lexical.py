from typing import Dict, List
from intent_classifier.models import Intent
from intent_classifier.text import extract_tokens, normalize_text


class LexicalEngine:
    """Motor de correspondência baseada em keywords/palavras-chave."""

    def __init__(self):
        self.intent_keywords: Dict[str, set[str]] = {}

    def fit(self, intents: List[Intent]) -> None:
        """Extrai e normaliza as keywords de cada intenção."""
        self.intent_keywords.clear()
        for intent in intents:
            normalized_kw = {normalize_text(kw) for kw in intent.keywords}
            self.intent_keywords[intent.id] = normalized_kw

    def predict(self, query: str) -> Dict[str, float]:
        """
        Calcula a nota lexical (0.0 a 1.0) para cada intenção com base na
        taxa de correspondência de palavras-chave.
        """
        query_tokens = extract_tokens(query)
        scores: Dict[str, float] = {}

        if not query_tokens:
            return {intent_id: 0.0 for intent_id in self.intent_keywords}

        for intent_id, keywords in self.intent_keywords.items():
            if not keywords:
                scores[intent_id] = 0.0
                continue

            # Termos que bateram exatamente
            matches = query_tokens.intersection(keywords)

            if not matches:
                scores[intent_id] = 0.0
                continue

            # Métrica simples e direta:
            # Proporção de palavras-chave da intenção encontradas na pergunta.
            # O min() garante teto de 1.0 caso haja mais acertos que o esperado.
            overlap_score = len(matches) / len(keywords)
            scores[intent_id] = min(1.0, float(overlap_score))

        return scores
