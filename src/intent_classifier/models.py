from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Intent:
    """Representa a definição de uma intenção carregada do JSON."""

    id: str
    description: str
    keywords: List[str]
    examples: List[str]


@dataclass
class Candidate:
    """Representa um candidato ranqueado na classificação."""

    intent_id: str
    score: float
    semantic_score: float
    lexical_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "lexical_score": round(self.lexical_score, 4),
        }


@dataclass
class ClassificationResult:
    """Resultado da classificação de uma única pergunta."""

    query: str
    predicted_intent: str
    score: float
    candidates: List[Candidate]
    expected_intent: Optional[str] = None
    is_correct: Optional[bool] = None
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "timestamp": self.timestamp,
            "query": self.query,
            "predicted_intent": self.predicted_intent,
            "score": round(self.score, 4),
            "candidates": [c.to_dict() for c in self.candidates],
            "config": self.config,
        }
        if self.expected_intent is not None:
            result["expected_intent"] = self.expected_intent
            result["is_correct"] = self.is_correct
        return result


@dataclass
class BatchTestCase:
    """Caso de teste para a bateria automatizada."""

    query: str
    expected_intent: str
