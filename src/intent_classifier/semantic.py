from typing import Dict, List
import numpy as np
from fastembed import TextEmbedding

from intent_classifier.models import Intent
from intent_classifier.text import normalize_text


class SemanticEngine:
    """Motor de similaridade semântica utilizando FastEmbed e Similaridade por Cosseno."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.model_name = model_name
        # Inicializa o modelo ONNX otimizado
        self.model = TextEmbedding(model_name=self.model_name)
        self.intent_embeddings: Dict[str, np.ndarray] = {}

    def fit(self, intents: List[Intent]) -> None:
        """
        Pré-calcula e armazena os embeddings para todos os exemplos de cada intenção.
        """
        self.intent_embeddings.clear()

        for intent in intents:
            if not intent.examples:
                continue

            # Normaliza os exemplos antes de gerar os embeddings
            normalized_examples = [
                normalize_text(ex) for ex in intent.examples
            ]

            # Converte os geradores do FastEmbed em array NumPy (shape: [N, embedding_dim])
            embeddings_list = list(self.model.embed(normalized_examples))
            self.intent_embeddings[intent.id] = np.array(embeddings_list)

    def _cosine_similarity(
        self, query_vector: np.ndarray, doc_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Calcula a similaridade por cosseno entre o vetor da pergunta
        e uma matriz de vetores de exemplos.
        """
        # Produto escalar (dot product)
        dot_product = np.dot(doc_vectors, query_vector)

        # Normas L2
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(doc_vectors, axis=1)

        # Prevenção contra divisão por zero
        norms = query_norm * doc_norms
        norms[norms == 0] = 1e-10

        return dot_product / norms

    def predict(self, query: str) -> Dict[str, float]:
        """
        Calcula a nota semântica (0.0 a 1.0) para cada intenção cadastrada.
        Retorna um dicionário {intent_id: score}.
        """
        normalized_query = normalize_text(query)
        if not normalized_query or not self.intent_embeddings:
            return {intent_id: 0.0 for intent_id in self.intent_embeddings}

        # Vetoriza a pergunta
        query_embedding = list(self.model.embed([normalized_query]))[0]

        scores: Dict[str, float] = {}
        for intent_id, example_vectors in self.intent_embeddings.items():
            sims = self._cosine_similarity(query_embedding, example_vectors)
            # A maior similaridade entre os exemplos define a nota semântica da intenção
            max_sim = float(np.max(sims))
            # Garante amplitude entre 0.0 e 1.0 (evitando ruídos negativos do cosseno)
            scores[intent_id] = max(0.0, max_sim)

        return scores
