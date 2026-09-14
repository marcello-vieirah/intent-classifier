import json
from pathlib import Path
from typing import List

from intent_classifier.models import BatchTestCase, Intent


def load_intents(file_path: str | Path) -> List[Intent]:
    """Carrega e valida o arquivo padronizado de intenções."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de intenções não encontrado em: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "O JSON de intenções deve conter uma lista de objetos."
        )

    intents: List[Intent] = []
    for item in data:
        for field in ["id", "description", "keywords", "examples"]:
            if field not in item:
                raise ValueError(
                    f"Intenção malformada. Campo obrigatório ausente: '{field}' em {item}"
                )

        intents.append(
            Intent(
                id=item["id"],
                description=item["description"],
                keywords=item["keywords"],
                examples=item["examples"],
            )
        )

    return intents


def load_batch_test(file_path: str | Path) -> List[BatchTestCase]:
    """Carrega os casos de teste da bateria de perguntas."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de bateria não encontrado em: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("O JSON da bateria deve conter uma lista de objetos.")

    cases: List[BatchTestCase] = []
    for item in data:
        if "query" not in item or "expected_intent" not in item:
            raise ValueError(
                f"Caso de teste malformado. Campos 'query' e 'expected_intent' são obrigatórios: {item}"
            )
        cases.append(
            BatchTestCase(
                query=item["query"], expected_intent=item["expected_intent"]
            )
        )

    return cases
