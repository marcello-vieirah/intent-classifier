import json
from pathlib import Path
from typing import Any, Dict, List

from intent_classifier.models import ClassificationResult


class ResultLogger:
    """Gerenciador de logs estruturados em .jsonl e relatórios de métricas em .json."""

    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / "history.jsonl"
        self.metrics_file = self.log_dir / "batch_metrics.json"
        self.errors_file = self.log_dir / "errors_batch_metrics.json"

    def log_result(self, result: ClassificationResult) -> None:
        """Registra um único resultado de forma incremental no history.jsonl."""
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def save_batch_metrics(
        self, results: List[ClassificationResult]
    ) -> Dict[str, Any]:
        """Calcula o resumo consolidado de uma bateria e gera relatórios .json."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = (correct / total) if total > 0 else 0.0

        metrics = {
            "total_queries": total,
            "correct_predictions": correct,
            "accuracy": round(accuracy, 4),
            "accuracy_percentage": f"{round(accuracy * 100, 2)}%",
            "results": [r.to_dict() for r in results],
        }

        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Filtra apenas os erros (is_correct é False)
        error_results = [r.to_dict() for r in results if r.is_correct is False]
        errors_data = {
            "total_errors": len(error_results),
            "errors": error_results,
        }

        with open(self.errors_file, "w", encoding="utf-8") as f:
            json.dump(errors_data, f, ensure_ascii=False, indent=2)

        return metrics
