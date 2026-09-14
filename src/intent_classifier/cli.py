import argparse
import sys
from pathlib import Path

from intent_classifier.classifier import IntentClassifier
from intent_classifier.loader import load_batch_test, load_intents
from intent_classifier.logger import ResultLogger


def run_interactive_mode(
    classifier: IntentClassifier, logger: ResultLogger
) -> None:
    """Executa o loop do modo interativo."""
    print("\n--- Modo Interativo (Pressione Ctrl+C ou digite 'sair' para encerrar) ---")

    while True:
        try:
            query = input("\nPergunta: ").strip()
            if not query:
                continue

            if query.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break

            result = classifier.classify(query)
            logger.log_result(result)

            if result.predicted_intent == classifier.unknown_label:
                print(
                    f"\nIntenção: {result.predicted_intent} (Abaixo do threshold mínimo de {classifier.min_score_threshold:.2f})"
                )
            else:
                print(f"\nIntenção: {result.predicted_intent}")

            print(f"Score do Top-1: {result.score:.2f}")
            print("\nCandidatos:")
            for idx, cand in enumerate(result.candidates, start=1):
                percentage = int(round(cand.score * 100))
                print(
                    f"{idx}. {cand.intent_id:<22} {cand.score:.2f} ({percentage}%)"
                )

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break


def run_batch_mode(
    classifier: IntentClassifier,
    logger: ResultLogger,
    batch_file_path: str | Path,
) -> None:
    """Executa a bateria de testes automatizados."""
    print(f"\nCarregando bateria de perguntas de: {batch_file_path}")
    test_cases = load_batch_test(batch_file_path)

    results = []
    print("\nExecutando classificação da bateria...")
    print("-" * 76)
    print(
        f"{'STATUS':<8} | {'PERGUNTA':<30} | {'ESPERADO':<15} | {'PREVISTO':<15} | {'SCORE':<7}"
    )
    print("-" * 76)

    for case in test_cases:
        result = classifier.classify(
            case.query, expected_intent=case.expected_intent
        )
        logger.log_result(result)
        results.append(result)

        status = "OK" if result.is_correct else "ERRO"
        print(
            f"{status:<8} | {case.query[:28]:<30} | {case.expected_intent[:14]:<15} | {result.predicted_intent[:14]:<15} | {result.score:.4f}"
        )

    metrics = logger.save_batch_metrics(results)

    print("-" * 76)
    print(
        f"\nBateria concluída: {metrics['correct_predictions']}/{metrics['total_queries']} acertos ({metrics['accuracy_percentage']})"
    )
    print(f"Relatório consolidado salvo em: {logger.metrics_file}")
    print(f"Relatório de erros salvo em: {logger.errors_file}")
    print(f"Histórico incremental salvo em: {logger.history_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classificador Híbrido de Intenções"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="Modo de execução: 'interactive' ou 'batch'",
    )
    parser.add_argument(
        "--intents",
        default="data/intents.json",
        help="Caminho do JSON com o cadastro de intenções",
    )
    parser.add_argument(
        "--batch-file",
        default="data/batch_test.json",
        help="Caminho do JSON com os testes da bateria",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.75,
        help="Peso do motor semântico (0.0 a 1.0)",
    )
    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=0.25,
        help="Peso do motor lexical (0.0 a 1.0)",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.40,
        help="Threshold mínimo de confiança para considerar uma intenção válida",
    )

    args = parser.parse_args()

    # Validação e carregamento das intenções
    try:
        intents = load_intents(args.intents)
    except Exception as e:
        print(f"Erro ao carregar arquivo de intenções: {e}", file=sys.stderr)
        sys.exit(1)

    print("Inicializando modelo e gerando embeddings em memória...")
    classifier = IntentClassifier(
        semantic_weight=args.semantic_weight,
        lexical_weight=args.lexical_weight,
        min_score_threshold=args.min_threshold,
    )
    classifier.fit(intents)
    logger = ResultLogger()

    if args.mode == "interactive":
        run_interactive_mode(classifier, logger)
    else:
        run_batch_mode(classifier, logger, args.batch_file)


if __name__ == "__main__":
    main()
