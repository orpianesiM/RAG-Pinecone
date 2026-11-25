import json
import logging
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from config import PROJECT_ROOT, get_top_k
from metrics import precision_recall_at_k
from retriever import RAGSystem

load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger(__name__)
GOLDEN_SET_PATH = PROJECT_ROOT / "benchmarks" / "golden_set.json"


def load_golden_set(path: str | Path = GOLDEN_SET_PATH) -> list[dict[str, object]]:
    try:
        benchmark = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"No se pudo leer el Golden Set {path}") from error

    if not isinstance(benchmark, list) or not benchmark:
        raise ValueError("El Golden Set debe ser una lista no vacía.")
    for position, case in enumerate(benchmark, start=1):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("pregunta"), str)
            or not isinstance(case.get("documento_id_esperado"), str)
        ):
            raise ValueError(
                f"El caso {position} debe contener pregunta y documento_id_esperado."
            )
    return benchmark


def evaluate(system: RAGSystem, benchmark: list[dict[str, object]]) -> dict[str, float]:
    k = get_top_k()
    precisions: list[float] = []
    recalls: list[float] = []

    for case in benchmark:
        question = str(case["pregunta"])
        expected_id = str(case["documento_id_esperado"])
        documents = system.retrieve(question, k=k)
        retrieved_ids = [
            str(document.metadata.get("document_id", ""))
            for document in documents
        ]
        result = precision_recall_at_k(
            retrieved_ids,
            [expected_id],
            k=k,
        )
        precisions.append(result.precision_at_k)
        recalls.append(result.recall_at_k)
        logger.info(
            "P@%d=%.2f R@%d=%.2f esperado=%s recuperado=%s",
            k,
            result.precision_at_k,
            k,
            result.recall_at_k,
            expected_id,
            retrieved_ids,
        )

    return {
        f"precision_at_{k}": mean(precisions),
        f"recall_at_{k}": mean(recalls),
        "preguntas_evaluadas": float(len(benchmark)),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    benchmark = load_golden_set()
    system = RAGSystem.from_environment()
    results = evaluate(system, benchmark)
    k = get_top_k()
    print(f"Preguntas evaluadas: {int(results['preguntas_evaluadas'])}")
    print(f"Precision@{k}: {results[f'precision_at_{k}']:.2%}")
    print(f"Recall@{k}:    {results[f'recall_at_{k}']:.2%}")


if __name__ == "__main__":
    main()
