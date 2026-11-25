from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    relevant_retrieved: int
    retrieved_count: int


def precision_recall_at_k(
    retrieved_ids: Iterable[str],
    relevant_ids: Iterable[str],
    *,
    k: int = 5,
) -> RetrievalMetrics:
    if k <= 0:
        raise ValueError("k debe ser mayor que cero.")

    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Se necesita al menos un documento relevante esperado.")

    unique_retrieved: list[str] = []
    seen: set[str] = set()
    for identifier in retrieved_ids:
        if identifier not in seen:
            unique_retrieved.append(identifier)
            seen.add(identifier)
        if len(unique_retrieved) == k:
            break

    hits = len(set(unique_retrieved) & relevant)
    return RetrievalMetrics(
        precision_at_k=hits / k,
        recall_at_k=hits / len(relevant),
        relevant_retrieved=hits,
        retrieved_count=len(unique_retrieved),
    )
