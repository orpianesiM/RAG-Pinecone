from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from documents import load_documents, split_documents
from evaluate import evaluate, load_golden_set
from metrics import precision_recall_at_k
from retriever import RAGSystem


class StaticRetriever(BaseRetriever):
    results: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager):
        return self.results


def _document(document_id: str, chunk_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "source": f"{document_id}.md",
            "page": 1,
            "category": "test",
            "tags": ["test"],
        },
    )


def test_precision_recall_at_five_uses_fixed_k_and_unique_documents() -> None:
    result = precision_recall_at_k(
        ["expected", "expected", "other"],
        ["expected"],
        k=5,
    )

    assert result.precision_at_k == pytest.approx(0.2)
    assert result.recall_at_k == 1.0
    assert result.retrieved_count == 2


def test_precision_recall_rejects_empty_relevance_and_invalid_k() -> None:
    with pytest.raises(ValueError, match="al menos un documento"):
        precision_recall_at_k(["document"], [], k=5)
    with pytest.raises(ValueError, match="k debe"):
        precision_recall_at_k(["document"], ["document"], k=0)


def test_hybrid_retriever_fuses_results_and_returns_unique_sources() -> None:
    chunks = [
        _document("a", "a-1", "Pinecone index dimensions embeddings cosine."),
        _document("a", "a-2", "Pinecone Serverless index vector metadata."),
        _document("b", "b-1", "BM25 ranks exact technical terms lexically."),
    ]
    vector = StaticRetriever(results=[chunks[0], chunks[2]])
    system = RAGSystem(corpus_chunks=chunks, vector_retriever=vector, weights=(0.5, 0.5))

    results = system.retrieve("Pinecone index", k=5)

    result_ids = [document.metadata["document_id"] for document in results]
    assert result_ids[0] == "a"
    assert len(result_ids) == len(set(result_ids))
    assert set(result_ids) == {"a", "b"}


def test_loads_markdown_tags_and_token_chunks() -> None:
    data_directory = Path(__file__).resolve().parents[1] / "data"
    documents = load_documents(data_directory)
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)

    assert len(documents) == 5
    assert chunks
    assert {chunk.metadata["document_id"] for chunk in chunks} == {
        "embeddings",
        "pinecone-serverless",
        "recuperacion-hibrida",
        "chunking-metadata",
        "evaluacion-retrieval",
    }
    assert all(chunk.metadata["source"] for chunk in chunks)
    assert all(isinstance(chunk.metadata["page"], int) for chunk in chunks)
    assert all(isinstance(chunk.metadata["tags"], list) for chunk in chunks)
    assert all(chunk.metadata["category"] != "general" for chunk in chunks)
    assert all(len(chunk.metadata["chunk_id"]) == 64 for chunk in chunks)


def test_golden_set_evaluation_reports_macro_precision_and_recall() -> None:
    benchmark = load_golden_set()

    class GoldenSetRetriever:
        def retrieve(self, query: str, *, k: int) -> list[Document]:
            expected = next(
                item["documento_id_esperado"]
                for item in benchmark
                if item["pregunta"] == query
            )
            docs = [
                _document(str(expected), f"{expected}-1", "Documento relevante."),
                *[
                    _document(f"distractor-{index}", f"other-{index}", "Otro documento.")
                    for index in range(4)
                ],
            ]
            return docs[:k]

    result = evaluate(GoldenSetRetriever(), benchmark)

    assert result["precision_at_5"] == pytest.approx(0.2)
    assert result["recall_at_5"] == pytest.approx(1.0)
    assert result["preguntas_evaluadas"] == 5.0
