import logging
import os
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import (
    get_data_directory,
    get_dimension,
    get_embedding_model,
    get_namespace,
    get_retriever_weights,
    get_top_k,
)
from documents import load_documents, split_documents, tokenize_for_bm25
from index import ensure_index

logger = logging.getLogger(__name__)


class RAGSystem:
    """Combina resultados BM25 y Pinecone con Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        corpus_chunks: list[Document],
        vector_retriever: Any,
        weights: tuple[float, float] | None = None,
        bm25_candidate_k: int = 20,
    ) -> None:
        if not corpus_chunks:
            raise ValueError("El recuperador híbrido necesita chunks para BM25.")
        bm25 = BM25Retriever.from_documents(
            corpus_chunks,
            preprocess_func=tokenize_for_bm25,
            k=bm25_candidate_k,
        )
        bm25_weight, vector_weight = weights or get_retriever_weights()
        self.ensemble = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=[bm25_weight, vector_weight],
            id_key="chunk_id",
        )
        self.top_k = get_top_k()

    @classmethod
    def from_environment(cls) -> "RAGSystem":
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("Configura OPENAI_API_KEY en el archivo .env.")

        source_documents = load_documents(get_data_directory())
        chunks = split_documents(source_documents)
        index = ensure_index()
        embeddings = OpenAIEmbeddings(
            model=get_embedding_model(),
            dimensions=get_dimension(),
        )
        store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=get_namespace(),
        )
        vector_retriever = store.as_retriever(search_kwargs={"k": 20})
        return cls(corpus_chunks=chunks, vector_retriever=vector_retriever)

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
    ) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacía.")
        requested = self.top_k if k is None else k
        if requested <= 0:
            raise ValueError("k debe ser mayor que cero.")

        ranked_chunks = self.ensemble.invoke(query.strip())
        results: list[Document] = []
        seen_documents: set[str] = set()
        seen_chunks: set[str] = set()
        for document in ranked_chunks:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            document_id = str(document.metadata.get("document_id", chunk_id))
            if chunk_id and chunk_id in seen_chunks:
                continue
            if document_id in seen_documents:
                continue
            if chunk_id:
                seen_chunks.add(chunk_id)
            seen_documents.add(document_id)
            results.append(document)
            if len(results) >= requested:
                break

        logger.info(
            "Recuperación híbrida: %d documentos únicos (k=%d).",
            len(results),
            requested,
        )
        return results


if __name__ == "__main__":
    import logging
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Uso: python retriever.py "tu consulta"')
    system = RAGSystem.from_environment()
    for position, document in enumerate(system.retrieve(question), start=1):
        print(
            f"{position}. {document.metadata.get('document_id')} | "
            f"{document.metadata.get('source')} pág. {document.metadata.get('page')}"
        )
        print(document.page_content)
