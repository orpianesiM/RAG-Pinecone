import logging
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import (
    get_data_directory,
    get_dimension,
    get_embedding_model,
    get_namespace,
)
from documents import load_documents, split_documents
from index import ensure_index

load_dotenv()
logger = logging.getLogger(__name__)


def _openai_embeddings() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Configura OPENAI_API_KEY en el archivo .env.")
    return OpenAIEmbeddings(
        model=get_embedding_model(),
        dimensions=get_dimension(),
    )


def _pinecone_metadata(document: Document) -> dict[str, object]:
    metadata = dict(document.metadata)
    metadata["text"] = document.page_content
    metadata["tags"] = metadata.get("tags") or ["sin-etiquetas"]
    return metadata


def ingest_documents(
    *,
    data_directory=None,
    index=None,
    embeddings=None,
    namespace: str | None = None,
    batch_size: int = 64,
) -> int:
    source_directory = data_directory or get_data_directory()
    source_documents = load_documents(source_directory)
    chunks = split_documents(source_documents)
    if not chunks:
        raise ValueError("La ingesta no produjo fragmentos para indexar.")

    pinecone_index = index or ensure_index()
    embedding_model = embeddings or _openai_embeddings()
    target_namespace = namespace if namespace is not None else get_namespace()
    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=embedding_model,
        text_key="text",
        namespace=target_namespace,
    )

    inserted_ids: list[str] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [str(chunk.metadata["chunk_id"]) for chunk in batch]
        metadatas = [_pinecone_metadata(chunk) for chunk in batch]
        vector_store.add_texts(
            texts=[chunk.page_content for chunk in batch],
            metadatas=metadatas,
            ids=ids,
            namespace=target_namespace,
        )
        inserted_ids.extend(ids)

    logger.info(
        "Se indexaron %d chunks de %d documentos en %s (namespace=%s).",
        len(inserted_ids),
        len(source_documents),
        getattr(pinecone_index, "name", "Pinecone"),
        target_namespace or "(default)",
    )
    return len(inserted_ids)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    count = ingest_documents()
    print(f"Ingesta completa: {count} chunks.")
