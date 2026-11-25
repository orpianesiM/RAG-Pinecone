import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Falta configurar la variable de entorno {name}.")
    return value


def get_index_name() -> str:
    value = _required("INDEX_NAME").lower()
    if not value or len(value) > 45:
        raise ValueError("INDEX_NAME debe tener entre 1 y 45 caracteres.")
    if not all(char.isalnum() or char == "-" for char in value):
        raise ValueError("INDEX_NAME solo puede contener letras, números y guiones.")
    return value


def get_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", "preentrega4").strip()


def get_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()


def get_dimension() -> int:
    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
    if dimension <= 0:
        raise ValueError("PINECONE_DIMENSION debe ser mayor que cero.")
    return dimension


def get_data_directory() -> Path:
    configured = Path(os.getenv("RAG_DATA_DIRECTORY", str(PROJECT_ROOT / "data")))
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


def get_chunk_settings() -> tuple[int, int]:
    size = int(os.getenv("RAG_CHUNK_SIZE", "600"))
    overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
    if size < 500 or size > 800:
        raise ValueError("RAG_CHUNK_SIZE debe estar entre 500 y 800 tokens.")
    if overlap < 0 or overlap >= size:
        raise ValueError("RAG_CHUNK_OVERLAP debe ser menor que el tamaño del chunk.")
    return size, overlap


def get_top_k() -> int:
    value = int(os.getenv("RAG_TOP_K", "5"))
    if value != 5:
        raise ValueError("La evaluación requerida utiliza top_k=5.")
    return value


def get_retriever_weights() -> tuple[float, float]:
    bm25_weight = float(os.getenv("RAG_BM25_WEIGHT", "0.5"))
    vector_weight = float(os.getenv("RAG_VECTOR_WEIGHT", "0.5"))
    if bm25_weight < 0 or vector_weight < 0 or bm25_weight + vector_weight <= 0:
        raise ValueError("Los pesos del recuperador deben ser no negativos y sumar > 0.")
    return bm25_weight, vector_weight
