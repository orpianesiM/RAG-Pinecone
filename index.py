import logging
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from config import get_dimension, get_index_name

load_dotenv()
logger = logging.getLogger(__name__)


def pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Configura PINECONE_API_KEY en el archivo .env.")
    return Pinecone(api_key=api_key)


def ensure_index(
    *,
    client: Pinecone | None = None,
    index_name: str | None = None,
    dimension: int | None = None,
) -> object:
    pinecone = client or pinecone_client()
    name = index_name or get_index_name()
    expected_dimension = dimension or get_dimension()
    existing_names = {description.name for description in pinecone.list_indexes()}

    if name not in existing_names:
        cloud = os.getenv("PINECONE_CLOUD", "aws").strip().lower()
        region = os.getenv("PINECONE_REGION", "us-east-1").strip()
        logger.info(
            "Creando índice Serverless %s (%d dimensiones, %s/%s).",
            name,
            expected_dimension,
            cloud,
            region,
        )
        pinecone.create_index(
            name=name,
            dimension=expected_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        pinecone.describe_index(name)
    else:
        description = pinecone.describe_index(name)
        actual_dimension = getattr(description, "dimension", None)
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"El índice {name} tiene dimensión {actual_dimension}; "
                f"se requieren {expected_dimension}. Usa otro INDEX_NAME o recrea el índice."
            )
        actual_metric = getattr(description, "metric", None)
        if actual_metric and actual_metric.casefold() != "cosine":
            raise ValueError(
                f"El índice {name} usa métrica {actual_metric}; se requiere cosine."
            )

    return pinecone.Index(name)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_index()
    print(f"Índice Serverless listo: {get_index_name()}")
