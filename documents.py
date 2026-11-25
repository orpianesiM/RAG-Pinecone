import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_chunk_settings

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".md", ".json", ".pdf"}
TOKEN_ENCODING = "cl100k_base"


def _metadata_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _markdown_documents(path: Path, root: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    front_matter: dict[str, Any] = {}
    if text.startswith("---\n"):
        front_matter_text, separator, remainder = text[4:].partition("\n---")
        if separator:
            front_matter = yaml.safe_load(front_matter_text) or {}
            text = remainder.lstrip("\n")

    relative_source = path.relative_to(root).as_posix()
    return [
        Document(
            page_content=text.strip(),
            metadata={
                "source": relative_source,
                "page": 1,
                "document_id": str(front_matter.get("document_id", path.stem)),
                "category": str(front_matter.get("category", "general")),
                "tags": _metadata_list(front_matter.get("tags")),
            },
        )
    ]


def _json_documents(path: Path, root: Path) -> list[Document]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON inválido en {path}") from error

    records = payload if isinstance(payload, list) else [payload]
    documents: list[Document] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Cada elemento de {path.name} debe ser un objeto JSON.")
        content = record.get("text", record.get("content", record.get("body")))
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"El objeto {index} de {path.name} necesita text, content o body."
            )
        documents.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "source": path.relative_to(root).as_posix(),
                    "page": int(record.get("page", index + 1)),
                    "document_id": str(record.get("document_id", path.stem)),
                    "category": str(record.get("category", "general")),
                    "tags": _metadata_list(record.get("tags")),
                },
            )
        )
    return documents


def _pdf_documents(path: Path, root: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    result: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning("Se omite página PDF sin texto extraíble: %s (%d)", path.name, page_number)
            continue
        result.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.relative_to(root).as_posix(),
                    "page": page_number,
                    "document_id": path.stem,
                    "category": "pdf",
                    "tags": [],
                },
            )
        )
    return result


def load_documents(data_directory: str | Path) -> list[Document]:
    root = Path(data_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".md":
            documents.extend(_markdown_documents(path, root))
        elif path.suffix.lower() == ".json":
            documents.extend(_json_documents(path, root))
        else:
            documents.extend(_pdf_documents(path, root))

    documents = [document for document in documents if document.page_content.strip()]
    if not documents:
        raise ValueError(f"No se encontraron documentos con texto en {root}.")
    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    configured_size, configured_overlap = get_chunk_settings()
    size = configured_size if chunk_size is None else chunk_size
    overlap = configured_overlap if chunk_overlap is None else chunk_overlap
    if size < 500 or size > 800 or overlap < 0 or overlap >= size:
        raise ValueError("Usa chunks de 500-800 tokens y un overlap menor que el tamaño.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING,
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    tokenizer = tiktoken.get_encoding(TOKEN_ENCODING)

    for chunk in chunks:
        source = str(chunk.metadata["source"])
        document_id = str(chunk.metadata["document_id"])
        page = int(chunk.metadata["page"])
        start_index = int(chunk.metadata.get("start_index", 0))
        content = chunk.page_content
        chunk_id = hashlib.sha256(
            f"{document_id}:{page}:{start_index}:{content}".encode("utf-8")
        ).hexdigest()
        chunk.metadata.update(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": source,
                "page": page,
                "tags": _metadata_list(chunk.metadata.get("tags")),
                "token_count": len(tokenizer.encode(content)),
            }
        )
    return chunks


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w+\b", text.casefold())
