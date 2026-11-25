# Pre-entrega 4: Recuperación híbrida con Pinecone

Pipeline de ingesta y recuperación híbrida para documentos técnicos. Los
chunks se indexan en un índice Pinecone Serverless y un recuperador local
BM25 se combina con la búsqueda vectorial por Reciprocal Rank Fusion (RRF).
El script `evaluate.py` calcula Precision@5 y Recall@5 contra un Golden Set.

## Requisitos

- Python 3.11 o superior
- Cuenta y clave API de Pinecone
- Clave API de OpenAI para `text-embedding-3-small`

## Preparación

En PowerShell, desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edita `.env` y configura al menos:

```dotenv
PINECONE_API_KEY=tu-clave-de-pinecone
OPENAI_API_KEY=tu-clave-de-openai
INDEX_NAME=rag-hibrido-demo
```

No agregues `.env` al repositorio: `.gitignore` ya lo excluye. Ajusta
`PINECONE_CLOUD` y `PINECONE_REGION` si tu cuenta debe alojar el índice en otra
región. Los valores incluidos (`aws`, `us-east-1`) son opciones de ejemplo.

## Crear o comprobar el índice Serverless

```powershell
python index.py
```

El script comprueba si `INDEX_NAME` ya existe. Si no existe, crea un índice
denso Serverless con métrica `cosine` y dimensión `1536`, usando
`text-embedding-3-small`. Si ya existe, valida dimensión y métrica y falla con
un error explícito ante un mismatch. El script solo crea infraestructura al
ejecutarse con claves configuradas; no se necesita para ejecutar las pruebas
locales.

## Ingestar documentos

Coloca archivos Markdown, JSON o PDF con texto seleccionable dentro de `data/`
(también busca en subdirectorios) y ejecuta:

```powershell
python ingest.py
```

Los archivos de ejemplo del repositorio documentan embeddings, Pinecone
Serverless, recuperación híbrida, fragmentación y evaluación. El pipeline usa
`RecursiveCharacterTextSplitter` con un medidor de tokens (`cl100k_base`),
tamaño objetivo de 600 tokens y overlap de 80. Los metadatos de cada vector
incluyen `text` (el fragmento original), `source`, `page`, `category`, `tags`,
`document_id`, `chunk_id` y `token_count`. Así el resultado contiene el texto
sin una consulta a otra base de datos.

Markdown puede incluir metadatos YAML al inicio del archivo:

```yaml
---
document_id: guia-api
category: arquitectura
tags: [api, autenticacion]
---
Contenido del documento...
```

Los JSON pueden ser un objeto o una lista de objetos con `text`, `content` o
`body`, y opcionalmente `document_id`, `page`, `category` y `tags`. Para PDF se
extrae texto y número de página; los PDFs escaneados requieren OCR previo.
Los IDs de chunk son deterministas, por lo que las reingestas hacen upsert en
lugar de crear duplicados. Para esta práctica, mantén sincronizadas las fuentes
locales que alimentan BM25 con los documentos indexados en Pinecone.

## Consultar el recuperador híbrido

```powershell
python retriever.py "¿Qué dimensión tiene text-embedding-3-small?"
```

También se puede usar desde Python:

```python
from retriever import RAGSystem

rag = RAGSystem.from_environment()
documents = rag.retrieve("¿Cómo funciona la búsqueda BM25?")
for document in documents:
    print(document.metadata["source"], document.page_content)
```

`RAGSystem` combina BM25 local con el retriever de Pinecone usando pesos
iguales por defecto. El ensemble consulta hasta 20 candidatos por estrategia,
fusiona sus posiciones con RRF y devuelve como máximo cinco documentos fuente
distintos. El namespace por defecto es `preentrega4`; usa uno diferente por
entorno o partición de datos.

## Evaluación: Precision@5 y Recall@5

El archivo `benchmarks/golden_set.json` contiene cinco preguntas y el
`documento_id_esperado` para cada una. Ejecuta:

```powershell
python evaluate.py
```

El script imprime las métricas por pregunta en los logs y el promedio macro en
consola. Se evalúan IDs de documentos fuente únicos para evitar contar dos
chunks del mismo archivo como dos documentos relevantes. Precision@5 es hits
dividido por cinco; Recall@5 es hits dividido por la cantidad de documentos
relevantes esperados (uno por pregunta en este Golden Set). Los cinco pares
constituyen un smoke test reproducible; conviene ampliarlos con datos
representativos antes de tomar decisiones de producción.

## Tests locales sin credenciales ni llamadas cloud

```powershell
python -m pytest -q
```

Los tests validan la carga y los metadatos, la configuración de chunking, la
fusión BM25/semántica y las fórmulas de Precision@k y Recall@k, sin conectarse a
OpenAI ni a Pinecone.
