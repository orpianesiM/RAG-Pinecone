---
document_id: chunking-metadata
category: ingestion
tags: [chunking, metadata, pdf, markdown, json]
---
# Chunking y metadatos de ingesta

Los documentos se fragmentan con `RecursiveCharacterTextSplitter` usando un
medidor de tokens: el tamaño objetivo es 600 tokens y la superposición es 80.
El método recursivo intenta conservar párrafos y oraciones antes de dividir
un bloque. El solapamiento ayuda a mantener contexto en los límites de los
fragmentos. El tamaño real de un fragmento corto al final de una página puede
ser menor que el objetivo.

Cada chunk guarda su contenido junto con el nombre relativo del archivo, la
página, la categoría, etiquetas, el ID del documento, el ID del fragmento y el
conteo de tokens. Los archivos Markdown pueden especificar `document_id`,
`category` y `tags` en un encabezado YAML; los PDF conservan el número de
página; los JSON pueden incluir texto y metadatos en cada objeto. El lector
PDF extrae texto seleccionable y no aplica OCR a documentos escaneados.

El identificador de chunk se deriva del documento, la página, la posición y el
contenido. Así los registros mantienen IDs estables entre corridas cuando el
contenido no cambia. El texto se almacena en la metadata con la clave `text`,
que el adaptador `PineconeVectorStore` usa para reconstruir los documentos.
