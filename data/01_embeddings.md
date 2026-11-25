---
document_id: embeddings
category: fundamentos
tags: [embeddings, vectores, similitud]
---
# Embeddings y similitud vectorial

Un embedding representa un texto como un vector numérico. Un modelo de
embeddings asigna representaciones a documentos y consultas para comparar sus
significados. La similitud coseno mide la orientación relativa de dos vectores;
un valor mayor suele señalar mayor similitud, pero no es una probabilidad ni
garantiza que una afirmación sea verdadera.

OpenAI `text-embedding-3-small` genera vectores de 1536 dimensiones por
defecto. La dimensión del índice vectorial debe coincidir exactamente con la
dimensión de los embeddings que se insertan y consultan. Documentos y consultas
deben usar el mismo modelo, la misma configuración y el mismo espacio vectorial.
No se deben mezclar embeddings de distintos modelos en una misma comparación.

Los embeddings son útiles para encontrar paráfrasis y conceptos relacionados
aunque no coincidan las palabras exactas. En cambio, nombres propios, códigos,
versiones y siglas pueden beneficiarse de una recuperación léxica que busque
las palabras literalmente.
