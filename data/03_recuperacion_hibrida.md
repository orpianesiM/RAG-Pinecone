---
document_id: recuperacion-hibrida
category: retrieval
tags: [bm25, hybrid-search, reciprocal-rank-fusion, pinecone]
---
# Recuperación híbrida: BM25 y vectores

BM25 es una función de ranking léxico que puntúa documentos según las palabras
de la consulta, su frecuencia y la frecuencia de los términos en la colección.
Es especialmente útil para términos técnicos poco comunes, nombres propios,
identificadores y siglas. Un recuperador vectorial compara embeddings y puede
encontrar textos semánticamente relacionados aunque empleen vocabulario
diferente.

La búsqueda híbrida combina ambos tipos de evidencia. En esta implementación,
BM25 indexa localmente los mismos fragmentos que se enviaron a Pinecone; el
recuperador vectorial consulta el índice remoto y el namespace configurados.
LangChain `EnsembleRetriever` combina las listas mediante Reciprocal Rank
Fusion (RRF), una fusión basada en posiciones que no exige que BM25 y Pinecone
produzcan scores con escalas comparables.

Los pesos del ensemble controlan cuánto influye cada lista en el orden final.
El ejemplo comienza con pesos 0.5 para BM25 y 0.5 para búsqueda vectorial.
Los candidatos se limitan antes de fusionar y la aplicación devuelve hasta
cinco documentos fuente distintos. Una evaluación con preguntas reales ayuda
a elegir pesos y cantidad de candidatos adecuados al dominio.

Un recuperador BM25 requiere cargar el corpus local en memoria cada vez que se
inicia el proceso. Pinecone conserva el índice semántico en la nube. El equipo
de ingesta y el de recuperación deben usar los mismos archivos, reglas de
fragmentación, IDs estables, modelo de embeddings, índice y namespace.
