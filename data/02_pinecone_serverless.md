---
document_id: pinecone-serverless
category: infraestructura
tags: [pinecone, serverless, namespaces, metadata]
---
# Índices Serverless de Pinecone

Pinecone Serverless almacena vectores en un índice gestionado, sin que la
aplicación administre nodos de base de datos. Para crear un índice denso se
declaran un nombre, una dimensión, una métrica de distancia y la región y
nube de despliegue. El índice de esta práctica utiliza métrica coseno y 1536
dimensiones, compatibles con el modelo `text-embedding-3-small` sin reducción
de dimensión.

Un registro vectorial puede guardar el texto original dentro de sus metadatos,
junto con campos filtrables o informativos. En esta práctica, cada registro
incluye `text`, `source`, `page`, `category`, `tags`, `document_id` y
`chunk_id`. Así se recupera el fragmento directamente desde el resultado
vectorial, sin consultar otra base de datos.

Los namespaces particionan los vectores dentro del mismo índice. Una aplicación
puede asignar namespaces separados por cliente, equipo o tipo de contenido y
limitar cada consulta al namespace autorizado. El namespace no reemplaza los
controles de acceso: el servicio debe decidir cuál puede consultar cada usuario.

El nombre del índice y las claves de API se configuran mediante variables de
entorno. El script de inicialización comprueba si el índice ya existe, valida
la dimensión y la métrica, y solo crea un índice Serverless si hace falta.
