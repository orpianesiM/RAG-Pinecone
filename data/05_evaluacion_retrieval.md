---
document_id: evaluacion-retrieval
category: evaluation
tags: [precision-at-k, recall-at-k, golden-set, ranking]
---
# Evaluación de recuperadores

Un Golden Set contiene preguntas y los documentos que deberían responderlas.
Precision@k mide qué fracción de los primeros k resultados es relevante:
número de documentos relevantes recuperados dividido por k. Recall@k mide qué
fracción de los documentos relevantes esperados aparece entre esos k
resultados. Si cada pregunta tiene un solo documento esperado, Recall@5 vale
uno cuando aparece y cero cuando no aparece.

Este proyecto evalúa cinco preguntas, una por cada documento de ejemplo, y
usa `document_id` para reconocer la relevancia. Para evitar inflar los
resultados, se evalúan documentos fuente únicos: varios fragmentos del mismo
archivo cuentan como un solo resultado. Precision@5 usa cinco como denominador
aun si se recuperan menos de cinco documentos; así un sistema que retorna
pocos resultados no obtiene una precisión artificialmente alta.

Los promedios macro informan el comportamiento global del benchmark, pero no
sustituyen una evaluación representativa del uso real. Un Golden Set pequeño
sirve como smoke test reproducible. Para comparar cambios de pesos o modelos,
conviene aumentar la cantidad de preguntas, etiquetar más de un documento
relevante por consulta y analizar errores de recuperación.
