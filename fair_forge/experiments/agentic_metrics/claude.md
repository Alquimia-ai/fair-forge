# Experimentos: Métricas Agentic (pass@K, pass^K, Tool Correctness)

## Objetivo

Prototipar y validar las métricas para evaluación de agentes:
- **pass@K**: Evalúa si al menos una respuesta generada es correcta
- **pass^K**: Evalúa si todas las K respuestas generadas son correctas
- **Tool Correctness**: Evalúa si el agente utiliza las herramientas correctas en el contexto adecuado

## Decisiones de Diseño

### 1. Estructura de K Respuestas
- Múltiples Datasets con el mismo `qa_id` pero diferentes `assistant_id`
- Cada Dataset representa una respuesta diferente del agente a la misma query

### 2. Método de Evaluación de Corrección
- **Método**: Similarity score usando embeddings
- **Modelo**: SentenceTransformer (`all-MiniLM-L6-v2` o similar)
- **Métrica**: Cosine similarity
- **Threshold**: A determinar experimentalmente (típicamente 0.7-0.9)

### 3. Tool Correctness - Aspectos a Evaluar
1. ✅ **Selección correcta de tool**: ¿Eligió la herramienta correcta?
2. ✅ **Parámetros correctos**: ¿Los parámetros son apropiados?
3. ✅ **Secuencia/orden de tools**: ¿Se llamaron en el orden correcto?
4. ✅ **Uso del resultado**: ¿Se utilizó correctamente el resultado?

## Estructura de Datos

### Batch.agentic (comportamiento actual)
```python
{
    "tools_used": [
        {
            "tool_name": "calculator",
            "parameters": {"operation": "multiply", "a": 5, "b": 3},
            "result": 15,
            "step": 1
        },
        {
            "tool_name": "formatter",
            "parameters": {"value": 15, "format": "string"},
            "result": "15",
            "step": 2
        }
    ],
    "final_answer_uses_tools": True
}
```

### Batch.ground_truth_agentic (comportamiento esperado)
```python
{
    "expected_tools": [
        {
            "tool_name": "calculator",
            "parameters": {"operation": "multiply", "a": 5, "b": 3},
            "step": 1
        },
        {
            "tool_name": "formatter",
            "parameters": {"value": 15, "format": "string"},
            "step": 2
        }
    ],
    "tool_sequence_matters": True
}
```

## Experimentos a Realizar

### Experimento 1: Similarity Threshold Tuning
- [ ] Probar diferentes modelos de embeddings
- [ ] Evaluar similarity scores con respuestas correctas/incorrectas
- [ ] Determinar threshold óptimo
- [ ] Considerar normalización de respuestas (lowercase, punctuation)

### Experimento 2: Tool Selection Evaluation
- [ ] Comparar `tools_used[].tool_name` vs `expected_tools[].tool_name`
- [ ] Calcular accuracy: tools_correctos / total_tools
- [ ] Manejar tools extra o faltantes

### Experimento 3: Parameter Accuracy
- [ ] Comparar parámetros exactos vs similares
- [ ] Decidir: exact match o similarity score para valores
- [ ] Manejar parámetros opcionales

### Experimento 4: Sequence Correctness
- [ ] Evaluar orden de llamadas
- [ ] Decidir: orden estricto o parcial
- [ ] Calcular edit distance de secuencias

### Experimento 5: Result Utilization
- [ ] Verificar si `final_answer_uses_tools` es True
- [ ] Buscar evidencia de uso de resultados en la respuesta final
- [ ] Calcular score de utilización

## Métricas Finales

### pass@K
```
pass@K = 1 if any(similarity(response_i, ground_truth) >= threshold for i in range(K)) else 0
```

### pass^K
```
pass^K = 1 if all(similarity(response_i, ground_truth) >= threshold for i in range(K)) else 0
```

### Tool Correctness Score
```
tool_correctness = (
    w1 * tool_selection_score +
    w2 * parameter_accuracy +
    w3 * sequence_score +
    w4 * utilization_score
) / (w1 + w2 + w3 + w4)

# Pesos por defecto: w1=w2=w3=w4=0.25 (igual peso)
```

## Notas y Observaciones

### Fecha: [Pendiente]
- Comenzando experimentación...

---

## Preguntas Pendientes

1. ¿Qué hacer si K varía entre diferentes qa_ids?
2. ¿Normalizar respuestas antes de calcular similarity?
3. ¿Cómo manejar tools opcionales vs requeridos?
4. ¿Penalizar uso de tools extra no especificados?

## Referencias

- Fair-Forge Context metric: `/fair_forge/metrics/context.py`
- Fair-Forge BestOf metric: `/fair_forge/metrics/best_of.py`
- Toxicity clustering: `/fair_forge/metrics/toxicity.py`
- Schema base: `/fair_forge/schemas/common.py`
