# Agentic Metrics - Experiments

Experimentos para evaluar agentes usando métricas pass@K y pass^K con LLM como juez.

## Problema Resuelto

**Problema original:** Usar embeddings + cosine similarity falla cuando:
- Ground truth es largo y detallado
- Respuesta del agente es corta pero correcta
- Resultado: Falsos negativos (respuestas correctas marcadas como incorrectas)

**Solución:** Usar un LLM (Groq) para evaluar semánticamente si la respuesta es correcta.

## Archivos

### Implementación Principal
- **`agentic_llm_simple.py`** - Métrica Agentic usando LLM directamente (RECOMENDADO ✅)
- **`agentic_llm.py`** - Versión con Judge (más compleja, no usar)
- **`agentic.py`** - Versión original con embeddings (problema de falsos negativos)

### Tests y Ejemplos
- **`test_simple.py`** - Test simple con mock LLM ✅
- **`example_groq.py`** - Ejemplo de uso con Groq API ✅
- **`groq_example.py`** - Ejemplo anterior (deprecado)

### Utilidades
- **`agentic_prompts.py`** - Prompts para evaluación
- **`agentic_judge_schemas.py`** - Schemas Pydantic
- **`utils.py`** - Funciones utilitarias (embedding-based, deprecado para este uso)
- **`similarity_fixes.py`** - Análisis del problema de embeddings

## Uso Recomendado

### 1. Instalar dependencias

```bash
pip install langchain-groq
export GROQ_API_KEY="your-api-key"
```

### 2. Crear tu Retriever

```python
from fair_forge.core import Retriever
from fair_forge.schemas import Dataset, Batch

class MyRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        # Retornar múltiples datasets con mismo qa_id
        # pero diferentes assistant_id (K respuestas)
        return [
            Dataset(
                session_id="session_1",
                assistant_id="agent_1",  # Primera respuesta
                conversation=[
                    Batch(
                        query="What is 5 * 3?",
                        assistant="15",
                        ground_truth_assistant="15",
                        qa_id="q1",  # Mismo qa_id
                    )
                ],
            ),
            Dataset(
                session_id="session_1",
                assistant_id="agent_2",  # Segunda respuesta
                conversation=[
                    Batch(
                        query="What is 5 * 3?",
                        assistant="The answer is 15",
                        ground_truth_assistant="15",
                        qa_id="q1",  # Mismo qa_id
                    )
                ],
            ),
        ]
```

### 3. Ejecutar evaluación

```python
from langchain_groq import ChatGroq
from agentic_llm_simple import AgenticLLMSimple

# Inicializar modelo Groq
model = ChatGroq(
    model="llama-3.3-70b-versatile",  # o "gpt-oss-120B"
    temperature=0,
    api_key="your-key"
)

# Ejecutar evaluación
results = AgenticLLMSimple.run(
    MyRetriever,
    model=model,
    correctness_threshold=0.7
)

# Ver resultados
for metric in results:
    print(f"QA: {metric.qa_id}")
    print(f"pass@{metric.k}: {metric.pass_at_k}")
    print(f"pass^{metric.k}: {metric.pass_pow_k}")
    print(f"Correct: {len(metric.correct_indices)}/{metric.k}")
```

## Métricas

### pass@K
Al menos una de K respuestas es correcta.

```
pass@K = True si existe al menos 1 respuesta correcta
```

### pass^K
Todas las K respuestas son correctas.

```
pass^K = True si todas las K respuestas son correctas
```

## Estructura de Datos

### Dataset con K respuestas

Para evaluar K respuestas de la misma pregunta:
- Mismo `qa_id` en todos los Datasets
- Diferente `assistant_id` (agent_1, agent_2, ..., agent_K)

```python
[
    Dataset(qa_id="q1", assistant_id="agent_1"),  # Respuesta 1
    Dataset(qa_id="q1", assistant_id="agent_2"),  # Respuesta 2
    Dataset(qa_id="q1", assistant_id="agent_3"),  # Respuesta 3
]
```

### Batch Fields

```python
Batch(
    query="Pregunta del usuario",
    assistant="Respuesta del agente",
    ground_truth_assistant="Respuesta esperada",
    qa_id="identificador_pregunta"
)
```

## Modelos Groq Disponibles

- **llama-3.3-70b-versatile** (recomendado) - Rápido y preciso
- **llama-3.1-70b-versatile** - Versión anterior
- **mixtral-8x7b-32768** - Bueno para contextos largos
- **gpt-oss-120B** - Si tienen acceso a este modelo custom

## Tests

```bash
# Test simple
uv run python test_simple.py

# Ejemplo con Groq (requiere API key)
export GROQ_API_KEY="your-key"
uv run python example_groq.py
```

## Próximos Pasos

- [ ] Implementar evaluación de tool correctness con LLM
- [ ] Mover a `fair_forge/metrics/` cuando esté validado
- [ ] Tests formales en `tests/metrics/`
- [ ] Documentación en Mintlify

## Comparación de Enfoques

| Enfoque | Pros | Contras |
|---------|------|---------|
| **Embeddings + Cosine Similarity** | Rápido, barato | Falsos negativos con textos asimétricos |
| **LLM Judge (Simple)** ✅ | Preciso, maneja asimetrías | Más lento, requiere API |
| **LLM Judge (con abstracción)** | Reutiliza infraestructura | Complejo, innecesario |

## Referencias

- Análisis del problema: `similarity_fixes.py`
- Memory notes: `fair_forge/experiments/agentic_metrics/CLAUDE.md`
- Groq API: https://console.groq.com/
