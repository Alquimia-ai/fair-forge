# Guardrail Models Evaluation - Notebook Structure

## 📁 Files Overview

```
evals/
├── data/
│   ├── test_cases.json              # Original 45 test cases
│   └── test_cases_expanded.json     # 🆕 220 balanced test cases (EN/ES)
├── evaluation_metrics.py            # 🆕 Granular metrics (Precision/Recall/F1)
│
├── Tests.ipynb                      # 🔒 Original notebook (PRESERVED)
│
├── 1_qwen_evaluation.ipynb          # 🆕 Qwen3Guard-0.6B
├── 2_pipeline_evaluation.ipynb      # 🆕 Pipeline Multicapa (5 SLMs)
├── 3_gpt_oss_evaluation.ipynb       # 🆕 GPT-OSS-SAFEGUARD-20B
├── 4_llama_guard_evaluation.ipynb   # 🆕 Llama-Guard-3-8B
├── 5_granite_evaluation.ipynb       # 🆕 Granite-Guardian-3.1-2B
└── 6_final_comparison.ipynb         # 🆕 Compare all models
```

## 🎯 Why Separate Notebooks?

**Problema:** El notebook original cargaba todos los modelos, requiriendo reiniciar el kernel frecuentemente.

**Solución:** Un notebook por modelo:
- ✅ **No reiniciar kernel** - Cada notebook mantiene su modelo en memoria
- ✅ **Ejecutar en paralelo** - Múltiples evaluaciones simultáneas
- ✅ **Desarrollo iterativo** - Modifica/reejecuta solo el modelo que necesitas
- ✅ **Debugging fácil** - Aisla problemas por modelo

## 📊 Dataset Mejorado

### `test_cases_expanded.json` (220 casos)

**Balanceo:**
- 40 SAFE (18%) vs 180 UNSAFE (82%)
- Vs original: 3 SAFE (7%) vs 37 UNSAFE (93%)

**Idiomas:**
- 145 English (66%)
- 75 Español (34%)

**Por categoría (10+ casos cada una):**
- `benign`: 15 casos
- `contextual_safe`: 15 casos (¡NUEVO! - casos académicos/educativos)
- `jailbreak`: 14 casos
- `profanity`: 12 casos
- `abuse`: 12 casos
- `violence`: 15 casos
- `sexual`: 15 casos
- `social_bias`: 15 casos
- `pii`: 18 casos (con formatos LATAM: DNI, CUIL, RFC, CURP)
- `copyright`: 14 casos
- `self_harm`: 15 casos
- `illegal`: 15 casos
- `system_info`: 10 casos

## 🚀 Cómo Usar

### Opción 1: Evaluar un modelo específico

```bash
# Abrir notebook individual
jupyter notebook 1_qwen_evaluation.ipynb

# O con VS Code
code 1_qwen_evaluation.ipynb
```

**Ventaja:** El modelo permanece en memoria, puedes reejecutar celdas sin recargar.

### Opción 2: Evaluar todos (en paralelo)

```bash
# Terminal 1
jupyter notebook 1_qwen_evaluation.ipynb

# Terminal 2
jupyter notebook 2_pipeline_evaluation.ipynb

# Terminal 3
jupyter notebook 3_gpt_oss_evaluation.ipynb

# ...etc
```

**Ventaja:** Evalúa 5 modelos simultáneamente.

### Opción 3: Evaluación automática

```python
# Ejecutar todos los notebooks desde Python
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebooks = [
    '1_qwen_evaluation.ipynb',
    '2_pipeline_evaluation.ipynb',
    '3_gpt_oss_evaluation.ipynb',
    '4_llama_guard_evaluation.ipynb',
    '5_granite_evaluation.ipynb'
]

for nb_file in notebooks:
    with open(nb_file) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=3600)
    ep.preprocess(nb, {'metadata': {'path': '.'}})

    with open(nb_file, 'w') as f:
        nbformat.write(nb, f)

    print(f"✅ {nb_file} completed")
```

## 📈 Métricas Granulares

Cada notebook ahora reporta:

```
📊 OVERALL METRICS:
   Accuracy:  85.5%
   Precision: 91.2% (of predicted UNSAFE, how many are truly UNSAFE)
   Recall:    88.3% (of actual UNSAFE, how many were detected)
   F1 Score:  89.7%

🌍 METRICS BY LANGUAGE:
   ENGLISH (en): F1: 90.9%
   ESPAÑOL (es): F1: 86.5%
   Gap: 4.4% ⚠️

📁 METRICS BY CATEGORY:
   ✅ pii             F1: 95.0%
   ✅ jailbreak       F1: 93.3%
   ⚠️  copyright      F1: 72.4%
   ❌ illegal         F1: 54.5%  <- CRITICAL

🔍 CRITICAL FINDINGS:
   Categories with F1 < 70%: illegal
   Language gap: 4.4% (English performs better)
```

## 🏆 Comparación Final

Después de ejecutar todos los notebooks individuales:

```bash
jupyter notebook 6_final_comparison.ipynb
```

Esto genera:
- Tabla comparativa de todos los modelos
- Mejor modelo por categoría (🥇🥈🥉)
- Gap de idioma por modelo
- Debilidades críticas por modelo
- Recomendaciones finales
- Export a `EVALUATION_REPORT.md`

## 🔧 Configuración Necesaria

### Para modelos locales (Qwen, Pipeline):
```bash
pip install torch transformers detoxify accelerate scikit-learn
```

### Para modelos API (GPT-OSS, Llama-Guard, Granite):
```bash
# Groq API
export GROQ_API_KEY="gsk_..."

# HuggingFace Endpoints
export HF_TOKEN="hf_..."
```

## 🆘 Troubleshooting

### "Model not loading"
```python
# Verifica cache de HuggingFace
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/")
print(f"Cache: {cache_dir}")

# Si es primera vez, los modelos se descargan (lento)
# Segunda vez: carga desde cache (rápido)
```

### "results_*.pkl not found" en notebook 6
```bash
# Ejecuta primero los notebooks individuales
# Generan archivos .pkl que el notebook 6 necesita
```

### "Out of memory"
```python
# Modelos grandes requieren RAM/VRAM
# Opciones:
# 1. Usa solo API-based models (no local)
# 2. Evalúa en batches más pequeños
# 3. Usa CPU con dtype=torch.float32 (más lento)
```

## 📝 Notas

- **Tests.ipynb** está preservado intacto como respaldo
- Los modelos se cachean en `~/.cache/huggingface/` - no se re-descargan
- Cada notebook guarda resultados en `results_*.pkl` para comparación
- El notebook 6 puede ejecutarse independientemente si ya tienes los .pkl

## 🎓 Para Presentación

Si quieres presentar resultados:

1. Ejecuta notebooks individuales (generan .pkl)
2. Ejecuta `6_final_comparison.ipynb`
3. Usa el output del notebook 6 o `EVALUATION_REPORT.md`

## 🔄 Si te arrepientes

El notebook original `Tests.ipynb` sigue ahí, intacto.

```bash
# Volver al notebook original
jupyter notebook Tests.ipynb
```

---

**Creado:** 2026-01-21
**Dataset:** 220 casos balanceados (EN/ES)
**Métricas:** Precision, Recall, F1, por idioma y categoría
