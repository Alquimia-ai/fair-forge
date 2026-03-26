# 🚀 Quick Start Guide

## 5 minutos para empezar

### 1. Verifica instalación
```bash
cd /Users/frino/Desktop/Alquimia/fair-forge/fair_forge/guardians/experiments/evals/
source .venv/bin/activate

# Instala scikit-learn si aún no lo hiciste
pip install scikit-learn
```

### 2. Elige tu modelo favorito

**Modelos locales (más rápido para empezar):**
```bash
jupyter notebook 1_qwen_evaluation.ipynb
# O
jupyter notebook 2_pipeline_evaluation.ipynb
```

**Modelos API (requieren tokens):**
```bash
# Configura tokens primero
export HF_TOKEN="hf_..."           # Para Llama-Guard y Granite
export GROQ_API_KEY="gsk_..."      # Para GPT-OSS

jupyter notebook 3_gpt_oss_evaluation.ipynb
```

### 3. Ejecuta el notebook

**En Jupyter:**
- Cell → Run All

**En VS Code:**
- Run All Cells

**Desde terminal:**
```python
jupyter nbconvert --to notebook --execute 1_qwen_evaluation.ipynb
```

### 4. Revisa resultados

El notebook imprimirá:
```
✅ Dataset: 220 casos
   SAFE: 40 | UNSAFE: 180

📥 Cargando modelo...
✅ Modelo cargado

Evaluando 220 casos...
✅ Evaluación completada

📊 OVERALL METRICS:
   Accuracy:  XX.X%
   Precision: XX.X%
   Recall:    XX.X%
   F1 Score:  XX.X%

🌍 METRICS BY LANGUAGE:
   ...

📁 METRICS BY CATEGORY:
   ...
```

### 5. Compara todos (opcional)

Después de ejecutar 2+ notebooks:
```bash
jupyter notebook 6_final_comparison.ipynb
```

Verás tabla comparativa y recomendaciones.

## ⚡ Tip Pro

**No quieres esperar las descargas?**

Los modelos ya están en cache si ejecutaste `Tests.ipynb` antes:
```bash
ls ~/.cache/huggingface/hub/
```

Si ya están ahí, la carga es instantánea (solo a RAM).

## 🎯 Qué Esperar

**Primera ejecución:**
- Descarga modelos: 5-10 min (solo una vez)
- Evaluación 220 casos: 2-5 min por modelo

**Ejecuciones posteriores:**
- Carga modelo: 10-30 seg (desde cache)
- Evaluación 220 casos: 2-5 min

**Ventaja de notebooks separados:**
- Mantienes kernel vivo = 0 tiempo de recarga
- Modificas código, reejecutas celda = 10 segundos

## 📊 Archivos Generados

Cada notebook crea:
```
results_qwen.pkl         # DataFrame con resultados
results_pipeline.pkl
results_gpt_oss.pkl
results_llama_guard.pkl
results_granite.pkl
```

El notebook 6 los lee y genera:
```
EVALUATION_REPORT.md     # Reporte markdown para presentación
```

## ❓ Problemas Comunes

**"ModuleNotFoundError: sklearn"**
```bash
pip install scikit-learn
```

**"File not found: test_cases_expanded.json"**
```bash
# Verifica que estás en el directorio correcto
pwd
# Debería mostrar: .../evals/

# Verifica que el archivo existe
ls -lh data/test_cases_expanded.json
```

**"Model download too slow"**
```bash
# Usa mirror de HuggingFace
export HF_ENDPOINT="https://hf-mirror.com"
```

**"Out of memory"**
```python
# En el notebook, cambia:
device_map="auto"  →  device_map="cpu"
dtype=torch.bfloat16  →  dtype=torch.float32
```

## 🆘 Necesitas ayuda?

1. Lee `README_NOTEBOOKS.md` (más detallado)
2. Revisa `Tests.ipynb` (notebook original como referencia)
3. Verifica logs del notebook - errores aparecen en celdas

---

**Listo para empezar? → Abre `1_qwen_evaluation.ipynb` 🚀**
