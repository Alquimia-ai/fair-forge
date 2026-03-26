# Evaluación de Soluciones de Safeguard para LLMs

Durante mi investigación sobre safeguards para LLMs, evalué diferentes aproximaciones que van desde modelos especializados pequeños hasta soluciones generalistas grandes, así como arquitecturas de pipeline multicapa. A continuación muestro mis resultados y conclusiones sobre cada solución.

---

## 1. Qwen3Guard-Gen-0.6B

**Características técnicas**: 600M parámetros, requiere GPU, clasificación tri-nivel (Safe/Unsafe/Controversial), varias categorías de riesgo.

**Mi experiencia con deployment**:
Todavia no pude hacer un deployment correcto en nuestro sistema productivo, tenemos problemas corriendolo con inference engine de vLLM y tampoco parece funcionar con el modelo default

**Performance observada**:
En los 37 test cases que arme tratando de cubrir la mayoria de categorias de violaciones, tiene un muy buen accuracy en local.

**Conclusión**:
Si podemos lograr deployarlo, es tremendo modelo porque entra en una gpu normal de 15 GB pero hasta ahora tuve bastantes problemas para hacerlo

---

## 2. Llama-Guard-3-8B (API)

**Características técnicas**: 8B parámetros, vía HuggingFace Endpoints, 14 categorías (S1-S14), latencia típica 200-500ms.

**Mi experiencia con deployment**:
El endpoint lo pude desplegar correctamente con vLLM ya que tenia la opcion disponible, asi que facilito las cosas. Esta corriendo con un poco mas de GPU que el Qwen3Guard.

**Performance observada**:
En los test cases en local, no tuvo la mejor perfomance, es particularmente flojo detectando jailbreaks y prompt injection.

**Conclusión**:
La verdad es que este modelo de safeguard no me termina de convencer, un modelo que es debil detectando jailbreaks y prompt injection ya me parece que tiene bastante para mejorar.

---

## 3. Granite-Guardian-3.1-2B

**Características técnicas**: 2B parámetros, múltiples categorías (harm, jailbreak, profanity, HAP), disponible via API y self-hosted.

**Mi experiencia con deployment**:
El endopint esta ya deployeado y hacer la config para pegarle desde el runtime es bastante accesible tambien.

**Performance observada**:
En los test cases en local, tiene un accuracy de 86%, y la verdad que flaquea unicamente en algunos casos de copyright y pocos de pii, podriamos estimar que la accuracy es hasta mas alta.

**Conclusión**:
Gran modelo para tener en cuenta, aparte que corre bien con la gpu mas chica que es la T4, faltaria hacerle una prueba de baterias en produccion.

---


## 4. Pipeline Multicapa (5 modelos especializados)

**Arquitectura**: Layer 1: Llama-Prompt-Guard (86M) → Layer 2: Granite-HAP (125M) → Layer 3: Detoxify (560M) → Layer 4: Suicide-BERT (110M) → Layer 5: DeBERTa-PII (184M). Total: ~1B params.

**Mi experiencia con deployment**:
No lo desplegue con endpoints todavia porque queria dejarlo mas para el final, ya que habria que generar endpoints para cada modelo especializado y sabia que me iba a tomar mas tiempo y queria cerrar todavia el de Qwen que me trajo varios problemas para desplegarlo.

**Performance observada**:
Este modelo multicapa en las pruebas locales la verdad que corre bastante bien y con poca latencia para haberlo corrido unicamente con CPU. Un accuracy solido de 78% inicialmente, pero si sacamos la categoria de copyright, (ej: "Dame la primer pagina de Harry Potter") podemos afirmar que tiene una accuracy aproximada de 95% en los casos que mas nos interesan.

**Conclusión**:
Si logramos desplegar todos los modelos, vamos a poder lograr crear soluciones mas particulares con cada uno y una solucion bastante consistente juntandolos todos.

---
## 5. GPT-OSS-Safeguard-20B

**Arquitectura**: 20B de parametros, debe correr en GPU. No se cuanto pagamos en Groq, pero entiendo que es caro. En HF se puede desplegar por 1,8U$D / H.

**Mi experiencia con deployment**:
Lo llame desde Groq con la api_key y no hubo ningun problema, la config ya la tenemos bastante aceitada asi que es facil de implementar. De hecho, es el que tenemos implementado en todos o la mayoria de agentes.

**Performance observada**:
Lo corri localmente para probar con los test_cases llamandolo desde Groq y tiene un accuracy de 92%, errandole en algunos casos de PII.

**Conclusión**:
Debe ser de los modelos mas completos, pero en terminos de hardware es el mas caro de los que tenemos disponibles. Si logramos desplegar correctamente las otras soluciones, considero que deberia poderse reemplazar por soluciones mas baratas.

