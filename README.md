# Machine Learning desde Primeros Principios en JAX para Software Devs: De Reglas Fijas a tu Primera Red Neuronal en 100 Líneas

- **Por qué es relevante:** La mayoría de cursos presentan Machine Learning como una caja negra de capas prefabricadas. Para un ingeniero de software que ya conoce `jax.numpy`, `@jax.jit`, `jax.grad` y `jax.vmap` ([jax-for-software-devs](https://github.com/ykro/jax-for-software-devs)), un modelo de Machine Learning se entiende mucho mejor como una función parametrizada cuyos argumentos `params` se ajustan automáticamente minimizando una `loss function` con derivadas exactas.
- **Para quién es:** Desarrolladores de software, ingenieros backend y arquitectos de sistemas que quieren construir intuición matemática e ingenieril sólida sobre regresión, `SGD`/`Momentum`/`Adam`, estabilidad numérica `LogSumExp`, redes `MLP` y regularización `weight decay` en JAX puro sin depender de PyTorch.
- **Quién debería verlo:** Equipos de ingeniería que necesitan reemplazar cadenas frágiles de reglas `if/else` por modelos aprendidos desde telemetría real, diagnosticar problemas de `overfitting` y preparar la base antes de diseñar arquitecturas modernas con Flax NNX.
- **Serie completa:** Segunda entrega de la Serie B dentro de [JAX & OpenXLA Field Guide](https://github.com/ykro/jax-openxla-field-guide).

---

## Arquitectura y Modelo Mental

### 1. De Heurísticas `if/else` Manuales a un `Training Loop` con Pesos Aprendidos

En ingeniería de software tradicional escribimos reglas fijas (`if cpu > 0.75: latency += 14.0`) y las combinamos con los datos de entrada para producir una respuesta. Cuando el sistema cambia, un humano tiene que adivinar nuevos umbrales. En Machine Learning invertimos el flujo, entregamos los datos de entrada junto con las respuestas históricas observadas, medimos la distancia cuadrática media (`MSE`) y dejamos que `jax.value_and_grad` descubra los pesos óptimos `w, b` dentro de un `training loop` compilado con `@jax.jit`.

![Comparación entre reglas fijas de software y el loop de aprendizaje en JAX](diagrams/01-rules-vs-learned-loop.svg)

### 2. Dataflow Forward y Backward de una Red Neuronal (`MLP` de 2 Capas)

Apilar dos multiplicaciones de matrices sin una activación intermedia colapsa algebraicamente en una sola transformación lineal `(X @ W1) @ W2 == X @ (W1 @ W2)`. Al intercalar una no-linealidad como `jax.nn.gelu` y calcular la pérdida con el truco numérico `LogSumExp`, una red neuronal de 2 capas en 80 líneas de JAX deforma el espacio de representación y resuelve fronteras de decisión curvas con `99.47%` de exactitud.

![Dataflow forward y backward de un MLP de dos capas en JAX](diagrams/02-mlp-forward-backward-dataflow.svg)

### 3. Diagnóstico de `Overfitting` con `Train/Val Split`, `Weight Decay` y `Validation Gate`

Evaluar un modelo sobre el mismo conjunto de datos con el que se calculan los gradientes equivale a probar software únicamente contra mocks estáticos. Cuando el modelo tiene más grados de libertad que señal real, memoriza el ruido del `train set` y dispara el error en producción. La combinación de un `Train/Val Split` estricto, penalización `L2 Weight Decay` sobre la norma de los pesos y un `Validation Gate` (`early stopping`) restaura la capacidad de generalización.

![Comparación entre memorización por overfitting y regularización con L2 Weight Decay](diagrams/03-overfitting-vs-regularization.svg)

---

## Resultados Reales de Verificación (`verify_ml_principles.py`)

Benchmarks deterministas ejecutados con Python `3.12.11`, JAX `0.12.0.1` y NumPy `2.5.4`:

| Experimento | Archivo | Baseline / Enfoque Ingenuo | Modelo Aprendido en JAX | Mejora / Verificación |
| :--- | :--- | :--- | :--- | :--- |
| **01. Reglas vs Pesos Aprendidos (`MSE`)** | `experiments/01_rules_vs_learned_weights.py` | `531.3102 ms²` (Heurística `if/else`) | `0.3949 ms²` (Regresión Lineal `@jit`) | **1345.28x menor error** (recupera `w` con error máx `0.1513`) |
| **02. Geometría `SGD` vs `Momentum` vs `Adam`** | `experiments/02_gradient_descent_geometry.py` | `SGD` `8.5645` (`180` steps) / `Momentum` `0.4526` | `Adam` `0.000001` (`79` steps a `<0.05`) | **761.17x speedup** (`@jit` + `lax.scan` vs eager `loop`) |
| **03. `LogSumExp` y Clasificación Multiclase** | `experiments/03_logistic_classification_cross_entropy.py` | `33.33%` (`exp(120)` desborda a `NaN`) | `99.94%` exactitud (`LogSumExp` finito) | **PASS** (`max_diff = 0.0` vs `jax.nn.log_softmax`) |
| **04. `MLP` de 2 Capas vs Colapso Lineal** | `experiments/04_two_layer_mlp_from_scratch.py` | `32.53%` (2 capas lineales sin activación) | `99.47%` (`MLP` con `GELU`, 64 neuronas) | **PASS** (colapso lineal verificado `Δ = 1.43e-06`) |
| **05. `Overfitting` vs `L2 Weight Decay`** | `experiments/05_overfitting_and_weight_decay.py` | `9.9728` Val MSE (`wd=0.0`, brecha `34.39x`) | `2.2289` Val MSE (`wd=0.35` + `Val Gate`) | **4.47x menor Val MSE** (`\|\|w\|\|₂` baja de `3.43` a `0.70`) |

Salida íntegra de terminal generada por `verify_ml_principles.py`:

```text
==============================================================================
THE OUTSIDE JUDGE — ML from First Principles in JAX Verification Report
==============================================================================
Runtime : Python 3.12.11 | JAX 0.12.0.1 | NumPy 2.5.4 | Backend: cpu
------------------------------------------------------------------------------
Experiment                         | Baseline / Naive | Learned / JAX ML | Gain / Check
------------------------------------------------------------------------------
01. Rules vs Learned Linear MSE    | 531.3102 MSE     | 0.3949 MSE       | 1345.28x lower
02. SGD vs Momentum vs Adam        | SGD 180 steps    | Adam 79 steps    | 761.17x @jit
03. LogSumExp & Cross-Entropy      | 33.33% (NaN exp) | 99.94% (Finite)  | PASS (<1e-7)
04. 2-Layer MLP vs Linear Stack    | 32.53% (Linear)  | 99.47% (GELU MLP) | PASS (>97%) 
05. Overfitting vs Weight Decay    | 9.9728 Val MSE   | 2.2289 Val MSE   | 4.47x lower 
------------------------------------------------------------------------------
Saved JSON report to: benchmarks/latest_results.json
ALL DETERMINISTIC CHECKS PASSED.
```

---

## Estructura del Repositorio

```text
.
├── benchmarks/
│   ├── latest_results.json                              # Métricas verificadas por verify_ml_principles.py
│   └── terminal_output.txt                              # Salida íntegra del Outside Judge
├── diagrams/
│   ├── 01-rules-vs-learned-loop.{html,svg,png}
│   ├── 02-mlp-forward-backward-dataflow.{html,svg,png}
│   └── 03-overfitting-vs-regularization.{html,svg,png}
├── experiments/
│   ├── 01_rules_vs_learned_weights.py                   # De reglas if/else manuales a regresión lineal con MSE
│   ├── 02_gradient_descent_geometry.py                  # SGD, Momentum y Adam en 30 líneas de PyTrees puros
│   ├── 03_logistic_classification_cross_entropy.py      # Desbordamiento float32 en Softmax, LogSumExp y Cross-Entropy
│   ├── 04_two_layer_mlp_from_scratch.py                 # Colapso lineal vs MLP de 2 capas con GELU e init de He
│   └── 05_overfitting_and_weight_decay.py               # Overfitting en 28 métricas ruidosas vs L2 Weight Decay
├── gcp/
│   └── run_on_demand_ml.sh                              # Runner on-demand con auto-teardown en Cloud TPU v5e y L4 GPU
├── tests/
│   └── test_ml_principles.py                            # Suite pytest determinista (5 tests)
├── Makefile
├── pyproject.toml
└── verify_ml_principles.py                              # El Outside Judge (aserciones de convergencia y generalización)
```

---

## Onboarding Paso a Paso

### 1. Instalación desde cero (macOS / Linux)

```bash
git clone https://github.com/ykro/ml-from-first-principles-in-jax.git
cd ml-from-first-principles-in-jax
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Recorrido guiado por los 5 experimentos (Orden recomendado)

Cada script está diseñado como una lección autocontenida que se ejecuta en menos de un segundo en CPU:

```bash
# Paso 1 — De heurísticas if/else escritas a mano a pesos aprendidos con MSE y value_and_grad
python experiments/01_rules_vs_learned_weights.py

# Paso 2 — Geometría de optimización: SGD vs Momentum vs Adam implementados con jax.tree.map
python experiments/02_gradient_descent_geometry.py

# Paso 3 — De números a probabilidades: por qué exp() falla en float32 y cómo funciona LogSumExp
python experiments/03_logistic_classification_cross_entropy.py

# Paso 4 — Tu primera red neuronal (MLP de 2 capas) y por qué hacen falta activaciones no lineales
python experiments/04_two_layer_mlp_from_scratch.py

# Paso 5 — Cómo detectar y corregir overfitting con Train/Val Split, Weight Decay y Validation Gate
python experiments/05_overfitting_and_weight_decay.py
```

### 3. Verificación completa (Linter + Tests + Outside Judge)

```bash
# Ejecutar únicamente el Outside Judge y regenerar benchmarks/latest_results.json
python verify_ml_principles.py

# Ejecutar ruff, pytest (5 tests) y verify_ml_principles.py en un solo comando
make all
```

### 4. Ejecución on-demand con auto-teardown en Google Cloud (Cloud TPU v5e o NVIDIA L4 GPU)

El script `gcp/run_on_demand_ml.sh` aprovisiona recursos on-demand con auto-teardown en el proyecto `ai-experiments-487722` y garantiza la destrucción automática al finalizar mediante `trap cleanup EXIT INT TERM`.

```bash
# Ejecutar verificación en una instancia on-demand de Cloud TPU v5e-1 (us-central1-a)
make gcp-tpu

# Ejecutar verificación en un Vertex AI Custom Job on-demand con GPU NVIDIA L4
make gcp-gpu
```
