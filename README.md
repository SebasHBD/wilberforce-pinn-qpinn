# Simulación del péndulo de Wilberforce con PINNs y Q-PINNs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![PennyLane](https://img.shields.io/badge/PennyLane-quantum-purple)
![License](https://img.shields.io/badge/License-MIT-green)

Código, datos y resultados del Trabajo Fin de Máster **_Simulación computacional
del péndulo de Wilberforce: un estudio comparativo entre métodos numéricos
tradicionales, PINNs y Q-PINNs_**.

Máster Universitario en Ingeniería Matemática y Computación
Universidad Internacional de La Rioja (UNIR), 2026
Autor: Sebastián [Apellido] · Tutor: Dagoberto Mayorca Torres

---

## Descripción

El péndulo de Wilberforce acopla una oscilación vertical `z(t)` con una torsión
`θ(t)` mediante una constante de acoplamiento `ε`. Cuando las frecuencias
naturales de ambos modos coinciden, la energía se transfiere periódicamente
entre ellos y aparece un batido característico: la oscilación vertical se
extingue mientras crece la torsional, y viceversa.

Ese comportamiento lo convierte en un banco de pruebas exigente para redes
informadas por la física. El sistema es de segundo orden, acoplado y
multiescala: la dinámica rápida tiene un periodo natural de unos 2.72 s,
mientras que la envolvente del batido opera en una escala diez veces mayor. Una
red debe capturar ambas simultáneamente, lo que expone directamente el sesgo
espectral que afecta a las PINNs. Al mismo tiempo, el sistema tiene solución
numérica de referencia bien establecida, de modo que el error puede medirse sin
ambigüedad.

Sobre ese banco de pruebas se compara una PINN clásica con una variante híbrida
cuántico-clásica, en la que un circuito variacional de 4 qubits sustituye al
núcleo de la red. La pregunta es si el circuito permite alcanzar una precisión
comparable con un número sustancialmente menor de parámetros entrenables.

Se comparan tres aproximaciones al mismo problema:

| Enfoque | Método |
|---|---|
| Numérico | RK4 de paso fijo, `h = 0.01 s` (Simulink) |
| Clásico | Physics-Informed Neural Network |
| Híbrido | Quantum-Classical PINN (PennyLane + PyTorch) |

---

## Resultados

Error relativo en norma L2 frente a la solución numérica de referencia, sobre
`t ∈ [0, 15] s`. Media ± desviación estándar sobre 3 semillas.

| Modelo | Parámetros | Error en `z(t)` | Error en `θ(t)` |
|:---|---:|---:|---:|
| PINN clásica | 115 842 | — | — |
| Q-PINN híbrida | 34 526 | — | — |

> **[COMPLETAR]** al terminar las corridas.

<!-- Al tener las figuras, descomenta:
<p align="center">
  <img src="results/figures/comparativa_semilla1_15s.png" width="800">
</p>
-->

---

## Instalación

Requiere Python 3.9 o superior.

```bash
git clone https://github.com/SebasHBD/wilberforce-pinn-qpinn.git
cd wilberforce-pinn-qpinn

python3 -m venv .wilberforce
source .wilberforce/bin/activate    # Windows: .wilberforce\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Verificación rápida de que el entorno funciona (corrida corta, sin valor
científico):

```bash
python scripts/train_pinn.py --prueba
```

---

## Reproducción de los resultados

Los scripts se ejecutan en este orden. El argumento `--semilla` selecciona la
corrida; los resultados de la memoria usan las semillas 1, 2 y 3.

| Paso | Comando |
|:---|:---|
| 1. Entrenar la PINN clásica | `python scripts/train_pinn.py --semilla 1` |
| 2. Evaluar contra el baseline | `python scripts/evaluate_pinn.py --semilla 1` |
| 3. Entrenar el Q-PINN | `python scripts/train_qpinn.py --semilla 1` |
| 4. Evaluar el Q-PINN | `python scripts/evaluate_qpinn.py --semilla 1` |

Las tres semillas en una sola orden:

```bash
for s in 1 2 3; do python scripts/train_pinn.py --semilla $s; done
```

**Salidas.** Los checkpoints se guardan en `results/checkpoints/`, las figuras
en `results/figures/` y las métricas en `results/metrics/` como archivos JSON.

---

## Formulación del problema

Adimensionalizando con `τ = ω_z · t`, el sistema acoplado se reduce a:

$$u'' + u + A\,v = 0$$

$$v'' + K\,v + B\,u = 0$$

Con los parámetros de Berg & Marshall (1990):

| Parámetro | Símbolo | Valor |
|:---|:---:|---:|
| Masa | `m` | 0.4905 kg |
| Constante elástica longitudinal | `k` | 2.627 N/m |
| Constante elástica torsional | `δ` | 7.443 × 10⁻⁴ N·m/rad |
| Momento de inercia | `I` | 1.39 × 10⁻⁴ kg·m² |
| Constante de acoplamiento | `ε` | 9.27 × 10⁻³ N |

Condiciones iniciales: `z₀ = 0.1 m`, `ż₀ = 0`, `θ₀ = 0`, `θ̇₀ = 0`.

Los coeficientes adimensionales resultantes son:

| Coeficiente | Valor |
|:---|---:|
| `K` | 0.99981 |
| `A` | 0.01764 |
| `B` | 0.62260 |

El coeficiente `K` es el cociente entre las frecuencias naturales torsional y
longitudinal, elevado al cuadrado. Su proximidad a la unidad indica que el
montaje está sintonizado en una condición de resonancia casi perfecta, que es
precisamente el régimen en el que se produce la transferencia completa de
energía entre los dos modos.

El periodo de batido asociado es de aproximadamente 25.9 s, mientras que el
horizonte de evaluación llega hasta 15 s. La comparación cubre por tanto algo
más de la mitad de un ciclo de transferencia: incluye la extinción del modo
longitudinal y el crecimiento del torsional, pero no el retorno al estado
inicial. Los errores reportados describen el régimen transitorio de la primera
transferencia, no el comportamiento de la solución a largo plazo.

---

## Arquitecturas

### PINN clásica — 115 842 parámetros

Embedding de Fourier (64 frecuencias, `σ = 0.5`) seguido de un MLP de 7 capas
ocultas de 128 neuronas, activación `tanh` e inicialización de Xavier.

### Q-PINN híbrida — 34 526 parámetros

| Bloque | Configuración |
|:---|:---|
| Embedding de Fourier | 64 frecuencias, `σ = 0.5` → 128 dimensiones |
| Encoder clásico | 128 → 64 → 128 → 4, salida sigmoide |
| Circuito variacional | 4 qubits, 2 × `StronglyEntanglingLayers` (PennyLane) |
| Decoder clásico | 4 → 128 → 128 → 2 |

Diferenciación del circuito por backpropagation (`diff_method="backprop"`) sobre
el simulador `default.qubit`.

### Entrenamiento

Curriculum progresivo sobre horizontes de 5, 10 y 15 s. En cada etapa, Adam
(`lr = 10⁻³`) seguido de refinamiento con L-BFGS. La pérdida física incorpora
ponderación causal (`ε_causal = 0.01`) y el peso de las condiciones iniciales se
ajusta por gradient annealing (`α = 0.01`) a partir de la época 500.

---


---

## Hardware y tiempos

Todas las corridas se ejecutaron en un MacBook Air M4, en CPU. La aceleración
por GPU no aporta beneficio a la escala de 4 qubits empleada.

| Modelo | Etapa | Tiempo |
|:---|:---:|---:|
| PINN | 0–5 s | — |
| PINN | 0–10 s | — |
| PINN | 0–15 s | — |
| Q-PINN | 0–5 s | — |
| Q-PINN | 0–10 s | — |
| Q-PINN | 0–15 s | — |

> **[COMPLETAR]** con los tiempos medidos.

---

## Limitaciones

> **[ESCRIBE TÚ]** Ver nota al final de esta sección.

---

## Referencias

1. Berg, R. E. y Marshall, T. S. (1990). Wilberforce pendulum oscillations and
   normal modes. *American Journal of Physics*, 58(1), 32–38.
2. Wang, S., Teng, Y. y Perdikaris, P. (2020). Understanding and mitigating
   gradient pathologies in physics-informed neural networks.
3. Wang, S., Wang, H. y Perdikaris, P. (2021). On the eigenvector bias of
   Fourier feature networks.
4. Wang, S. y Perdikaris, P. (2024). Respecting causality for training
   physics-informed neural networks.
5. Trahan, C., Loveland, M. y Dent, S. (2024). Quantum physics-informed neural
   networks.
6. Panichi, G., Corli, S. y Prati, E. (2025). Quantum physics-informed neural
   networks for multi-variable PDEs.

---

## Licencia

Distribuido bajo licencia MIT. Ver [`LICENSE`](LICENSE).
