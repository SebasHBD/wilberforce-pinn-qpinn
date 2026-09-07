# Simulación del péndulo de Wilberforce con PINNs y Q-PINNs

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45.1-purple)
![License](https://img.shields.io/badge/License-MIT-green)

Código, datos y resultados del Trabajo Fin de Máster **_Simulación computacional
del péndulo de Wilberforce: un estudio comparativo entre métodos numéricos
tradicionales, PINNs y Q-PINNs_**.

Máster Universitario en Ingeniería Matemática y Computación
Universidad Internacional de La Rioja (UNIR), septiembre de 2026
Autor: Sebastián Quiñones Arredondo · Tutor: Dagoberto Mayorca Torres

---

## Descripción

El péndulo de Wilberforce acopla una oscilación vertical `z(t)` con una torsión
`θ(t)` mediante una constante de acoplamiento `ε`. Cuando las frecuencias
naturales de ambos modos casi coinciden, la energía se transfiere de forma
periódica y prácticamente completa entre ellos: en la referencia numérica, la
amplitud longitudinal se anula alrededor de los 12.9 s y el ciclo de batido
completo ronda los 25.8 s.

Esa dinámica lo convierte en un banco de pruebas exigente para redes informadas
por la física. El sistema es multiescala: una oscilación portadora rápida cuya
amplitud está modulada por una envolvente un orden de magnitud más lenta. El
modelo debe capturar ambas simultáneamente y respetar la causalidad temporal, lo
que expone directamente las patologías de entrenamiento documentadas en la
literatura. Al ser lineal y disponer de solución numérica de referencia, todo
error es atribuible al método.

Sobre ese banco de pruebas se compara una PINN clásica con una variante híbrida
cuántico-clásica, en la que un circuito variacional de 4 qubits sustituye al
núcleo de la red. La hipótesis de partida era que la Q-PINN alcanzaría una
precisión comparable con un 70 % menos de parámetros entrenables. **Los
resultados no la confirman:** la PINN clásica obtiene menor error, resulta más
estable frente a la inicialización aleatoria y entrena en aproximadamente la
mitad de tiempo.

| Enfoque | Método |
|---|---|
| Numérico (referencia) | RK4 de paso fijo, `h = 0.01 s` (Simulink, MATLAB 2025b) |
| Clásico | Physics-Informed Neural Network |
| Híbrido | Quantum-Classical PINN (PennyLane + PyTorch) |

---

## Resultados

Error relativo en norma L2 frente a la referencia RK4, sobre `t ∈ [0, 15] s`,
con las semillas 1, 21 y 42.

| Modelo | Parámetros | Error en `z(t)` | Error en `θ(t)` | CV de `z` |
|:---|---:|---:|---:|---:|
| PINN clásica | 115 842 | 13.89 ± 3.11 % | 9.81 ± 3.60 % | 22 % |
| Q-PINN híbrida | 34 526 | 29.21 ± 14.13 % | 22.32 ± 12.22 % | 48 % |

### Detalle por semilla

| Modelo | Semilla | `z(t)` | `θ(t)` |
|:---|---:|---:|---:|
| PINN | 1 | 17.48 % | 13.92 % |
| PINN | **21** | **11.94 %** | **7.18 %** |
| PINN | 42 | 12.26 % | 8.34 % |
| Q-PINN | 1 | 24.65 % | 17.15 % |
| Q-PINN | 21 | 45.06 % | 36.28 % |
| Q-PINN | **42** | **17.94 %** | **13.53 %** |

El contraste más claro aparece en los extremos: la peor corrida de la PINN
(17.48 %) equivale a la mejor de la Q-PINN (17.94 %). El coeficiente de
variación —48 % frente al 22 %— confirma que el modelo híbrido es
considerablemente menos estable ante la inicialización aleatoria. Las métricas
MSE y MAE completas, para los tres horizontes, están en `results/metrics/`.

---

## Instalación

El entorno de referencia es Python 3.13. Las versiones están fijadas en
`requirements.txt` y son necesarias para reproducir los resultados numéricos.

| Componente | Versión |
|:---|:---|
| Python | 3.13.15 |
| PyTorch | 2.11.0 (CPU) |
| PennyLane | 0.45.1 |
| NumPy | 2.1.3 |

```bash
git clone https://github.com/SebasHBD/wilberforce-pinn-qpinn.git
cd wilberforce-pinn-qpinn

python3.13 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Verificación rápida de que el entorno funciona (corrida corta, sin valor
científico):

```bash
python scripts/train_pinn.py --prueba
python scripts/train_qpinn.py --prueba
```

---

## Reproducción de los resultados

Los scripts se ejecutan en este orden. El argumento `--semilla` selecciona la
corrida; los resultados reportados usan las semillas 1, 21 y 42.

| Paso | Comando |
|:---|:---|
| 1. Entrenar la PINN clásica | `python scripts/train_pinn.py --semilla 21` |
| 2. Evaluar contra el baseline | `python scripts/evaluate_pinn.py --semilla 21` |
| 3. Entrenar el Q-PINN | `python scripts/train_qpinn.py --semilla 42` |
| 4. Evaluar el Q-PINN | `python scripts/evaluate_qpinn.py --semilla 42` |

Las tres semillas en una sola orden:

```bash
for s in 1 21 42; do
  python scripts/train_pinn.py --semilla $s && python scripts/evaluate_pinn.py --semilla $s
done
```

**Salidas.** Los checkpoints se guardan en `results/checkpoints/` (uno por
modelo, semilla y horizonte), las figuras en `results/figures/` y las métricas en
`results/metrics/` como archivos JSON.

**Verificación de la migración.** El código de este repositorio procede de
notebooks de Google Colab. Se comprobó que el número de parámetros entrenables y
el valor de la función de pérdida en la época 0 coinciden con los del entorno
original.

---

## Formulación del problema

Partiendo del lagrangiano del sistema con el convenio de acoplamiento de
Berg & Marshall (factor 1/2), las ecuaciones de Euler-Lagrange dan:

$$m\ddot{z} + kz + \tfrac{1}{2}\epsilon\theta = 0, \qquad I\ddot{\theta} + \delta\theta + \tfrac{1}{2}\epsilon z = 0$$

Adimensionalizando con `τ = ω_z · t`, donde `ω_z = √(k/m)`, y normalizando las
salidas con `z = z₀u`, `θ = θ₀v`:

$$u'' + u + A\,v = 0$$

$$v'' + K\,v + B\,u = 0$$

Parámetros físicos, tomados de Berg & Marshall (1991):

| Parámetro | Símbolo | Valor |
|:---|:---:|---:|
| Masa | `m` | 0.4905 kg |
| Constante elástica | `k` | 2.627 N/m |
| Constante torsional | `δ` | 7.443 × 10⁻⁴ N·m/rad |
| Momento de inercia | `I` | 1.39 × 10⁻⁴ kg·m² |
| Constante de acoplamiento | `ε` | 9.27 × 10⁻³ N/rad |

Condiciones iniciales: `z₀ = 0.1 m`, `ż₀ = 0`, `θ₀ = 0`, `θ̇₀ = 0`. Escalas
características: `z₀ = 0.1 m`, `θ₀ = 1 rad`.

| Coeficiente | Expresión | Valor |
|:---|:---|---:|
| `K` | `ω_θ² / ω_z²` | 0.9998 |
| `A` | `ε θ₀ / (2 k z₀)` | 0.0176 |
| `B` | `m ε z₀ / (2 I k θ₀)` | 0.6226 |

`K` es el cociente de las frecuencias naturales al cuadrado. Su proximidad a la
unidad indica que el montaje está sintonizado en resonancia casi perfecta, que
es el régimen donde se produce la transferencia completa de energía entre modos.

El horizonte de evaluación llega hasta 15 s, frente a un periodo de batido de
unos 25.8 s. La comparación cubre por tanto algo más de la primera mitad del
ciclo: incluye la extinción del modo longitudinal en torno a los 12.9 s, pero no
el retorno al estado inicial.

---

## Arquitecturas

### PINN clásica — 115 842 parámetros

| Bloque | Entrada → Salida | Activación | Parámetros |
|:---|:---|:---|---:|
| Fourier embedding | 1 → 128 | — (fijo) | 0 |
| Capas ocultas 1–7 | 128 → 128 | tanh | 115 584 |
| Salida | 128 → 2 | lineal | 258 |

Inicialización normal de Xavier.

### Q-PINN híbrida — 34 526 parámetros

| Bloque | Entrada → Salida | Activación | Parámetros |
|:---|:---|:---|---:|
| Fourier embedding | 1 → 128 | — (fijo) | 0 |
| Codificador clásico | 128 → 64 → 128 → 4 | tanh, tanh, sigmoide | 17 092 |
| VQC | 4 → 4 | — | 24 |
| Decodificador clásico | 4 → 128 → 128 → 2 | tanh, tanh, lineal | 17 410 |

El circuito variacional emplea 4 qubits con codificación angular
(`AngleEmbedding`, rotación `RY`) y dos capas `StronglyEntanglingLayers`, con
lectura de los valores esperados `⟨Z⟩` de los cuatro qubits. Simulado en
`default.qubit` con diferenciación por backpropagation.

### Técnicas incorporadas

| Técnica | Configuración | Referencia |
|:---|:---|:---|
| Fourier feature embedding | 64 frecuencias, `σ = 0.5`, no entrenable | Wang et al. (2021) |
| Gradient annealing | `λ_IC` inicial 100, recorte [10, 200], `α = 0.01`, calentamiento 500 épocas | Wang et al. (2020) |
| Entrenamiento causal | `ε_c = 0.01`, activo solo durante Adam | Wang et al. (2024) |

### Curriculum de entrenamiento

| Horizonte | Puntos de colocación | Épocas Adam | LR | Pasos L-BFGS |
|---:|---:|---:|:---|---:|
| 0–5 s | 3 000 | 3 000 | 1 × 10⁻³ | 1 000 |
| 0–10 s | 5 000 | 4 000 | 1 × 10⁻³ | 1 000 |
| 0–15 s | 10 000 | 7 000 | 1 × 10⁻³ | 1 000 |

Durante la fase L-BFGS se congela `λ_IC` y se desactiva la ponderación causal:
el optimizador evalúa la pérdida varias veces por paso en su búsqueda de línea,
y una pérdida que cambia entre evaluaciones invalida la dirección de descenso.

Ambos modelos comparten técnicas, hiperparámetros, adimensionalización y
condiciones iniciales. La única diferencia es la arquitectura, lo que permite
atribuirle a ella las diferencias observadas.


---

## Hardware y tiempos

Las corridas reportadas se ejecutaron en el entorno de CPU de Google Colab
(Ubuntu 22.04, AMD EPYC 7B12, 12 GiB de RAM, sin GPU asignada). No se empleó
aceleración por GPU: con un circuito de 4 qubits y capas de 128 neuronas, el
coste de transferencia supera la ganancia por paralelización. El coste dominante
no es el tamaño de las operaciones sino el cálculo de segundas derivadas por
diferenciación automática y el número de retropropagaciones por época.

Tiempos de entrenamiento, en minutos:

| Modelo | Semilla | 0–5 s | 0–10 s | 0–15 s | Total |
|:---|---:|---:|---:|---:|---:|
| PINN | 1 | 34.16 | 84.89 | 304.08 | 7.1 h |
| PINN | 21 | 152.47 | 189.22 | 565.74 | 15.1 h |
| PINN | 42 | 59.81 | 186.65 | 496.34 | 12.4 h |
| Q-PINN | 1 | 105.71 | 340.43 | 825.19 | 21.2 h |
| Q-PINN | 21 | 103.56 | 340.64 | 926.81 | 22.9 h |
| Q-PINN | 42 | 136.96 | 283.25 | 677.14 | 18.3 h |

El Q-PINN entrena en aproximadamente el doble de tiempo que la PINN pese a tener
un 70 % menos de parámetros: la conversión entre PennyLane y PyTorch para
construir el grafo de segundas derivadas introduce un cuello de botella. La
variación entre semillas se debe a L-BFGS, cuyo número de evaluaciones por paso
depende de la región del espacio de parámetros.

---

## Limitaciones

**Simulación, no hardware cuántico.** El circuito se ejecuta en `default.qubit`,
un entorno ideal sin ruido, decoherencia ni errores de compuerta. Los resultados
podrían variar sobre hardware NISQ real.

**Sin ventaja cuántica.** Todo el entrenamiento —retropropagación y optimización
de parámetros— ocurre en hardware clásico. La arquitectura es híbrida, de modo
que no cabe afirmar ninguna ventaja cuántica a partir de estos resultados.

**Horizonte temporal.** La evaluación cubre hasta 15 s, frente a un periodo de
batido de 25.8 s. Se desconoce el comportamiento de los modelos en el segundo
semiperiodo. La ecuación no incorpora disipación, por lo que el sistema modelado
es ideal y conservativo.

**Referencia numérica.** El error se mide contra una simulación RK4, no contra
datos experimentales.

**Hiperparámetros compartidos.** Mantener la misma configuración en ambos
modelos hace válida la comparación, pero no garantiza que sea la óptima para
cada uno.

**Desbalance de escalas.** Con `θ₀ = 1 rad`, `u` oscila en [-1, 1] mientras `v`
lo hace en [-6, 6]. Como la pérdida física suma los residuos al cuadrado sin
ponderar, el residuo rotacional domina por un factor cercano a 36 y la
retropropagación se concentra en corregir `v`. Esto explica que el error
traslacional supere al rotacional en todas las semillas. La escala que
equilibraría ambos residuos es `θ₀ = z₀√(m/I) ≈ 5.94 rad`, con la que
`A = B ≈ 0.105`.

**Codificación del VQC.** La salida del codificador pasa por una sigmoide antes
de entrar al circuito, restringiendo los ángulos de rotación a [0, 1] y
recorriendo solo la mitad del ciclo trigonométrico. Además, la codificación se
aplica una sola vez y sobre la salida del codificador en lugar de sobre `τ`, lo
que limita el número de armónicos que el circuito puede generar —precisamente la
capacidad que motivó elegir esta arquitectura.

**Número de semillas.** Tres corridas permiten observar la dispersión, pero no
sostener afirmaciones estadísticas sólidas sobre la diferencia entre modelos.

---

## Referencias

1. Berg, R. E. y Marshall, T. S. (1991). Wilberforce pendulum oscillations and
   normal modes. *American Journal of Physics*, 59(1), 32–38.
2. Wang, S., Teng, Y. y Perdikaris, P. (2020). Understanding and mitigating
   gradient pathologies in physics-informed neural networks.
3. Wang, S., Wang, H. y Perdikaris, P. (2021). On the eigenvector bias of Fourier
   feature networks. *CMAME*, 384, 113938.
4. Wang, S., Sankaran, S. y Perdikaris, P. (2024). Respecting causality for
   training physics-informed neural networks. *CMAME*, 421, 116813.
5. Raissi, M., Perdikaris, P. y Karniadakis, G. E. (2019). Physics-informed
   neural networks. *Journal of Computational Physics*, 378, 686–707.
6. Trahan, C., Loveland, M. y Dent, S. (2024). Quantum physics-informed neural
   networks. *Entropy*, 26(8), 649.
7. Panichi, G., Corli, S. y Prati, E. (2025). Quantum physics-informed neural
   networks for multi-variable PDEs. *Physical Review Applied*, 25.
8. Bergholm, V. et al. (2018). PennyLane: automatic differentiation of hybrid
   quantum-classical computations. arXiv:1811.04968.

---

## Licencia

Distribuido bajo licencia MIT. Ver [`LICENSE`](LICENSE).