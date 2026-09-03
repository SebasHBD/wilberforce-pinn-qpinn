"""Evalua la PINN entrenada contra el baseline de MATLAB en los tres horizontes.

Requiere haber corrido antes train_pinn.py con la misma semilla.

Uso:
    python scripts/evaluate_pinn.py
    python scripts/evaluate_pinn.py --semilla 7
"""

import argparse
import json
import os

import numpy as np
import torch

from wilberforce import physics
from wilberforce.evaluate import evaluar_y_graficar_pinn
from wilberforce.models import PINN

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HORIZONTES = [5, 10, 15]

parser = argparse.ArgumentParser()
parser.add_argument("--semilla", type=int, default=1)
args = parser.parse_args()

# Misma semilla que en el entrenamiento: aunque B_freq se sobrescribe desde
# el checkpoint, la instanciacion debe ser identica.
SEED = args.semilla
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Semilla fijada en SEED = {SEED}")

pinn = PINN(K=physics.K, A=physics.A, B=physics.B,
            u0=physics.u0, u0_dot=physics.u0_dot,
            v0=physics.v0, v0_dot=physics.v0_dot)

resultados = []
for t_max_eval in HORIZONTES:
    resultados.append(evaluar_y_graficar_pinn(pinn, t_max_eval, SEED))

salida = os.path.join(RAIZ, "results", "metrics", f"pinn_metricas_semilla{SEED}.json")
os.makedirs(os.path.dirname(salida), exist_ok=True)
with open(salida, "w", encoding="utf-8") as f:
    json.dump({"semilla": SEED, "modelo": "PINN clasica", "horizontes": resultados},
              f, indent=2, ensure_ascii=False)

print(f"\nMetricas guardadas en {salida}")