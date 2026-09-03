"""Entrena la PINN clasica siguiendo el curriculum de 3 etapas.

Uso:
    python scripts/train_pinn.py
    python scripts/train_pinn.py --semilla 7
    python scripts/train_pinn.py --prueba
"""

import argparse
import os
import time

import numpy as np
import torch

from wilberforce import physics
from wilberforce.checkpoints import guardar_checkpoint
from wilberforce.models import PINN
from wilberforce.plots import graficar_diagnostico, graficar_prediccion

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# t_max, n_puntos, epocas_adam, epocas_lbfgs
ETAPAS = [
    (5.0, 3000, 3000, 1000),
    (10.0, 5000, 4000, 1000),
    (15.0, 10000, 7000, 1000),
]

parser = argparse.ArgumentParser()
parser.add_argument("--semilla", type=int, default=1)
parser.add_argument("--prueba", action="store_true",
                    help="Corrida corta de verificacion, sin valor cientifico")
args = parser.parse_args()

if args.prueba:
    # Solo para comprobar que el pipeline corre de extremo a extremo
    ETAPAS = [(5.0, 200, 10, 2)]

SEED = args.semilla
torch.manual_seed(SEED)
np.random.seed(SEED)

pinn = PINN(K=physics.K, A=physics.A, B=physics.B,
            u0=physics.u0, u0_dot=physics.u0_dot,
            v0=physics.v0, v0_dot=physics.v0_dot)

print(f"K={physics.K:.5f}  A={physics.A:.5f}  B={physics.B:.5f}")
print(f"Parametros entrenables: {pinn.contar_parametros():,}")

for t_max, n_puntos, epocas_adam, epocas_lbfgs in ETAPAS:
    print(f"\n=== Entrenamiento 0 a {t_max} segundos (semilla {SEED}) ===")

    tau_max = physics.omega_z * t_max
    tau_tensor = torch.linspace(0, tau_max, n_puntos).view(-1, 1).requires_grad_(True)

    start_time = time.time()
    loss_ic, loss_phy, loss_lambda, grad_ic, grad_phy, ntk_eigvals, ntk_epochs = pinn.train(
        tau=tau_tensor, lr_adam=1e-3, epochs_adam=epocas_adam, epochs_lbfgs=epocas_lbfgs
    )
    end_time = time.time()

    tiempo_total_segundos = end_time - start_time
    print("--- Fin del Entrenamiento ---")
    print(f"Tiempo total: {tiempo_total_segundos:.2f} segundos "
          f"({tiempo_total_segundos/60:.2f} minutos)")

    base = os.path.join(RAIZ, "results", "figures", f"semilla{SEED}_t{int(t_max)}")
    graficar_diagnostico(loss_phy, loss_ic, loss_lambda, grad_phy, grad_ic,
                         ntk_eigvals, ntk_epochs,
                         ruta_salida=base + "_diagnostico.png")

    tau_pred = torch.linspace(0, tau_max, 1000).view(-1, 1)
    graficar_prediccion(pinn, tau_pred, ruta_salida=base + "_prediccion.png")

    guardar_checkpoint(pinn, int(t_max), SEED)