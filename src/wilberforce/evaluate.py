"""Comparativa de la PINN contra el baseline numerico de MATLAB."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import physics
from .checkpoints import cargar_checkpoint

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ruta_baseline = os.path.join(RAIZ, "data", "baseline")
ruta_figuras = os.path.join(RAIZ, "results", "figures")


def cargar_datos_matlab():
    """Carga la solucion de referencia RK4 exportada desde Simulink."""
    t_matlab = pd.read_csv(os.path.join(ruta_baseline, "datos_sim_tiempo_2.csv"),
                           header=None).values.flatten()
    z_matlab = pd.read_csv(os.path.join(ruta_baseline, "datos_sim_z_2.csv"),
                           header=None).values.flatten()
    theta_matlab = pd.read_csv(os.path.join(ruta_baseline, "datos_sim_theta_2.csv"),
                               header=None).values.flatten()
    return t_matlab, z_matlab, theta_matlab


def calc_errores(real, pred, nombre_var):
    mse = np.mean((real - pred) ** 2)
    mae = np.mean(np.abs(real - pred))

    norma_diff = np.linalg.norm(real - pred)
    norma_real = np.linalg.norm(real)
    err_rel = (norma_diff / norma_real) * 100 if norma_real != 0 else 0.0

    print(f"--- {nombre_var} ---")
    print(f"  MSE (Error Cuadratico Medio): {mse:.6e}")
    print(f"  MAE (Error Absoluto Medio):   {mae:.6e}")
    print(f"  Error Relativo (Norma L2):    {err_rel:.4f} %\n")

    return {"mse": float(mse), "mae": float(mae), "l2_rel_pct": float(err_rel)}


def evaluar_y_graficar_pinn(pinn, t_max_eval, semilla, etiqueta="PINN Clasica"):
    print("\n" + "=" * 70)
    print(f"COMPARATIVA REAL vs {etiqueta.upper()} | HORIZONTE: 0 a {t_max_eval} SEGUNDOS")
    print("=" * 70)

    cargar_checkpoint(pinn, t_max_eval, semilla)

    t_matlab, z_matlab, theta_matlab = cargar_datos_matlab()

    # 1. Filtrar los datos de MATLAB para la ventana actual
    mascara = (t_matlab >= 0.0) & (t_matlab <= float(t_max_eval))
    t_eval = t_matlab[mascara]
    z_real = z_matlab[mascara]
    theta_real = theta_matlab[mascara]

    # 2. Convertir el tiempo fisico al tiempo adimensional tau de la red
    tau_eval = torch.tensor(t_eval * physics.omega_z, dtype=torch.float32).view(-1, 1)

    # 3. Prediccion con la PINN
    pred_pinn = pinn.predict(tau_eval)
    z_pinn = pred_pinn[:, 0] * physics.Z0
    theta_pinn = pred_pinn[:, 1] * physics.Theta0

    # 4. Metricas de error
    print("\n" + "." * 25 + " METRICAS DE ERROR " + "." * 25)
    metricas_z = calc_errores(z_real, z_pinn, "Movimiento Translacional (z)")
    metricas_theta = calc_errores(theta_real, theta_pinn, "Movimiento Torsional (theta)")

    # 5. Graficas de comportamiento a lo largo del tiempo
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_eval, z_real, '--', color='darkorange', label='MATLAB (Real)',
             linewidth=2.5, zorder=2)
    plt.plot(t_eval, z_pinn, '-', color='dodgerblue', label=etiqueta,
             linewidth=1.5, alpha=0.8, zorder=1)
    plt.title(f'Movimiento Translacional $z(t)$ - [0 a {t_max_eval}s]',
              fontsize=12, fontweight='bold')
    plt.xlabel('Tiempo $t$ (s)')
    plt.ylabel('Desplazamiento $z$ (m)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(t_eval, theta_real, '--', color='darkorange', label='MATLAB (Real)',
             linewidth=2.5, zorder=2)
    plt.plot(t_eval, theta_pinn, '-', color='dodgerblue', label=etiqueta,
             linewidth=1.5, alpha=0.8, zorder=1)
    plt.title(f'Movimiento Torsional $\\theta(t)$ - [0 a {t_max_eval}s]',
              fontsize=12, fontweight='bold')
    plt.xlabel('Tiempo $t$ (s)')
    plt.ylabel('Angulo $\\theta$ (rad)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    os.makedirs(ruta_figuras, exist_ok=True)
    nombre = f"comparativa_semilla{semilla}_{t_max_eval}s.pdf"
    plt.savefig(os.path.join(ruta_figuras, nombre), format="pdf", bbox_inches="tight")
    plt.close()

    return {"t_max": t_max_eval, "z": metricas_z, "theta": metricas_theta}