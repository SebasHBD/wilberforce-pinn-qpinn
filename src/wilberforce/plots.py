"""Figuras de diagnostico y de prediccion."""

import os

import matplotlib.pyplot as plt

from . import physics


def graficar_diagnostico(loss_phy, loss_ic, loss_lambda, grad_phy, grad_ic,
                         ntk_eigvals, ntk_epochs, ruta_salida=None):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analisis de Entrenamiento PINN - Pendulo de Wilberforce', fontsize=16)

    # --- A. Evolucion de Perdidas ---
    axs[0, 0].plot(loss_phy, label='Loss Fisica', color='blue', alpha=0.8)
    axs[0, 0].plot(loss_ic, label='Loss IC', color='orange', alpha=0.8)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Historial de Loss')
    axs[0, 0].set_xlabel('Epocas (Adam + L-BFGS)')
    axs[0, 0].set_ylabel('Loss (Log)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    # --- B. Dinamica de Gradientes ---
    axs[0, 1].plot(grad_phy, label='Max Grad Fisica', color='purple')
    axs[0, 1].plot(grad_ic, label='Mean Grad IC', color='green')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Comportamiento de Gradientes (Fisica vs IC)')
    axs[0, 1].set_xlabel('Epocas (Adam)')
    axs[0, 1].set_ylabel('Magnitud del Gradiente (Log)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, ls="--", alpha=0.5)

    # --- C. Evolucion del Lambda Dinamico ---
    axs[1, 0].plot(loss_lambda, label='Lambda IC (Dinamico)', color='red')
    axs[1, 0].set_title('Evolucion del Hiperparametro Lambda')
    axs[1, 0].set_xlabel('Epocas (Adam + L-BFGS)')
    axs[1, 0].set_ylabel('Valor de Lambda')
    axs[1, 0].legend()
    axs[1, 0].grid(True, ls="--", alpha=0.5)

    # --- D. Sesgo Espectral (Autovalores del NTK) ---
    for i, epoch in enumerate(ntk_epochs):
        # Invertimos para graficar del mas dominante al menos dominante
        axs[1, 1].plot(ntk_eigvals[i][::-1], marker='o', label=f'Epoca {epoch}')

    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Espectro del NTK (Diagnostico de Sesgo Espectral)')
    axs[1, 1].set_xlabel('Indice del Autovalor')
    axs[1, 1].set_ylabel('Magnitud del Autovalor (Log)')
    axs[1, 1].legend(fontsize='small', ncol=2)
    axs[1, 1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def graficar_prediccion(model, tau_pred, ruta_salida=None):
    predicciones_adimensionales = model.predict(tau_pred)

    u_pred = predicciones_adimensionales[:, 0]
    v_pred = predicciones_adimensionales[:, 1]

    # Recuperamos las variables fisicas originales
    t_real = tau_pred.numpy().flatten() / physics.omega_z
    z_pred_real = u_pred * physics.Z0
    theta_pred_real = v_pred * physics.Theta0

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_real, z_pred_real, label='Prediccion $z$ (Longitudinal)', color='blue')
    plt.title("Movimiento Vertical")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Desplazamiento")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_real, theta_pred_real, label='Prediccion $\\theta$ (Torsional)', color='red')
    plt.title("Movimiento de Rotacion")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Angulo (rad)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=150)
    plt.close(fig)