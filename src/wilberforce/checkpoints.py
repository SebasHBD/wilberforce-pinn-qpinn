"""Guardado y carga de los pesos de la PINN por etapa del curriculum."""

import os

import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ruta_checkpoints = os.path.join(RAIZ, "results", "checkpoints")
os.makedirs(ruta_checkpoints, exist_ok=True)


def guardar_checkpoint(model, tiempo_alcanzado, semilla):
    """
    Guarda los pesos de la red usando el tiempo (t_max) como identificador.
    """
    nombre_archivo = f"pesos_pinn_semilla{semilla}_t{tiempo_alcanzado}.pth"
    ruta_completa = os.path.join(ruta_checkpoints, nombre_archivo)

    checkpoint = {
        'tiempo_alcanzado': tiempo_alcanzado,
        'semilla': semilla,
        'params_data': [p.data.clone() for p in model.params],
        'B_freq': model.B_freq.data.clone()
    }
    torch.save(checkpoint, ruta_completa)
    print(f"Progreso guardado: pesos estables hasta t={tiempo_alcanzado}s en {nombre_archivo}")


def cargar_checkpoint(model, tiempo_a_cargar, semilla):
    """
    Carga los pesos de un checkpoint especifico para reanudar desde ese punto.
    """
    nombre_archivo = f"pesos_pinn_semilla{semilla}_t{tiempo_a_cargar}.pth"
    ruta_completa = os.path.join(ruta_checkpoints, nombre_archivo)

    if os.path.exists(ruta_completa):
        checkpoint = torch.load(ruta_completa)
        for i, p in enumerate(model.params):
            p.data = checkpoint['params_data'][i]
        model.B_freq.data = checkpoint['B_freq']
        print(f"Checkpoint de t={checkpoint['tiempo_alcanzado']}s cargado con exito.")
        return True
    else:
        print(f"No se encontro el archivo {nombre_archivo}. La red comenzara desde cero.")
        return False

def guardar_checkpoint_qpinn(model, tiempo_alcanzado, semilla):
    """
    Guarda el estado del Q-PINN. Al ser un nn.Module se usa state_dict().
    """
    nombre_archivo = f"pesos_qpinn_semilla{semilla}_t{tiempo_alcanzado}.pth"
    ruta_completa = os.path.join(ruta_checkpoints, nombre_archivo)

    checkpoint = {
        'tiempo_alcanzado': tiempo_alcanzado,
        'semilla': semilla,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, ruta_completa)
    print(f"Progreso guardado: pesos estables hasta t={tiempo_alcanzado}s en {nombre_archivo}")


def cargar_checkpoint_qpinn(model, tiempo_a_cargar, semilla):
    """
    Carga un checkpoint del Q-PINN para reanudar o evaluar.
    """
    nombre_archivo = f"pesos_qpinn_semilla{semilla}_t{tiempo_a_cargar}.pth"
    ruta_completa = os.path.join(ruta_checkpoints, nombre_archivo)

    if os.path.exists(ruta_completa):
        checkpoint = torch.load(ruta_completa)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint de t={checkpoint['tiempo_alcanzado']}s cargado con exito.")
        return True
    else:
        print(f"No se encontro el archivo {nombre_archivo}. La red comenzara desde cero.")
        return False