"""Reproducibilidad (semillas) y detección automática del dispositivo de cómputo."""

import os
import random

import numpy as np
import torch


def fijar_semilla(semilla: int = 42) -> None:
    """Fija todas las fuentes de aleatoriedad: Python, NumPy y PyTorch (CPU y GPU)."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    # Algoritmos deterministas en GPU (algo más lentos, pero repetibles)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(semilla)


def obtener_dispositivo() -> torch.device:
    """GPU NVIDIA (cuda) si hay, si no Apple Silicon (mps), si no CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describir_entorno(dispositivo: torch.device) -> dict:
    info = {"torch": torch.__version__, "dispositivo": str(dispositivo)}
    if dispositivo.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
