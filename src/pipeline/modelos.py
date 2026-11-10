"""Arquitecturas base: un clasificador lineal y un MLP de pocas capas."""

import torch
from torch import nn


class ClasificadorLineal(nn.Module):
    """Regresión logística multiclase: una sola capa lineal. Sirve de baseline."""

    def __init__(self, n_entradas: int, n_clases: int):
        super().__init__()
        self.capa = nn.Linear(n_entradas, n_clases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.capa(x)          # logits crudos: CrossEntropyLoss aplica el Softmax


class ClasificadorMLP(nn.Module):
    """MLP: capas ocultas densas con ReLU (y dropout opcional) y una capa de salida lineal."""

    def __init__(self, n_entradas: int, n_clases: int, capas_ocultas=(128, 64), dropout: float = 0.0):
        super().__init__()
        capas, n_anterior = [], n_entradas
        for n_neuronas in capas_ocultas:
            capas += [nn.Linear(n_anterior, n_neuronas), nn.ReLU()]
            if dropout > 0:
                capas.append(nn.Dropout(dropout))
            n_anterior = n_neuronas
        capas.append(nn.Linear(n_anterior, n_clases))   # logits crudos
        self.red = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)


def construir_modelo(config, n_entradas: int, n_clases: int) -> nn.Module:
    if config.arquitectura == "lineal":
        return ClasificadorLineal(n_entradas, n_clases)
    if config.arquitectura == "mlp":
        return ClasificadorMLP(n_entradas, n_clases, config.capas_ocultas, config.dropout)
    raise ValueError(f"Arquitectura desconocida: {config.arquitectura!r} (usar 'lineal' o 'mlp')")


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
