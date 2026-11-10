"""Ciclo de entrenamiento, ciclo de validación, tracking de métricas y early stopping."""

import copy
import time

import torch
from sklearn.metrics import f1_score
from torch import nn


def entrenar_una_epoca(modelo, loader, criterio, optimizador, dispositivo):
    """Una pasada por el train. Devuelve la loss y el accuracy promedio de la época."""
    modelo.train()                                   # activa dropout (si hay)
    loss_total, aciertos, n = 0.0, 0, 0
    for entradas, etiquetas in loader:
        # Datos y modelo SIEMPRE en el mismo dispositivo
        entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)

        optimizador.zero_grad()                      # 1. limpiar gradientes (se acumulan por defecto)
        logits = modelo(entradas)                    # 2. forward pass
        loss = criterio(logits, etiquetas)           # 3. pérdida
        loss.backward()                              # 4. backward: autograd calcula los gradientes
        optimizador.step()                           # 5. actualización de pesos (Adam)

        # La loss del batch es un promedio: se pondera por el tamaño (el último batch puede ser más chico)
        loss_total += loss.item() * len(etiquetas)
        aciertos += (logits.argmax(dim=1) == etiquetas).sum().item()
        n += len(etiquetas)
    return loss_total / n, aciertos / n


@torch.no_grad()                                     # evaluar no necesita gradientes
def evaluar(modelo, loader, criterio, dispositivo):
    """Loss, accuracy y F1 macro sobre datos que el modelo no usa para entrenar."""
    modelo.eval()                                    # desactiva dropout
    loss_total, n = 0.0, 0
    predicciones, reales = [], []
    for entradas, etiquetas in loader:
        entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)
        logits = modelo(entradas)
        loss_total += criterio(logits, etiquetas).item() * len(etiquetas)
        n += len(etiquetas)
        predicciones.append(logits.argmax(dim=1).cpu())
        reales.append(etiquetas.cpu())
    predicciones, reales = torch.cat(predicciones), torch.cat(reales)
    return {
        "loss": loss_total / n,
        "accuracy": (predicciones == reales).float().mean().item(),
        "f1_macro": f1_score(reales, predicciones, average="macro"),
    }


def entrenar(modelo, loader_train, loader_val, config, dispositivo, verbose=True, cada=5):
    """
    Entrena con Adam y CrossEntropyLoss, registra las métricas por época y aplica early stopping:
    se queda con los pesos de la época con menor loss de validación.
    """
    modelo = modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config.learning_rate)

    historial = []
    mejor_loss, mejor_epoca, mejores_pesos = float("inf"), 0, None
    inicio = time.perf_counter()

    for epoca in range(1, config.epocas + 1):
        loss_tr, acc_tr = entrenar_una_epoca(modelo, loader_train, criterio, optimizador, dispositivo)
        val = evaluar(modelo, loader_val, criterio, dispositivo)
        historial.append({"epoca": epoca, "train_loss": loss_tr, "train_accuracy": acc_tr,
                          "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_f1_macro": val["f1_macro"]})

        if val["loss"] < mejor_loss:                 # mejoró la validación: guardamos esta versión
            mejor_loss, mejor_epoca = val["loss"], epoca
            mejores_pesos = copy.deepcopy(modelo.state_dict())

        if verbose and (epoca == 1 or epoca % cada == 0):
            print(f"Época {epoca:>3}/{config.epocas} | train loss {loss_tr:.4f} acc {acc_tr:.3f} | "
                  f"val loss {val['loss']:.4f} acc {val['accuracy']:.3f} f1 {val['f1_macro']:.3f}")

        if epoca - mejor_epoca >= config.paciencia:
            if verbose:
                print(f"Early stopping en la época {epoca}: {config.paciencia} épocas sin mejorar la validación")
            break

    modelo.load_state_dict(mejores_pesos)            # volvemos a la mejor época
    resumen = {"mejor_epoca": mejor_epoca, "mejor_val_loss": mejor_loss,
               "epocas_corridas": len(historial), "segundos": time.perf_counter() - inicio}
    if verbose:
        print(f"Mejor época: {mejor_epoca} (val loss {mejor_loss:.4f}) | {resumen['segundos']:.1f} s")
    return modelo, historial, resumen
