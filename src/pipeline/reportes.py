"""Gráfico de curvas de entrenamiento y guardado de resultados."""

import json

import matplotlib.pyplot as plt
import pandas as pd


def graficar_curvas(historial, mejor_epoca=None, titulo="", ruta=None):
    df = pd.DataFrame(historial)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    ejes[0].plot(df["epoca"], df["train_loss"], label="train", lw=2)
    ejes[0].plot(df["epoca"], df["val_loss"], label="validación", lw=2, ls="--")
    ejes[0].set_title("Pérdida (CrossEntropy) por época")
    ejes[0].set_ylabel("loss")
    ejes[1].plot(df["epoca"], df["train_accuracy"], label="train", lw=2)
    ejes[1].plot(df["epoca"], df["val_accuracy"], label="validación", lw=2, ls="--")
    ejes[1].set_title("Accuracy por época")
    ejes[1].set_ylabel("accuracy")
    for eje in ejes:
        eje.set_xlabel("época")
        eje.grid(alpha=0.3)
        if mejor_epoca:
            eje.axvline(mejor_epoca, color="grey", lw=1, ls=":", label=f"mejor época ({mejor_epoca})")
        eje.legend()
    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=120, bbox_inches="tight")
    return fig


def guardar_json(datos, ruta):
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False))
