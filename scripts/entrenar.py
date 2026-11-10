"""
Pipeline completo de punta a punta: entorno -> datos -> modelo -> entrenamiento/validación -> test.

Uso (desde la raíz del repositorio):
    python scripts/entrenar.py                         # MLP con la configuración por defecto
    python scripts/entrenar.py --arquitectura lineal   # baseline lineal
    python scripts/entrenar.py --lr 0.01 --epocas 60
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.config import DIR_FIGURAS, DIR_MODELOS, DIR_REPORTES, Config  # noqa: E402
from pipeline.datos import preparar_datos  # noqa: E402
from pipeline.entorno import describir_entorno, fijar_semilla, obtener_dispositivo  # noqa: E402
from pipeline.entrenamiento import entrenar, evaluar  # noqa: E402
from pipeline.modelos import construir_modelo, contar_parametros  # noqa: E402
from pipeline.reportes import graficar_curvas, guardar_json  # noqa: E402


def parsear_argumentos():
    base = Config()
    p = argparse.ArgumentParser(description="Entrena el clasificador base y lo evalúa.")
    p.add_argument("--arquitectura", choices=["lineal", "mlp"], default=base.arquitectura)
    p.add_argument("--lr", type=float, default=base.learning_rate, help="learning rate de Adam")
    p.add_argument("--epocas", type=int, default=base.epocas)
    p.add_argument("--batch-size", type=int, default=base.batch_size)
    p.add_argument("--semilla", type=int, default=base.semilla)
    a = p.parse_args()
    return replace(base, arquitectura=a.arquitectura, learning_rate=a.lr, epocas=a.epocas,
                   batch_size=a.batch_size, semilla=a.semilla)


def main():
    config = parsear_argumentos()

    # 1. Entorno: semillas y dispositivo
    fijar_semilla(config.semilla)
    dispositivo = obtener_dispositivo()
    entorno = describir_entorno(dispositivo)
    print(f"PyTorch {entorno['torch']} | dispositivo: {entorno['dispositivo']}")

    # 2. Datos
    (loader_train, loader_val, loader_test), info, _ = preparar_datos(config)
    print(f"Datos: {info['n_train']} train / {info['n_val']} validación / {info['n_test']} test | "
          f"{info['n_variables']:,} términos TF-IDF, {info['n_clases']} clases")

    # 3. Modelo
    modelo = construir_modelo(config, info["n_variables"], info["n_clases"]).to(dispositivo)
    print(f"Modelo {config.arquitectura}: {contar_parametros(modelo):,} parámetros")

    # 4. Entrenamiento + validación por época
    modelo, historial, resumen = entrenar(modelo, loader_train, loader_val, config, dispositivo)

    # 5. Evaluación final en test (una sola vez, con los pesos de la mejor época)
    test = evaluar(modelo, loader_test, nn.CrossEntropyLoss(), dispositivo)
    print(f"Test -> loss {test['loss']:.4f} | accuracy {test['accuracy']:.3f} | F1 macro {test['f1_macro']:.3f}")

    # 6. Guardar artefactos
    for carpeta in (DIR_MODELOS, DIR_REPORTES, DIR_FIGURAS):
        carpeta.mkdir(parents=True, exist_ok=True)
    nombre = config.arquitectura
    torch.save(modelo.state_dict(), DIR_MODELOS / f"{nombre}.pt")
    pd.DataFrame(historial).to_csv(DIR_REPORTES / f"historial_{nombre}.csv", index=False)
    graficar_curvas(historial, resumen["mejor_epoca"], f"Banking77 · clasificador {nombre} · Adam lr={config.learning_rate:g}",
                    DIR_FIGURAS / f"curvas_{nombre}.png")
    guardar_json({"entorno": entorno, "config": config.a_dict(),
                  "datos": {k: v for k, v in info.items() if k != "clases"},
                  "parametros": contar_parametros(modelo),
                  # sin "segundos": el tiempo cambia en cada corrida y ensuciaría el repositorio
                  "entrenamiento": {k: v for k, v in resumen.items() if k != "segundos"}, "test": test},
                 DIR_REPORTES / f"metricas_{nombre}.json")
    print(f"Artefactos guardados en models/ y reports/ ({nombre})")


if __name__ == "__main__":
    main()
