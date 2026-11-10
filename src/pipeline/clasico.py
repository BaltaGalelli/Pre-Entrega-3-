"""
Clasificación supervisada clásica (Pre-entrega 3): TF-IDF + Naive Bayes / SVM lineal / Regresión Logística.

Regla anti-fuga: el TfidfVectorizer se ajusta SOLO con los textos de entrenamiento (fit_transform)
y a validación y test se les aplica transform. Para el modelo final se usa un Pipeline de
scikit-learn, que hace eso solo.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Paleta: un azul para magnitudes, tres colores fijos para los modelos (siempre en el mismo orden)
TINTA, TINTA2, GRILLA, SUPERFICIE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
COLOR_MODELO = {"Naive Bayes": "#2a78d6", "SVM lineal": "#eb6834", "Regresión logística": "#1baf7a"}
MAPA_AZUL = LinearSegmentedColormap.from_list("azules", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def crear_modelos(semilla=42):
    """Los tres clasificadores de referencia, con sus parámetros por defecto."""
    return {
        "Naive Bayes": MultinomialNB(),
        "SVM lineal": LinearSVC(random_state=semilla),
        "Regresión logística": LogisticRegression(max_iter=1000, random_state=semilla),
    }


def crear_vectorizador(ngram_range=(1, 2), max_features=None, min_df=2, **kwargs):
    """TF-IDF para texto ya preprocesado (lemas en minúscula separados por espacios).

    token_pattern=r"\\S+" respeta los tokens que dejó spaCy (incluidos los de una letra y los números),
    y lowercase=False porque el texto ya viene en minúsculas.
    """
    parametros = {"token_pattern": r"\S+", "lowercase": False}
    parametros.update(kwargs)
    return TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df, **parametros)


def crear_pipeline(vectorizador, modelo):
    """Vectorizador + clasificador en un solo objeto: fit ajusta el TF-IDF solo con lo que recibe."""
    return Pipeline([("tfidf", vectorizador), ("clasificador", modelo)])


def memoria_mb(matriz):
    """Memoria que ocupa una matriz esparsa CSR (data + indices + indptr)."""
    return (matriz.data.nbytes + matriz.indices.nbytes + matriz.indptr.nbytes) / 1e6


def experimentar(textos_train, y_train, textos_val, y_val, grilla_vectorizador, modelos):
    """Prueba cada configuración del vectorizador con cada modelo.

    El vectorizador se ajusta una vez por configuración, solo con textos_train, y la misma matriz
    se reutiliza para los tres modelos. Devuelve un DataFrame con las métricas de validación.
    """
    filas = []
    for config in grilla_vectorizador:
        vectorizador = crear_vectorizador(**config)
        X_train = vectorizador.fit_transform(textos_train)      # fit SOLO con train
        X_val = vectorizador.transform(textos_val)              # transform (sin fit) para validación
        for nombre, modelo in modelos.items():
            t0 = time.time()
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_val)
            filas.append({
                "modelo": nombre,
                "ngram_range": str(config.get("ngram_range", (1, 2))),
                "max_features": config.get("max_features") or "sin límite",
                "términos": X_train.shape[1],
                "memoria train (MB)": round(memoria_mb(X_train), 1),
                "accuracy val": round(accuracy_score(y_val, pred), 4),
                "F1 macro val": round(f1_score(y_val, pred, average="macro"), 4),
                "segundos": round(time.time() - t0, 1),
            })
    return pd.DataFrame(filas)


def evaluar(textos_train, y_train, textos_val, y_val, vectorizador, modelo):
    """Entrena un pipeline (vectorizador + modelo) y devuelve sus métricas de validación."""
    pipeline = crear_pipeline(vectorizador, modelo).fit(textos_train, y_train)
    pred = pipeline.predict(textos_val)
    return {"términos": len(pipeline.named_steps["tfidf"].vocabulary_),
            "accuracy val": round(accuracy_score(y_val, pred), 4),
            "F1 macro val": round(f1_score(y_val, pred, average="macro"), 4)}


def graficar_experimentos(resultados, ruta=None):
    """Dot plot: F1 macro de validación por configuración del vectorizador, un color por modelo."""
    etiquetas = (resultados["ngram_range"].map({"(1, 1)": "unigramas", "(1, 2)": "uni + bigramas"})
                 + " · " + resultados["max_features"].map(lambda v: v if isinstance(v, str) else f"{v:,}".replace(",", ".")))
    resultados = resultados.assign(configuracion=etiquetas)
    orden = list(dict.fromkeys(etiquetas))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(orden) + 1.4), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    for nombre, color in COLOR_MODELO.items():
        datos = resultados[resultados["modelo"] == nombre].set_index("configuracion").reindex(orden)
        ax.scatter(datos["F1 macro val"], range(len(orden)), s=64, color=color, label=nombre,
                   edgecolor=SUPERFICIE, linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(orden)), orden)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRILLA, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRILLA)
    ax.tick_params(colors=TINTA2, length=0)
    ax.set_xlabel("F1 macro en validación", color=TINTA2)
    ax.set_title("TF-IDF: ngram_range y max_features por modelo", loc="left", color=TINTA, fontweight="bold")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9,
              labelcolor=TINTA2, borderaxespad=1.8)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=180)
    return fig


def graficar_matriz_confusion(y_true, y_pred, clases, ruta=None):
    """Matriz de confusión en cantidades y normalizada por fila (= recall de cada clase en la diagonal)."""
    matriz = confusion_matrix(y_true, y_pred, labels=clases)
    normalizada = matriz / matriz.sum(axis=1, keepdims=True)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SUPERFICIE)
    for ax, datos, titulo, formato in [(ejes[0], matriz, "Cantidades", "{:,.0f}"),
                                       (ejes[1], normalizada, "Normalizada por fila (recall)", "{:.1%}")]:
        ax.imshow(normalizada, cmap=MAPA_AZUL, vmin=0, vmax=1)
        for i in range(len(clases)):
            for j in range(len(clases)):
                texto = formato.format(datos[i, j]).replace(",", ".") if formato.startswith("{:,") else formato.format(datos[i, j])
                ax.text(j, i, texto, ha="center", va="center", fontsize=10,
                        color="white" if normalizada[i, j] > 0.5 else TINTA)
        ax.set_xticks(range(len(clases)), clases)
        ax.set_yticks(range(len(clases)), clases)
        ax.set_xlabel("Predicción", color=TINTA2)
        ax.set_ylabel("Categoría real", color=TINTA2)
        ax.set_title(titulo, loc="left", color=TINTA, fontweight="bold")
        ax.tick_params(colors=TINTA2, length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=180)
    return fig, matriz


def confusiones_principales(matriz, clases, n=5):
    """Los n pares (real -> predicha) con más errores, con su peso sobre la clase real."""
    filas = []
    for i, real in enumerate(clases):
        for j, predicha in enumerate(clases):
            if i != j:
                filas.append({"real": real, "predicha": predicha, "errores": int(matriz[i, j]),
                              "% de la clase real": round(100 * matriz[i, j] / matriz[i].sum(), 1)})
    return pd.DataFrame(filas).sort_values("errores", ascending=False).head(n).reset_index(drop=True)


def terminos_por_clase(pipeline, n=10):
    """Los n términos con más peso hacia cada clase (coeficientes del modelo lineal)."""
    vocabulario = pipeline.named_steps["tfidf"].get_feature_names_out()
    modelo = pipeline.named_steps["clasificador"]
    pesos = modelo.coef_ if hasattr(modelo, "coef_") else modelo.feature_log_prob_
    return pd.DataFrame({clase: [vocabulario[i] for i in np.argsort(-pesos[k])[:n]]
                         for k, clase in enumerate(modelo.classes_)})
