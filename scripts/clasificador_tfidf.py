"""
Pre-entrega 3: clasificador supervisado con TF-IDF sobre AG News, de punta a punta.

    python scripts/clasificador_tfidf.py

1. Carga AG News y aplica el preprocesamiento de la Pre-entrega 2 (regex + spaCy), con caché.
2. Separa una validación estratificada del train (el test oficial no se toca hasta el final).
3. Experimenta con ngram_range y max_features del TfidfVectorizer para Naive Bayes, SVM lineal
   y Regresión Logística, y ajusta la regularización del mejor modelo, siempre en validación.
4. Reentrena la mejor configuración con todo el train y la evalúa UNA vez en el test:
   classification_report, matriz de confusión, confusiones principales y términos por clase.
5. Guarda los resultados en reports/: experimentos (CSV), métricas (JSON), classification_report (TXT)
   y figuras (comparación de experimentos y matriz de confusión).
"""

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from pipeline.ag_news import CLASES, preparar_corpus  # noqa: E402
from pipeline.clasico import (confusiones_principales, crear_modelos, crear_pipeline, crear_vectorizador,  # noqa: E402
                              experimentar, graficar_experimentos, graficar_matriz_confusion, terminos_por_clase)
from pipeline.config import DIR_FIGURAS, DIR_REPORTES  # noqa: E402

SEMILLA = 42
COLUMNA = "text_nostop"            # lemas sin stop-words (se conservan negaciones y "us")
GRILLA = [{"ngram_range": (1, 1), "max_features": m} for m in (10_000, 30_000, None)] + \
         [{"ngram_range": (1, 2), "max_features": m} for m in (10_000, 30_000, 100_000, None)]
AJUSTE = {"SVM lineal": ("C", [0.1, 0.3, 1.0, 3.0], lambda v: LinearSVC(C=v, random_state=SEMILLA)),
          "Regresión logística": ("C", [1.0, 3.0, 10.0], lambda v: LogisticRegression(C=v, max_iter=1000, random_state=SEMILLA)),
          "Naive Bayes": ("alpha", [0.01, 0.1, 0.3, 1.0], lambda v: MultinomialNB(alpha=v))}


def main():
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    # 1. Datos preprocesados (Pre-entrega 2)
    train, test, informe = preparar_corpus()
    print("Corpus:", informe)

    # 2. Validación estratificada separada del train
    tr, val = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=SEMILLA)

    # 3a. Grilla del vectorizador x modelos
    resultados = experimentar(tr[COLUMNA], tr["label"], val[COLUMNA], val["label"], GRILLA, crear_modelos(SEMILLA))
    resultados.to_csv(DIR_REPORTES / "pe3_experimentos.csv", index=False)
    graficar_experimentos(resultados, DIR_FIGURAS / "pe3_experimentos.png")
    mejor = resultados.loc[resultados["F1 macro val"].idxmax()]
    print(resultados.sort_values("F1 macro val", ascending=False).head(5).to_string(index=False))

    # 3b. Regularización del mejor modelo, con la mejor configuración del vectorizador
    config_vec = {"ngram_range": ast.literal_eval(mejor["ngram_range"]),
                  "max_features": None if mejor["max_features"] == "sin límite" else int(mejor["max_features"])}
    parametro, valores, constructor = AJUSTE[mejor["modelo"]]
    vec_train = crear_vectorizador(**config_vec)
    X_tr, X_val = vec_train.fit_transform(tr[COLUMNA]), vec_train.transform(val[COLUMNA])
    ajuste = {v: round(f1_score(val["label"], constructor(v).fit(X_tr, tr["label"]).predict(X_val), average="macro"), 4)
              for v in valores}
    valor_elegido = max(ajuste, key=ajuste.get)
    print(f"Ajuste de {parametro} ({mejor['modelo']}):", ajuste)

    # 4. Modelo final: todo el train, evaluado UNA vez en el test
    final = crear_pipeline(crear_vectorizador(**config_vec), constructor(valor_elegido))
    final.fit(train[COLUMNA], train["label"])
    pred = final.predict(test[COLUMNA])
    reporte = classification_report(test["label"], pred, labels=CLASES, digits=4)
    print(reporte)
    encabezado = (f"{mejor['modelo']} ({parametro}={valor_elegido}) + TF-IDF {config_vec}, min_df=2\n"
                  f"Entrenado con {len(train):,} noticias de train, evaluado en el test oficial ({len(test):,} noticias)\n\n")
    (DIR_REPORTES / "pe3_classification_report.txt").write_text(encabezado + reporte)
    _, matriz = graficar_matriz_confusion(test["label"], pred, CLASES, DIR_FIGURAS / "pe3_matriz_confusion.png")

    metricas = {
        "corpus": {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in informe.items()},
        "modelo": mejor["modelo"], parametro: valor_elegido,
        "vectorizador": {"ngram_range": list(config_vec["ngram_range"]), "max_features": config_vec["max_features"],
                         "min_df": 2, "terminos": len(final.named_steps["tfidf"].vocabulary_)},
        "validacion": {"F1 macro": float(ajuste[valor_elegido])},
        "test": {"accuracy": round(accuracy_score(test["label"], pred), 4),
                 "F1 macro": round(f1_score(test["label"], pred, average="macro"), 4),
                 "reporte": classification_report(test["label"], pred, labels=CLASES, output_dict=True)},
        "matriz_confusion": {"clases": CLASES, "valores": matriz.tolist()},
        "confusiones_principales": confusiones_principales(matriz, CLASES).to_dict(orient="records"),
        "terminos_por_clase": terminos_por_clase(final).to_dict(orient="list"),
    }
    (DIR_REPORTES / "pe3_metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1))
    print("Resultados guardados en reports/")


if __name__ == "__main__":
    main()
