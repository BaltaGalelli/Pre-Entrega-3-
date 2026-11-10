"""
Carga y preparación del corpus AG News (Pre-entregas 2 y 3).

AG News (Zhang, Zhao y LeCun, 2015): noticias en inglés (título + bajada) de 4 categorías:
World, Sports, Business y Sci_Tech. Splits oficiales: 120.000 de train y 7.600 de test.

Fuente de los datos, en orden de prioridad:
1. Los CSV de la cátedra (`ag_news_train.csv` y `ag_news_test.csv`, columnas text y label),
   si se copian en data/ag_news/.
2. Si no están, la versión pública original (CSV sin encabezado: clase, título, bajada),
   descargada de un repositorio de GitHub fijado a un commit y verificada con SHA-256.

El preprocesamiento con spaCy tarda unos minutos, así que el resultado se guarda en
data/ag_news/procesado/ y las corridas siguientes lo leen de ahí.
"""

import hashlib
import urllib.request

import pandas as pd

from .config import DIR_AG_NEWS
from .preprocesamiento import deduplicar_train, preprocess_corpus

CLASES = ["World", "Sports", "Business", "Sci_Tech"]
ARCHIVOS_CATEDRA = {"train": "ag_news_train.csv", "test": "ag_news_test.csv"}

COMMIT = "555590db4219b1243abb1918effd6a7425a2d75f"
URL_BASE = f"https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/{COMMIT}/data/ag_news_csv"
SHA256 = {
    "train.csv": "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e",
    "test.csv": "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699",
}


def _sha256(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _normalizar_etiquetas(etiquetas):
    """Etiquetas como texto (World, Sci/Tech...) o como número (0-3 o 1-4) -> nombres de CLASES."""
    if pd.api.types.is_numeric_dtype(etiquetas):
        base = int(etiquetas.min())
        return etiquetas.map({base + i: clase for i, clase in enumerate(CLASES)})
    return etiquetas.str.strip().str.replace("/", "_", regex=False)


def _descargar_publico(directorio):
    """Descarga los CSV públicos si faltan y verifica que sean exactamente los esperados."""
    directorio.mkdir(parents=True, exist_ok=True)
    for archivo, hash_esperado in SHA256.items():
        ruta = directorio / archivo
        if not ruta.exists():
            urllib.request.urlretrieve(f"{URL_BASE}/{archivo}", ruta)
        if _sha256(ruta) != hash_esperado:
            ruta.unlink()
            raise ValueError(f"{archivo} no coincide con la versión esperada: se borró, volvé a correr.")


def cargar_ag_news(directorio=DIR_AG_NEWS):
    """Devuelve (train, test, fuente) con columnas text y label. El test oficial no se modifica."""
    if all((directorio / archivo).exists() for archivo in ARCHIVOS_CATEDRA.values()):
        splits = {}
        for split, archivo in ARCHIVOS_CATEDRA.items():
            df = pd.read_csv(directorio / archivo)[["text", "label"]]
            df["label"] = _normalizar_etiquetas(df["label"])
            splits[split] = df
        return splits["train"], splits["test"], "catedra"

    _descargar_publico(directorio)
    splits = {}
    for split in ("train", "test"):
        crudo = pd.read_csv(directorio / f"{split}.csv", header=None, names=["clase", "titulo", "bajada"])
        splits[split] = pd.DataFrame({"text": crudo["titulo"] + " " + crudo["bajada"],
                                      "label": _normalizar_etiquetas(crudo["clase"])})
    return splits["train"], splits["test"], "publico"


def preparar_corpus(directorio=DIR_AG_NEWS, usar_cache=True):
    """Carga, preprocesa (regex + spaCy) y deduplica el train. Devuelve (train, test, informe).

    Columnas resultantes: text (original), label, text_clean (lemas con stop-words)
    y text_nostop (lemas sin stop-words).
    """
    train, test, fuente = cargar_ag_news(directorio)
    dir_cache = directorio / "procesado"
    cache_train = dir_cache / f"train_{fuente}.parquet"
    cache_test = dir_cache / f"test_{fuente}.parquet"
    cache_informe = dir_cache / f"informe_{fuente}.json"

    if usar_cache and cache_train.exists() and cache_test.exists() and cache_informe.exists():
        informe = pd.read_json(cache_informe, typ="series").to_dict()
        return pd.read_parquet(cache_train), pd.read_parquet(cache_test), informe

    for df in (train, test):
        df["text_clean"], df["text_nostop"] = preprocess_corpus(df["text"].tolist())
    train, informe = deduplicar_train(train, test)
    informe["fuente"] = fuente
    informe["test"] = len(test)

    dir_cache.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cache_train, index=False)
    test.to_parquet(cache_test, index=False)
    pd.Series(informe).to_json(cache_informe)
    return train, test, informe
