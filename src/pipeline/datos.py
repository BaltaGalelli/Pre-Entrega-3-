"""
Carga y preparación del dataset Banking77.

Banking77 (Casanueva et al., 2020, PolyAI): 13.083 consultas reales de clientes
de un banco, en inglés, etiquetadas con 77 motivos de consulta ("card_arrival",
"lost_or_stolen_card", "declined_transfer", ...). Licencia CC BY 4.0.

Los CSV se descargan una sola vez desde el repositorio oficial, fijados a un
commit concreto y verificados con su hash SHA-256: así todos trabajan con
exactamente los mismos datos.
"""

import hashlib
import urllib.request

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import DIR_DATOS

COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
URL_BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{COMMIT}/banking_data"
SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def _sha256(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def descargar_banking77(directorio=DIR_DATOS):
    """Descarga los CSV si no están y verifica que sean exactamente los esperados."""
    directorio.mkdir(parents=True, exist_ok=True)
    for archivo, hash_esperado in SHA256.items():
        ruta = directorio / archivo
        if not ruta.exists():
            urllib.request.urlretrieve(f"{URL_BASE}/{archivo}", ruta)
        if _sha256(ruta) != hash_esperado:
            ruta.unlink()
            raise ValueError(f"{archivo} no coincide con la versión esperada: se borró, volvé a correr.")
    return directorio


def cargar_dataset(directorio=DIR_DATOS):
    """Devuelve los DataFrames de train y test oficiales (columnas: text, category)."""
    descargar_banking77(directorio)
    return pd.read_csv(directorio / "train.csv"), pd.read_csv(directorio / "test.csv")


def particionar(train_completo, tamano_validacion, semilla):
    """Separa la validación del train oficial, estratificada por clase. El test oficial no se toca."""
    return train_test_split(train_completo, test_size=tamano_validacion,
                            stratify=train_completo["category"], random_state=semilla)


class DatasetDisperso(Dataset):
    """
    Dataset de PyTorch sobre una matriz TF-IDF dispersa (scipy CSR).

    La matriz completa en formato denso ocuparía cientos de MB; se guarda dispersa
    y cada batch se convierte a denso recién al armarlo (ver `armar_batch`).
    """

    def __init__(self, X, y):
        self.X = sparse.csr_matrix(X, dtype=np.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)   # copia: el array de pandas puede ser de solo lectura

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return i

    def armar_batch(self, indices):
        X = torch.from_numpy(self.X[indices].toarray())
        return X, self.y[indices]


def crear_loader(dataset, batch_size, mezclar, semilla=None):
    generador = torch.Generator().manual_seed(semilla) if mezclar else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=mezclar, generator=generador,
                      collate_fn=dataset.armar_batch)


def preparar_datos(config):
    """Todo el flujo de datos: descargar, partir, vectorizar con TF-IDF y armar los DataLoaders."""
    train_completo, test = cargar_dataset()
    train, val = particionar(train_completo, config.tamano_validacion, config.semilla)

    # Etiquetas: categorías ordenadas alfabéticamente -> enteros 0..76
    clases = sorted(train_completo["category"].unique())
    a_indice = {c: i for i, c in enumerate(clases)}

    # TF-IDF ajustado SOLO con train (el vocabulario y los pesos IDF no ven validación ni test)
    vectorizador = TfidfVectorizer(ngram_range=(1, config.ngramas_max), min_df=config.frecuencia_minima,
                                   sublinear_tf=True, lowercase=True)
    X_train = vectorizador.fit_transform(train["text"])
    X_val, X_test = vectorizador.transform(val["text"]), vectorizador.transform(test["text"])

    datasets = [DatasetDisperso(X, df["category"].map(a_indice).to_numpy())
                for X, df in ((X_train, train), (X_val, val), (X_test, test))]
    loaders = (crear_loader(datasets[0], config.batch_size, mezclar=True, semilla=config.semilla),
               crear_loader(datasets[1], config.batch_size, mezclar=False),
               crear_loader(datasets[2], config.batch_size, mezclar=False))
    info = {"n_variables": X_train.shape[1], "n_clases": len(clases), "clases": clases,
            "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    return loaders, info, vectorizador
