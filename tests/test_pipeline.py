"""Tests rápidos de las piezas del pipeline (pytest)."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from pipeline.config import Config
from pipeline.datos import SHA256, DIR_DATOS, _sha256, cargar_dataset, particionar, preparar_datos
from pipeline.entorno import fijar_semilla, obtener_dispositivo
from pipeline.entrenamiento import entrenar, entrenar_una_epoca, evaluar
from pipeline.modelos import ClasificadorLineal, ClasificadorMLP, construir_modelo, contar_parametros

CONFIG_RAPIDA = replace(Config(), epocas=2)


@pytest.fixture(scope="module")
def datos():
    return preparar_datos(CONFIG_RAPIDA)


def test_archivos_verificados_y_sin_cambios():
    train, test = cargar_dataset()
    for archivo, hash_esperado in SHA256.items():
        assert _sha256(DIR_DATOS / archivo) == hash_esperado
    assert (len(train), len(test)) == (10003, 3080)
    assert train["category"].nunique() == test["category"].nunique() == 77


def test_particion_sin_solapamiento_y_estratificada():
    train_completo, test = cargar_dataset()
    train, val = particionar(train_completo, 0.15, 42)
    assert len(train) + len(val) == len(train_completo)
    assert not set(train.index) & set(val.index)
    assert not set(train["text"]) & set(test["text"])
    assert val["category"].nunique() == 77        # todas las clases llegan a validación


def test_vocabulario_solo_del_train(datos):
    (_, _, _), _, vectorizador = datos
    train_completo, test = cargar_dataset()
    train, _ = particionar(train_completo, CONFIG_RAPIDA.tamano_validacion, CONFIG_RAPIDA.semilla)
    analizar = vectorizador.build_analyzer()
    palabras_train = {t for texto in train["text"] for t in analizar(texto) if " " not in t}
    palabras_test = [t for texto in test["text"] for t in analizar(texto) if " " not in t]
    solo_en_test = {t for t in palabras_test if t not in palabras_train and palabras_test.count(t) >= 2}
    assert solo_en_test, "debería haber palabras que aparecen en test y nunca en train"
    # Si el TF-IDF se hubiera ajustado con el test, estas palabras estarían en el vocabulario
    assert not solo_en_test & set(vectorizador.vocabulary_)


def test_batches_densos_y_tipos(datos):
    (loader_train, _, _), info, _ = datos
    X, y = next(iter(loader_train))
    assert X.dtype == torch.float32 and y.dtype == torch.int64
    assert X.shape == (64, info["n_variables"]) and y.shape == (64,)
    assert int(y.max()) < info["n_clases"] == 77


@pytest.mark.parametrize("clase, ocultas", [(ClasificadorLineal, None), (ClasificadorMLP, (256,))])
def test_formas_y_parametros(clase, ocultas):
    modelo = clase(1000, 77) if ocultas is None else clase(1000, 77, ocultas, 0.3)
    assert modelo(torch.randn(5, 1000)).shape == (5, 77)
    esperado = 1000 * 77 + 77 if ocultas is None else (1000 + 1) * 256 + (256 + 1) * 77
    assert contar_parametros(modelo) == esperado


def test_arquitectura_desconocida():
    with pytest.raises(ValueError):
        construir_modelo(replace(Config(), arquitectura="transformer"), 100, 77)


def test_dispositivo_valido():
    assert obtener_dispositivo().type in {"cpu", "cuda", "mps"}


def test_una_epoca_baja_la_loss_y_actualiza_pesos(datos):
    (loader_train, _, _), info, _ = datos
    fijar_semilla(0)
    modelo, dispositivo = ClasificadorMLP(info["n_variables"], 77, (64,)), torch.device("cpu")
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    antes = evaluar(modelo, loader_train, criterio, dispositivo)["loss"]
    pesos_antes = [p.clone() for p in modelo.parameters()]
    entrenar_una_epoca(modelo, loader_train, criterio, optimizador, dispositivo)
    assert evaluar(modelo, loader_train, criterio, dispositivo)["loss"] < antes
    assert any(not torch.equal(a, b) for a, b in zip(pesos_antes, modelo.parameters()))


def test_evaluar_no_modifica_pesos(datos):
    (_, loader_val, _), info, _ = datos
    modelo = ClasificadorMLP(info["n_variables"], 77, (64,), 0.3)
    antes = [p.clone() for p in modelo.parameters()]
    resultado = evaluar(modelo, loader_val, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(antes, modelo.parameters()))
    assert 0 <= resultado["accuracy"] <= 1 and 0 <= resultado["f1_macro"] <= 1


def correr(semilla):
    config = replace(CONFIG_RAPIDA, semilla=semilla, capas_ocultas=(32,))
    fijar_semilla(config.semilla)
    (loader_train, loader_val, _), info, _ = preparar_datos(config)
    _, historial, _ = entrenar(construir_modelo(config, info["n_variables"], 77), loader_train, loader_val,
                               config, torch.device("cpu"), verbose=False)
    return [round(h["val_loss"], 6) for h in historial]


def test_reproducible_con_la_misma_semilla():
    assert correr(42) == correr(42)
    assert correr(42) != correr(7)
