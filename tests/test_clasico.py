"""Tests de la Pre-entrega 3: preprocesamiento reutilizado, carga de AG News y clasificador TF-IDF (pytest)."""

import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from pipeline.ag_news import CLASES, _normalizar_etiquetas
from pipeline.clasico import crear_modelos, crear_pipeline, crear_vectorizador, experimentar
from pipeline.preprocesamiento import STOP_WORDS, clean_text, deduplicar_train, preprocess_text

NOTICIA_RUIDOSA = ("Stocks End Up (Reuters) Reuters - Stocks rose past #36;46\\a barrel, said Dell Inc. (DELL.O) "
                   "&lt;A HREF=\"http://www.investor.reuters.com/x\"&gt;full quote&lt;/A&gt;")


def test_clean_text_saca_el_ruido_de_ag_news():
    limpio = clean_text(NOTICIA_RUIDOSA)
    for ruido in ["Reuters", "#36;", "\\", "DELL.O", "&lt;", "<", "http", "www"]:
        assert ruido not in limpio, ruido
    assert "$46" in limpio


def test_preprocess_text_lematiza_antes_de_pasar_a_minusculas():
    # Con el texto en minúscula, spaCy lematizaría "us" (Estados Unidos) como el pronombre "we"
    assert preprocess_text("US stocks rose") == "us stock rise"


def test_stop_words_conservan_negaciones_y_us():
    assert {"not", "no", "never", "us"}.isdisjoint(STOP_WORDS)
    assert {"the", "of", "and"} <= STOP_WORDS
    # "US" (el país) sobrevive; "us" pronombre se lematiza como "we" y se va como stop-word
    assert preprocess_text("The deal is not good for the US", remove_stopwords=True) == "deal not good us"
    assert preprocess_text("The deal is not good for us", remove_stopwords=True) == "deal not good"


@pytest.mark.parametrize("etiquetas", [pd.Series([1, 2, 3, 4]), pd.Series([0, 1, 2, 3]),
                                       pd.Series(["World", "Sports", "Business", "Sci/Tech"])])
def test_etiquetas_normalizadas(etiquetas):
    assert _normalizar_etiquetas(etiquetas).tolist() == CLASES


def test_deduplicar_solo_toca_el_train():
    train = pd.DataFrame({"text_clean": ["a", "a", "b", "b", "c", "d"],
                          "label": ["World", "World", "Sports", "Business", "World", "Sports"]})
    test = pd.DataFrame({"text_clean": ["d", "e"], "label": ["Sports", "World"]})
    limpio, informe = deduplicar_train(train, test)
    assert limpio["text_clean"].tolist() == ["a", "c"]       # "b" contradictoria, "a" duplicada, "d" igual al test
    assert informe == {"inicial": 6, "contradictorios": 2, "duplicados": 1, "iguales_al_test": 1, "final": 2}
    assert len(test) == 2


def test_vectorizador_respeta_los_tokens_de_spacy():
    vec = crear_vectorizador(ngram_range=(1, 1), min_df=1).fit(["u.s 2 a deal", "deal x"])
    assert {"u.s", "2", "a", "x"} <= set(vec.vocabulary_)    # el patrón por defecto de sklearn los descartaría


def test_sin_fuga_el_vocabulario_sale_solo_del_train():
    textos_train = ["oil price rise", "team win cup", "oil bank tax", "team coach season"]
    y_train = ["Business", "Sports", "Business", "Sports"]
    pipeline = crear_pipeline(crear_vectorizador(ngram_range=(1, 1), min_df=1), LinearSVC())
    pipeline.fit(textos_train, y_train)
    pipeline.predict(["zoomcoin oil rally"])                  # predecir no agrega vocabulario
    vocabulario = pipeline.named_steps["tfidf"].vocabulary_
    assert "zoomcoin" not in vocabulario and "rally" not in vocabulario
    assert set(vocabulario) == {w for t in textos_train for w in t.split()}


def test_experimentar_una_fila_por_configuracion_y_modelo():
    textos = ["oil price rise", "team win cup", "oil bank tax", "team coach season"] * 5
    etiquetas = ["Business", "Sports"] * 10
    grilla = [{"ngram_range": (1, 1), "max_features": None}, {"ngram_range": (1, 2), "max_features": 5}]
    resultados = experimentar(textos, etiquetas, textos, etiquetas, grilla, crear_modelos())
    assert len(resultados) == len(grilla) * 3
    assert resultados.loc[resultados["max_features"] == 5, "términos"].eq(5).all()
    assert resultados["F1 macro val"].between(0, 1).all()
