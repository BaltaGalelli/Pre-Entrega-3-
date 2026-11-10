"""
Preprocesamiento del texto de AG News (Pre-entrega 2, reutilizado en la Pre-entrega 3).

Pipeline: limpieza estructural con regex -> lematización con spaCy -> minúsculas.

- clean_text: entidades HTML rotas, etiquetas, URLs, mails, barras invertidas, tickers de bolsa
  y firmas de agencia ("(Reuters) Reuters - "), que en AG News funcionan como atajos de la categoría.
- preprocess_text / preprocess_corpus: lemas en minúscula; dos versiones de cada documento,
  con stop-words (text_clean) y sin ellas (text_nostop, conservando negaciones y "us").
- deduplicar_train: limpieza a nivel corpus, solo sobre train.
"""
import html
import re

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_SPACY

# ===== Patrones de limpieza estructural (se compilan una sola vez) =====
# AG News trae entidades HTML rotas: "#39;" (sin "&"), "quot;", "amp;", "&lt;b&gt;"...
PATRON_ENTIDAD_NUMERICA = re.compile(r"&?#(\d+);")                      # #39; -> '   #36; -> $
PATRON_ENTIDAD_SIN_AMP = re.compile(r"(?<!&)\b(quot|amp|lt|gt|nbsp|hellip|rsquo|lsquo|rdquo|ldquo|mdash|ndash);")
PATRON_HTML = re.compile(r"<[^>]*>")                                      # <b>, </A>, <A HREF="...">
PATRON_URL = re.compile(r"https?://\S+|www\.\S+")
PATRON_EMAIL = re.compile(r"\S+@\S+\.\S+")
PATRON_BARRA = re.compile(r"\\+")                                         # "\\" usado como salto de línea
# Firmas de agencia: "(Reuters)", "(AP)", "Reuters - ", "AFP - ", "--(Business Wire)--"...
AGENCIAS = r"Reuters|AP|AFP|CP|Canadian Press|Business Wire|PRNewswire|Ziff Davis|Forbes\.com|TechWeb|NewsFactor|Sporting News|Sports Network|SPACE\.com|washingtonpost\.com|Newsday"
PATRON_FIRMA_REPETIDA = re.compile(r"\(([A-Z][\w.&' ]{1,30})\)\s+\1\s+-+\s")   # "(PC World) PC World - " (cualquier fuente)
PATRON_AGENCIA_PARENTESIS = re.compile(rf"-*\s*\((?:{AGENCIAS})\)\s*-*")
PATRON_AGENCIA_GUION = re.compile(rf"\b(?:{AGENCIAS})\s+-+\s")
PATRON_TICKER = re.compile(r"\(?\b[A-Z]{2,5}\.(?:O|N|OQ|L|T|PA|DE|TO|AS|MI|HK|KS|AX|SA|MC|F)\b(?::\s*Quote,\s*Profile,\s*Research)?\)?")  # tickers de Reuters: (DELL.O), (MLS.N: Quote, Profile, Research)
PATRON_ESPACIOS = re.compile(r"\s+")
PATRON_TIENE_ALFANUMERICO = re.compile(r"[a-z0-9]")
PATRON_SIGLA_CON_PUNTOS = re.compile(r"^(?:[a-z]\.){2,}$")                    # u.s. -> us, u.n. -> un

MODELO_SPACY = "en_core_web_sm"
_nlp = None


def obtener_nlp():
    """Carga spaCy una sola vez (y descarga el modelo si falta). Solo tagger y lematizador:
    el parser y el NER no se usan y hacen todo más lento."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(MODELO_SPACY, disable=["parser", "ner"])
        except OSError:
            spacy.cli.download(MODELO_SPACY)
            _nlp = spacy.load(MODELO_SPACY, disable=["parser", "ner"])
    return _nlp


# Stop-words de spaCy con dos excepciones:
#  - las negaciones, que cambian el sentido de la frase;
#  - "us": como se lematiza con mayúsculas, el pronombre "us" ya quedó como "we"; el "us" que sobrevive es
#    casi siempre "US" (Estados Unidos), una palabra clave en noticias internacionales.
NEGACIONES = {"no", "not", "never", "nor", "none", "nobody", "nothing", "neither", "nowhere", "without", "cannot"}
STOP_WORDS = frozenset(STOP_WORDS_SPACY - NEGACIONES - {"us"})


def _entidad_numerica(match):
    codigo = int(match.group(1))
    # 145-151 son caracteres de Windows-1252 mal codificados (comillas y guiones)
    reemplazos = {145: "'", 146: "'", 147: '"', 148: '"', 150: "-", 151: " - "}
    return reemplazos.get(codigo, chr(codigo) if codigo < 0x110000 else " ")


def clean_text(text: str) -> str:
    """Limpieza estructural con regex: entidades, HTML, URLs, mails, barras, tickers y firmas de agencia."""
    text = PATRON_ENTIDAD_NUMERICA.sub(_entidad_numerica, text)
    text = PATRON_ENTIDAD_SIN_AMP.sub(r"&\1;", text)
    text = html.unescape(html.unescape(text))      # dos veces: hay entidades doblemente escapadas (&amp;lt;)
    text = PATRON_HTML.sub(" ", text)
    text = PATRON_URL.sub(" ", text)
    text = PATRON_EMAIL.sub(" ", text)
    text = PATRON_BARRA.sub(" ", text)
    text = PATRON_TICKER.sub(" ", text)
    text = PATRON_FIRMA_REPETIDA.sub(" ", text)
    text = PATRON_AGENCIA_PARENTESIS.sub(" ", text)
    text = PATRON_AGENCIA_GUION.sub(" ", text)
    return PATRON_ESPACIOS.sub(" ", text).strip()


def _lemas(doc, remove_stopwords: bool) -> str:
    """Lemas en minúscula, sin puntuación, espacios ni posesivos; opcionalmente sin stop-words."""
    lemas = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        lema = token.lemma_.lower()
        if lema == "'s" or not PATRON_TIENE_ALFANUMERICO.search(lema):
            continue
        if PATRON_SIGLA_CON_PUNTOS.match(lema):
            lema = lema.replace(".", "")
        if remove_stopwords and lema in STOP_WORDS:
            continue
        lemas.append(lema)
    return " ".join(lemas)


def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """Pipeline completo para un documento: regex -> lematización con spaCy -> minúsculas.

    Se lematiza ANTES de pasar a minúsculas: con el texto en minúscula, spaCy confunde
    "US" (Estados Unidos) con el pronombre "us" y lo lematiza como "we".
    """
    return _lemas(obtener_nlp()(clean_text(text)), remove_stopwords)


def preprocess_corpus(textos, batch_size: int = 1000):
    """Mismo pipeline que preprocess_text, en lote con spaCy (mucho más rápido).

    Devuelve dos versiones de cada documento: con stop-words (para modelos de secuencia)
    y sin stop-words (para bolsa de palabras, TF-IDF y n-gramas).
    """
    limpios = [clean_text(t) for t in textos]
    con_stop, sin_stop = [], []
    for doc in obtener_nlp().pipe(limpios, batch_size=batch_size):
        con_stop.append(_lemas(doc, remove_stopwords=False))
        sin_stop.append(_lemas(doc, remove_stopwords=True))
    return con_stop, sin_stop


def deduplicar_train(train, test, columna="text_clean"):
    """Limpieza a nivel corpus, solo sobre train (el test queda intacto).

    1. Saca los textos con etiquetas contradictorias (la misma noticia en dos categorías).
    2. Saca los duplicados exactos (se queda con la primera aparición).
    3. Saca de train las noticias idénticas a una del test, para que la evaluación no sea optimista.
    """
    n_clases = train.groupby(columna)["label"].transform("nunique")
    paso1 = train[n_clases == 1]
    paso2 = paso1.drop_duplicates(subset=columna)
    paso3 = paso2[~paso2[columna].isin(set(test[columna]))]
    informe = {"inicial": len(train), "contradictorios": len(train) - len(paso1),
               "duplicados": len(paso1) - len(paso2), "iguales_al_test": len(paso2) - len(paso3), "final": len(paso3)}
    return paso3.reset_index(drop=True), informe

