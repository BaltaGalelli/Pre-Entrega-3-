# Datos

## Banking77

| | |
|---|---|
| **Qué es** | 13.083 consultas reales de clientes de un banco, en inglés, etiquetadas con 77 motivos de consulta |
| **Partición oficial** | 10.003 de train (entre 35 y 187 por motivo) y 3.080 de test (40 por motivo) |
| **Fuente** | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) — también en [Hugging Face](https://huggingface.co/datasets/PolyAI/banking77) |
| **Licencia** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): uso libre citando a los autores |
| **Paper** | Casanueva, I., Temčinas, T., Gerz, D., Henderson, M. y Vulić, I. (2020). *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020 |

## Cómo se obtiene

No hay que descargar nada a mano: la primera vez que se corre el pipeline, `src/pipeline/datos.py` baja `train.csv` y `test.csv` a `data/banking77/`.

- La URL está **fijada a un commit** del repositorio oficial (`57ec275`), así que siempre se descarga la misma versión.
- Cada archivo se verifica con su **hash SHA-256**; si no coincide, se borra y el pipeline se detiene en lugar de entrenar con datos distintos.

| Archivo | SHA-256 |
|---|---|
| `train.csv` | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `test.csv` | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

`data/banking77/` está en el `.gitignore`: los datos se documentan acá en lugar de subirlos al repositorio.

## AG News (corpus del proyecto desde la Pre-entrega 2)

| | |
|---|---|
| **Qué es** | Noticias en inglés (título + bajada) de 4 categorías: World, Sports, Business y Sci_Tech |
| **Partición oficial** | 120.000 de train y 7.600 de test (30.000 y 1.900 por categoría) |
| **Fuente principal** | Los CSV de la cátedra: `ag_news_train.csv` y `ag_news_test.csv` (columnas `text` y `label`) |
| **Fuente alternativa** | La versión pública original, en [mhjabreel/CharCnn_Keras](https://github.com/mhjabreel/CharCnn_Keras/tree/555590db4219b1243abb1918effd6a7425a2d75f/data/ag_news_csv) (sin encabezado: clase, título, bajada) |
| **Paper** | Zhang, X., Zhao, J. y LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification*. NeurIPS 2015 |

### Cómo se obtiene

`src/pipeline/ag_news.py` decide solo:

1. Si `data/ag_news/ag_news_train.csv` y `data/ag_news/ag_news_test.csv` existen (los de la cátedra), usa esos.
2. Si no, descarga la versión pública a `data/ag_news/`, fijada a un commit (`555590d`) y verificada con SHA-256. Une título y bajada en `text`, que es como vienen los CSV de la cátedra.

| Archivo público | SHA-256 |
|---|---|
| `train.csv` | `76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e` |
| `test.csv` | `521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699` |

El corpus preprocesado (regex + spaCy + deduplicación del train) se guarda en `data/ag_news/procesado/` para no repetir los ~5 minutos de spaCy. Toda la carpeta `data/ag_news/` está en el `.gitignore`.
