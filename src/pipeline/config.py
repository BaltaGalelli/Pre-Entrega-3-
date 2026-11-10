"""
Configuración central del experimento.

Todo lo que se puede querer cambiar entre corridas vive acá, en un solo lugar,
para que cada resultado del README se pueda reproducir sabiendo qué valores se usaron.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DIR_DATOS = RAIZ / "data" / "banking77"
DIR_AG_NEWS = RAIZ / "data" / "ag_news"         # corpus del proyecto desde la Pre-entrega 2
DIR_MODELOS = RAIZ / "models"
DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figures"


@dataclass
class Config:
    semilla: int = 42
    # Partición: el test oficial de Banking77 queda intacto; la validación sale del train (estratificada)
    tamano_validacion: float = 0.15
    # Representación del texto: TF-IDF de palabras (unigramas y bigramas)
    ngramas_max: int = 2
    frecuencia_minima: int = 2         # un término tiene que aparecer en al menos 2 consultas del train
    # Modelo
    arquitectura: str = "mlp"          # "lineal" o "mlp"
    capas_ocultas: tuple = (256,)
    dropout: float = 0.3
    # Entrenamiento
    learning_rate: float = 1e-3
    batch_size: int = 64
    epocas: int = 30
    paciencia: int = 5                 # early stopping: épocas sin mejorar la loss de validación

    def a_dict(self):
        d = asdict(self)
        d["capas_ocultas"] = list(self.capas_ocultas)
        return d
