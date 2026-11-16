"""Motor de recomendación de videojuegos basado en feedback implícito.

Módulos:
    config: hiperparámetros y rutas.
    datos: carga y preparación del dataset.
    als: factorización de matrices por mínimos cuadrados alternados.
    popularidad: ranking no personalizado para arranque en frío.
    motor: fachada que orquesta modelo, arranque en frío y estado.
    evaluacion: validación offline con métricas de ranking.
"""

from .als import ModeloALS
from .config import CONFIG_ALS, CONFIG_EVALUACION, ConfiguracionALS, ConfiguracionEvaluacion
from .datos import Catalogo, DatosInteraccion, preparar_datos, resumen_dataset
from .motor import Interaccion, MotorRecomendacion, Recomendacion
from .popularidad import RecomendadorPopularidad

__all__ = [
    "CONFIG_ALS",
    "CONFIG_EVALUACION",
    "Catalogo",
    "ConfiguracionALS",
    "ConfiguracionEvaluacion",
    "DatosInteraccion",
    "Interaccion",
    "ModeloALS",
    "MotorRecomendacion",
    "RecomendadorPopularidad",
    "Recomendacion",
    "preparar_datos",
    "resumen_dataset",
]
