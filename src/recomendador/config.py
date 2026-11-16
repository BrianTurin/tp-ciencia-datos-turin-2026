"""Parámetros de configuración del motor de recomendación.

Se centralizan acá todos los hiperparámetros y rutas para que los scripts, la API
y las pruebas trabajen siempre contra los mismos valores, y para poder ajustarlos
desde variables de entorno sin tocar el código.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Raíz del proyecto (dos niveles por encima de este archivo: src/recomendador/ -> src/ -> raíz)
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_DATASET = Path(
    os.getenv("RUTA_DATASET", RAIZ_PROYECTO / "datos" / "dataset-videojuegos.csv")
)
DIRECTORIO_MODELO = Path(
    os.getenv("DIRECTORIO_MODELO", RAIZ_PROYECTO / "modelo")
)
RUTA_MODELO = DIRECTORIO_MODELO / "modelo_als.npz"

# Nombres de las columnas del dataset original, que viene sin encabezado.
COLUMNAS_DATASET = ["id_usuario", "titulo", "comportamiento", "valor", "descarte"]

COMPORTAMIENTO_COMPRA = "purchase"
COMPORTAMIENTO_JUEGO = "play"


@dataclass(frozen=True)
class ConfiguracionALS:
    """Hiperparámetros del modelo de factorización de matrices.

    El modelo implementado es ALS para *feedback implícito* (Hu, Koren y Volinsky,
    2008), que es el caso de este dataset: no hay puntuaciones explícitas, solo
    compras y horas jugadas. La preferencia se modela como binaria (el usuario
    interactuó o no) y la intensidad de la interacción se traslada a un peso de
    confianza.

    Atributos:
        factores: dimensión del espacio latente (cantidad de factores ocultos).
        regularizacion: coeficiente lambda de la penalización L2 (evita sobreajuste
            en usuarios con muy pocas interacciones).
        iteraciones: cantidad de pasadas completas del algoritmo alternante.
        alfa: escala de confianza. Cuanto más alto, más peso relativo tienen las
            interacciones intensas (muchas horas) frente a las débiles.
        epsilon_horas: constante de la transformación logarítmica de horas. Amortigua
            la cola larguísima de la distribución de horas jugadas.
        peso_compra: confianza base que aporta el solo hecho de haber comprado el
            juego, aun cuando el usuario nunca lo haya jugado.
        semilla: semilla del generador aleatorio, para que el entrenamiento sea
            reproducible.

    Los valores por defecto salieron de una búsqueda en grilla sobre el conjunto de
    validación (ver ``scripts/evaluar.py``). Dos observaciones de esa búsqueda:

    - La cantidad de factores es lo que más mueve la aguja, con rendimientos
      decrecientes: 64 -> 128 mejora el MAP@10 de forma clara, y 128 -> 192 casi no
      aporta pero duplica el tiempo de entrenamiento.
    - ``alfa`` regula un compromiso entre precisión y cobertura de catálogo. Valores
      altos concentran las recomendaciones en menos títulos; valores bajos mejoran
      el MAP pero achican la diversidad. Se eligió un punto intermedio, porque una
      lista precisa pero siempre igual incumple los criterios de novedad y
      diversidad que plantea la bibliografía.
    """

    factores: int = 128
    regularizacion: float = 1.0
    iteraciones: int = 15
    alfa: float = 2.0
    epsilon_horas: float = 1.0
    peso_compra: float = 1.0
    semilla: int = 42


@dataclass(frozen=True)
class ConfiguracionEvaluacion:
    """Parámetros del protocolo de validación offline.

    Atributos:
        k: longitud de la lista de recomendaciones que se evalúa.
        proporcion_prueba: fracción de ítems de cada usuario que se reserva para test.
        minimo_interacciones: cantidad mínima de ítems que debe tener un usuario para
            entrar en la evaluación (con menos, el split deja el entrenamiento vacío).
        semilla: semilla del particionado.
    """

    k: int = 10
    proporcion_prueba: float = 0.2
    minimo_interacciones: int = 5
    semilla: int = 42


CONFIG_ALS = ConfiguracionALS()
CONFIG_EVALUACION = ConfiguracionEvaluacion()
