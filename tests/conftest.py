"""Fixtures compartidas por las pruebas.

Las pruebas no usan el dataset real: entrenar sobre 129.000 interacciones haría que
la suite tarde minutos. En su lugar se genera un dataset sintético chico con una
estructura de preferencias conocida de antemano, lo que permite escribir aserciones
sobre el contenido de las recomendaciones y no solo sobre su forma.

Diseño del dataset sintético
----------------------------
Dos comunidades de gustos (shooters y estrategia), cada una con cinco títulos y
treinta usuarios. Cada usuario posee un subconjunto aleatorio de tres juegos de su
comunidad, de modo que los títulos de un mismo grupo co-ocurren de forma positiva:
esa correlación es exactamente la señal que el filtrado colaborativo tiene que
detectar.

Vale la pena señalar por qué *no* se usó el diseño más obvio, el de darle a cada
usuario todos los juegos de su grupo menos uno: con esa construcción, tener A y B
implica no tener C, y el modelo aprende correctamente una anti-correlación. Es
decir, el dataset diría lo contrario de lo que se quiere probar.

Unos pocos usuarios "puente" conectan ambas comunidades a través del quinto título
de cada grupo. Sin ninguna conexión, los dos bloques quedarían desacoplados y la
similitud entre grupos sería arbitraria; el puente se limita a un par de títulos
para que no contamine al resto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recomendador.als import ModeloALS  # noqa: E402
from recomendador.config import ConfiguracionALS  # noqa: E402
from recomendador.datos import construir_matriz  # noqa: E402
from recomendador.motor import MotorRecomendacion  # noqa: E402
from recomendador.popularidad import RecomendadorPopularidad  # noqa: E402

JUEGOS_SHOOTER = [f"Shooter {letra}" for letra in "ABCDE"]
JUEGOS_ESTRATEGIA = [f"Estrategia {letra}" for letra in "ABCDE"]

USUARIOS_POR_GRUPO = 30
JUEGOS_POR_USUARIO = 3
USUARIOS_PUENTE = 3
SEMILLA_FIXTURE = 7


@pytest.fixture(scope="session")
def interacciones_sinteticas() -> pd.DataFrame:
    """Construye un dataset chico con dos comunidades de gustos diferenciadas."""
    generador = np.random.default_rng(SEMILLA_FIXTURE)
    filas: list[tuple[int, str, bool, float]] = []

    for grupo, juegos in enumerate((JUEGOS_SHOOTER, JUEGOS_ESTRATEGIA)):
        for posicion in range(USUARIOS_POR_GRUPO):
            id_usuario = grupo * 100 + posicion + 1
            elegidos = generador.choice(juegos, size=JUEGOS_POR_USUARIO, replace=False)
            for juego in elegidos:
                filas.append((id_usuario, str(juego), True, 20.0))

    for indice in range(USUARIOS_PUENTE):
        id_puente = 900 + indice
        filas.append((id_puente, JUEGOS_SHOOTER[-1], True, 5.0))
        filas.append((id_puente, JUEGOS_ESTRATEGIA[-1], True, 5.0))

    return pd.DataFrame(
        filas, columns=["id_usuario", "titulo", "compro", "horas"]
    ).drop_duplicates(subset=["id_usuario", "titulo"], ignore_index=True)


@pytest.fixture(scope="session")
def config_prueba() -> ConfiguracionALS:
    """Configuración liviana, para que la suite corra en menos de un segundo."""
    return ConfiguracionALS(
        factores=16, regularizacion=1.0, iteraciones=30, alfa=2.0, semilla=7
    )


@pytest.fixture
def motor(interacciones_sinteticas, config_prueba) -> MotorRecomendacion:
    """Motor entrenado sobre el dataset sintético."""
    datos = construir_matriz(interacciones_sinteticas, config_prueba)
    modelo = ModeloALS(config_prueba).entrenar(datos.matriz, verboso=False)
    popularidad = RecomendadorPopularidad.entrenar(
        datos.interacciones, datos.catalogo, peso_previo=1.0
    )
    return MotorRecomendacion.desde_entrenamiento(datos, modelo, popularidad)
