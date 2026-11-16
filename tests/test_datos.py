"""Pruebas del preprocesamiento de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recomendador.config import ConfiguracionALS
from recomendador.datos import (
    calcular_confianza,
    consolidar_interacciones,
    construir_matriz,
)


def test_consolidar_une_compra_y_juego_en_una_fila():
    """Los dos eventos de un mismo par usuario-juego colapsan en un único registro."""
    crudos = pd.DataFrame(
        [
            (1, "Portal 2", "purchase", 1.0),
            (1, "Portal 2", "play", 13.6),
        ],
        columns=["id_usuario", "titulo", "comportamiento", "valor"],
    )

    resultado = consolidar_interacciones(crudos)

    assert len(resultado) == 1
    assert bool(resultado.iloc[0]["compro"]) is True
    assert resultado.iloc[0]["horas"] == pytest.approx(13.6)


def test_consolidar_toma_el_maximo_de_horas_en_eventos_repetidos():
    """El campo de horas es acumulado, así que los duplicados no se suman."""
    crudos = pd.DataFrame(
        [
            (1, "Dota 2", "purchase", 1.0),
            (1, "Dota 2", "play", 100.0),
            (1, "Dota 2", "play", 250.0),
        ],
        columns=["id_usuario", "titulo", "comportamiento", "valor"],
    )

    resultado = consolidar_interacciones(crudos)

    assert len(resultado) == 1
    assert resultado.iloc[0]["horas"] == pytest.approx(250.0)


def test_consolidar_conserva_los_comprados_sin_jugar():
    """Comprar sin jugar sigue siendo una señal y no debe perderse."""
    crudos = pd.DataFrame(
        [(1, "Spore", "purchase", 1.0)],
        columns=["id_usuario", "titulo", "comportamiento", "valor"],
    )

    resultado = consolidar_interacciones(crudos)

    assert len(resultado) == 1
    assert resultado.iloc[0]["horas"] == 0.0
    assert bool(resultado.iloc[0]["compro"]) is True


def test_confianza_crece_con_las_horas_pero_de_forma_comprimida():
    """La transformación logarítmica evita que los outliers dominen el ajuste."""
    config = ConfiguracionALS()
    confianzas = calcular_confianza(
        np.array([True, True, True]), np.array([0.0, 10.0, 1000.0]), config
    )

    assert confianzas[0] < confianzas[1] < confianzas[2]
    # Mil horas valen mucho menos que cien veces diez horas.
    assert confianzas[2] < 100 * confianzas[1]


def test_confianza_nunca_es_menor_a_uno():
    """Toda interacción observada debe pesar algo en el ajuste."""
    confianzas = calcular_confianza(np.array([False]), np.array([0.0]))
    assert confianzas[0] >= 1.0


def test_construir_matriz_respeta_dimensiones_y_celdas(interacciones_sinteticas):
    """La matriz tiene una celda observada por par usuario-juego, y ninguna más."""
    datos = construir_matriz(interacciones_sinteticas)

    assert datos.matriz.shape == (
        datos.catalogo.cantidad_usuarios,
        datos.catalogo.cantidad_juegos,
    )
    assert datos.matriz.nnz == len(interacciones_sinteticas)
    assert datos.matriz.data.min() >= 1.0


def test_catalogo_traduce_en_ambos_sentidos(interacciones_sinteticas):
    """El catálogo debe ser una biyección entre identificadores e índices."""
    datos = construir_matriz(interacciones_sinteticas)
    catalogo = datos.catalogo

    indice = catalogo.indice_de_juego("Shooter A")
    assert indice is not None
    assert catalogo.titulo_de_indice(indice) == "Shooter A"
    assert catalogo.indice_de_juego("Juego Inexistente") is None
    assert catalogo.indice_de_usuario(999999) is None
