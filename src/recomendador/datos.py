"""Carga y preparación de los datos de interacción usuario-juego.

El dataset original tiene una fila por evento, con dos tipos de comportamiento:

    id_usuario, titulo, "purchase", 1.0,   0
    id_usuario, titulo, "play",     273.0, 0

Este módulo lo convierte en lo que necesita el modelo: una matriz dispersa
usuario x juego donde cada celda observada guarda un *peso de confianza*, no una
puntuación. La preferencia en sí es binaria (hubo interacción o no hubo), y las
horas jugadas solo modulan cuánta confianza le damos a esa preferencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import (
    COLUMNAS_DATASET,
    COMPORTAMIENTO_COMPRA,
    COMPORTAMIENTO_JUEGO,
    CONFIG_ALS,
    ConfiguracionALS,
)


@dataclass
class Catalogo:
    """Diccionario bidireccional entre identificadores y posiciones de la matriz.

    El modelo trabaja con índices contiguos (0..n-1), pero el mundo exterior usa
    identificadores de usuario y títulos de juego. Esta clase traduce entre ambos.

    Atributos:
        usuarios: ids de usuario ordenados según su fila en la matriz.
        juegos: títulos ordenados según su columna en la matriz.
    """

    usuarios: np.ndarray
    juegos: np.ndarray

    def __post_init__(self) -> None:
        self._indice_usuario = {u: i for i, u in enumerate(self.usuarios)}
        self._indice_juego = {j: i for i, j in enumerate(self.juegos)}

    @property
    def cantidad_usuarios(self) -> int:
        return len(self.usuarios)

    @property
    def cantidad_juegos(self) -> int:
        return len(self.juegos)

    def indice_de_usuario(self, id_usuario: int) -> int | None:
        """Devuelve la fila del usuario, o None si no estaba en el entrenamiento."""
        return self._indice_usuario.get(id_usuario)

    def indice_de_juego(self, titulo: str) -> int | None:
        """Devuelve la columna del juego, o None si el título no existe."""
        return self._indice_juego.get(titulo)

    def titulo_de_indice(self, indice: int) -> str:
        return str(self.juegos[indice])

    def usuario_de_indice(self, indice: int) -> int:
        return int(self.usuarios[indice])


@dataclass
class DatosInteraccion:
    """Resultado del preprocesamiento, listo para entrenar.

    Atributos:
        matriz: matriz dispersa usuarios x juegos con los pesos de confianza.
        catalogo: traductor entre ids/títulos e índices de la matriz.
        interacciones: tabla consolidada (una fila por par usuario-juego), que se
            conserva para inspección, evaluación y cálculo de popularidad.
    """

    matriz: csr_matrix
    catalogo: Catalogo
    interacciones: pd.DataFrame


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """Lee el CSV crudo y normaliza sus columnas.

    El archivo no trae encabezado y su última columna es una constante en 0 que el
    propio enunciado describe como "sin datos", así que se descarta.

    Args:
        ruta: ubicación del CSV de interacciones.

    Returns:
        DataFrame con las columnas id_usuario, titulo, comportamiento y valor.

    Raises:
        FileNotFoundError: si la ruta no existe.
    """
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No se encontró el dataset en {ruta}")

    datos = pd.read_csv(
        ruta,
        header=None,
        names=COLUMNAS_DATASET,
        dtype={"id_usuario": np.int64, "titulo": str, "comportamiento": str},
    )
    datos = datos.drop(columns=["descarte"])
    datos["titulo"] = datos["titulo"].str.strip()
    datos["comportamiento"] = datos["comportamiento"].str.strip().str.lower()
    return datos


def consolidar_interacciones(datos: pd.DataFrame) -> pd.DataFrame:
    """Colapsa los eventos sueltos en una fila por par (usuario, juego).

    Un mismo par puede aparecer varias veces: una fila de compra, una de juego y,
    en el dataset original, algunos eventos duplicados. Se resuelve así:

    - ``compro``: True si existe al menos un evento de compra.
    - ``horas``: máximo de horas reportadas. Se toma el máximo y no la suma porque
      el campo del dataset es acumulado, de modo que sumarlo contaría dos veces las
      mismas horas en los registros repetidos.

    Args:
        datos: salida de :func:`cargar_dataset`.

    Returns:
        DataFrame con columnas id_usuario, titulo, compro y horas.
    """
    compras = (
        datos[datos["comportamiento"] == COMPORTAMIENTO_COMPRA]
        .groupby(["id_usuario", "titulo"], as_index=False)
        .size()
        .rename(columns={"size": "eventos_compra"})
    )

    juego = (
        datos[datos["comportamiento"] == COMPORTAMIENTO_JUEGO]
        .groupby(["id_usuario", "titulo"], as_index=False)["valor"]
        .max()
        .rename(columns={"valor": "horas"})
    )

    interacciones = compras.merge(juego, on=["id_usuario", "titulo"], how="outer")
    interacciones["horas"] = interacciones["horas"].fillna(0.0)
    interacciones["compro"] = interacciones["eventos_compra"].notna()
    interacciones = interacciones.drop(columns=["eventos_compra"])

    return interacciones[["id_usuario", "titulo", "compro", "horas"]]


def calcular_confianza(
    compro: np.ndarray, horas: np.ndarray, config: ConfiguracionALS = CONFIG_ALS
) -> np.ndarray:
    """Traduce compra y horas jugadas a un peso de confianza.

    Sigue la formulación de Hu, Koren y Volinsky (2008) para feedback implícito,
    en su variante logarítmica:

        c_ui = 1 + peso_compra * [compró] + alfa * log(1 + horas / epsilon)

    La lógica detrás de cada término:

    - El ``1`` es la confianza mínima: toda interacción observada pesa algo.
    - ``peso_compra`` recoge que comprar un juego ya es una señal de preferencia,
      aunque el usuario nunca lo haya abierto.
    - El término logarítmico evita que los usuarios con miles de horas en un único
      título dominen la factorización. La distribución de horas es de cola muy
      larga (mediana 4,5 h, máximo 11.754 h), así que sin la compresión unas pocas
      celdas concentrarían casi todo el peso del error.

    Args:
        compro: vector booleano indicando si hubo compra.
        horas: vector de horas jugadas (0 si nunca jugó).
        config: hiperparámetros del modelo.

    Returns:
        Vector de confianzas, siempre >= 1.
    """
    horas = np.maximum(np.asarray(horas, dtype=np.float64), 0.0)
    termino_horas = config.alfa * np.log1p(horas / config.epsilon_horas)
    termino_compra = config.peso_compra * np.asarray(compro, dtype=np.float64)
    return 1.0 + termino_compra + termino_horas


def construir_matriz(
    interacciones: pd.DataFrame, config: ConfiguracionALS = CONFIG_ALS
) -> DatosInteraccion:
    """Arma la matriz dispersa de confianza a partir de las interacciones consolidadas.

    Args:
        interacciones: salida de :func:`consolidar_interacciones`.
        config: hiperparámetros del modelo.

    Returns:
        Objeto :class:`DatosInteraccion` con la matriz, el catálogo y la tabla.
    """
    usuarios = np.sort(interacciones["id_usuario"].unique())
    juegos = np.sort(interacciones["titulo"].unique())
    catalogo = Catalogo(usuarios=usuarios, juegos=juegos)

    filas = interacciones["id_usuario"].map(catalogo._indice_usuario).to_numpy()
    columnas = interacciones["titulo"].map(catalogo._indice_juego).to_numpy()
    valores = calcular_confianza(
        interacciones["compro"].to_numpy(), interacciones["horas"].to_numpy(), config
    )

    matriz = csr_matrix(
        (valores, (filas, columnas)),
        shape=(catalogo.cantidad_usuarios, catalogo.cantidad_juegos),
        dtype=np.float64,
    )

    return DatosInteraccion(
        matriz=matriz, catalogo=catalogo, interacciones=interacciones
    )


def preparar_datos(
    ruta: Path, config: ConfiguracionALS = CONFIG_ALS
) -> DatosInteraccion:
    """Ejecuta el preprocesamiento completo: leer, consolidar y vectorizar.

    Args:
        ruta: ubicación del CSV de interacciones.
        config: hiperparámetros del modelo.

    Returns:
        Objeto :class:`DatosInteraccion` listo para entrenar.
    """
    crudos = cargar_dataset(ruta)
    interacciones = consolidar_interacciones(crudos)
    return construir_matriz(interacciones, config)


def resumen_dataset(datos: pd.DataFrame) -> dict[str, float | int]:
    """Calcula estadísticos descriptivos del dataset crudo.

    Se usa para el entendimiento de los datos y para dejar registro de sobre qué
    versión del dataset se entrenó un modelo.

    Args:
        datos: salida de :func:`cargar_dataset`.

    Returns:
        Diccionario con métricas básicas de volumen, dispersión y actividad.
    """
    interacciones = consolidar_interacciones(datos)
    cantidad_usuarios = int(interacciones["id_usuario"].nunique())
    cantidad_juegos = int(interacciones["titulo"].nunique())
    celdas = cantidad_usuarios * cantidad_juegos
    jugados = interacciones[interacciones["horas"] > 0]

    return {
        "eventos": int(len(datos)),
        "pares_usuario_juego": int(len(interacciones)),
        "usuarios": cantidad_usuarios,
        "juegos": cantidad_juegos,
        "densidad": float(len(interacciones) / celdas) if celdas else 0.0,
        "juegos_por_usuario_promedio": float(len(interacciones) / cantidad_usuarios),
        "juegos_por_usuario_mediana": float(
            interacciones.groupby("id_usuario").size().median()
        ),
        "proporcion_comprados_sin_jugar": float(
            (interacciones["horas"] == 0).mean()
        ),
        "horas_mediana": float(jugados["horas"].median()) if len(jugados) else 0.0,
        "horas_maximo": float(jugados["horas"].max()) if len(jugados) else 0.0,
    }
