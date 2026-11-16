"""Validación offline del sistema de recomendación.

Protocolo
---------
El problema es de *ranking* (top-k), no de predicción de puntuaciones, así que no
se evalúa con error cuadrático sino con métricas de lista ordenada.

Para cada usuario con suficientes interacciones se reserva una fracción de sus
juegos como conjunto de prueba y se entrena con el resto. Después se le piden k
recomendaciones al modelo, que no vio esos juegos ocultos, y se mide cuántos
recuperó y en qué posiciones.

El particionado es por usuario y no global porque un corte global dejaría a muchos
usuarios sin ninguna interacción en entrenamiento, y el modelo no podría siquiera
construirles un perfil: estaríamos midiendo arranque en frío en lugar de calidad
de la personalización.

Métricas
--------
- **Precisión@k**: proporción de las k recomendaciones que el usuario efectivamente
  consumió. Penaliza el ruido en la lista.
- **Recall@k**: proporción de los juegos ocultos que aparecieron en la lista. Está
  acotado por k/|test|, así que su valor absoluto se lee con cuidado.
- **MAP@k**: promedia la precisión en cada acierto, de modo que premia colocar los
  aciertos arriba y no al final de la lista.
- **NDCG@k**: idem, con descuento logarítmico por posición.
- **Cobertura de catálogo**: qué porción del catálogo llega a recomendarse alguna
  vez. Un modelo con buena precisión pero cobertura ínfima está recomendando
  siempre los mismos éxitos, y eso incumple los criterios de novedad y diversidad.

Como referencia se evalúa también el recomendador por popularidad: en feedback
implícito con distribución tan sesgada, un modelo personalizado que no le gana a
la popularidad no está aportando nada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .als import ModeloALS
from .config import CONFIG_ALS, CONFIG_EVALUACION, ConfiguracionALS, ConfiguracionEvaluacion
from .datos import Catalogo, calcular_confianza
from .popularidad import RecomendadorPopularidad


@dataclass
class ParticionEvaluacion:
    """Resultado del particionado por usuario.

    Atributos:
        entrenamiento: interacciones usadas para ajustar el modelo.
        prueba: mapa de índice de usuario a conjunto de índices de juegos ocultos.
        catalogo: catálogo construido sobre el dataset completo, para que los
            índices sean los mismos en ambas particiones.
    """

    entrenamiento: pd.DataFrame
    prueba: dict[int, set[int]]
    catalogo: Catalogo


def particionar(
    interacciones: pd.DataFrame,
    catalogo: Catalogo,
    config: ConfiguracionEvaluacion = CONFIG_EVALUACION,
) -> ParticionEvaluacion:
    """Separa, para cada usuario, una fracción de sus juegos como conjunto de prueba.

    Args:
        interacciones: tabla consolidada con columnas id_usuario, titulo, compro, horas.
        catalogo: catálogo del dataset completo.
        config: parámetros del particionado.

    Returns:
        Objeto :class:`ParticionEvaluacion`.
    """
    generador = np.random.default_rng(config.semilla)
    indices_prueba: list[int] = []
    prueba: dict[int, set[int]] = {}

    for id_usuario, grupo in interacciones.groupby("id_usuario", sort=False):
        if len(grupo) < config.minimo_interacciones:
            continue

        cantidad = max(1, int(round(len(grupo) * config.proporcion_prueba)))
        # Nunca se oculta todo: el usuario tiene que conservar señal en entrenamiento.
        cantidad = min(cantidad, len(grupo) - 1)
        elegidos = generador.choice(grupo.index.to_numpy(), size=cantidad, replace=False)

        indice_usuario = catalogo.indice_de_usuario(int(id_usuario))
        if indice_usuario is None:
            continue

        columnas = {
            catalogo.indice_de_juego(titulo)
            for titulo in interacciones.loc[elegidos, "titulo"]
        }
        prueba[indice_usuario] = {c for c in columnas if c is not None}
        indices_prueba.extend(elegidos.tolist())

    entrenamiento = interacciones.drop(index=indices_prueba)
    return ParticionEvaluacion(
        entrenamiento=entrenamiento, prueba=prueba, catalogo=catalogo
    )


def _matriz_desde(
    interacciones: pd.DataFrame, catalogo: Catalogo, config: ConfiguracionALS
) -> csr_matrix:
    """Construye la matriz de confianza de una partición, respetando el catálogo global."""
    filas = interacciones["id_usuario"].map(catalogo._indice_usuario).to_numpy()
    columnas = interacciones["titulo"].map(catalogo._indice_juego).to_numpy()
    valores = calcular_confianza(
        interacciones["compro"].to_numpy(), interacciones["horas"].to_numpy(), config
    )
    return csr_matrix(
        (valores, (filas, columnas)),
        shape=(catalogo.cantidad_usuarios, catalogo.cantidad_juegos),
        dtype=np.float64,
    )


def _precision_promedio(aciertos: np.ndarray, relevantes: int) -> float:
    """Calcula la precisión promedio (AP) de una lista de aciertos booleanos.

    Args:
        aciertos: vector booleano, posición i indica si la i-ésima recomendación
            estaba en el conjunto de prueba.
        relevantes: cantidad total de ítems relevantes del usuario.

    Returns:
        Precisión promedio, normalizada por el mínimo entre k y los relevantes, de
        modo que el máximo alcanzable sea 1 aun cuando k < |relevantes|.
    """
    if relevantes == 0 or not aciertos.any():
        return 0.0
    posiciones = np.arange(1, len(aciertos) + 1)
    precision_acumulada = np.cumsum(aciertos) / posiciones
    return float(
        np.sum(precision_acumulada * aciertos) / min(len(aciertos), relevantes)
    )


def _ndcg(aciertos: np.ndarray, relevantes: int) -> float:
    """Calcula el NDCG de una lista con relevancia binaria.

    Args:
        aciertos: vector booleano de aciertos por posición.
        relevantes: cantidad total de ítems relevantes del usuario.

    Returns:
        NDCG en [0, 1].
    """
    if relevantes == 0:
        return 0.0
    descuento = 1.0 / np.log2(np.arange(2, len(aciertos) + 2))
    ganancia = float(np.sum(aciertos * descuento))
    ideal = float(np.sum(descuento[: min(len(aciertos), relevantes)]))
    return ganancia / ideal if ideal > 0 else 0.0


def _metricas_de_listas(
    listas: dict[int, list[int]], prueba: dict[int, set[int]], cantidad_juegos: int, k: int
) -> dict[str, float]:
    """Agrega las métricas de ranking sobre todos los usuarios evaluados.

    Args:
        listas: recomendaciones por índice de usuario.
        prueba: ítems ocultos por índice de usuario.
        cantidad_juegos: tamaño del catálogo, para la cobertura.
        k: longitud de las listas.

    Returns:
        Diccionario con las métricas promediadas.
    """
    precisiones, recalls, aps, ndcgs = [], [], [], []
    recomendados_global: set[int] = set()

    for indice_usuario, recomendados in listas.items():
        relevantes = prueba[indice_usuario]
        if not relevantes:
            continue
        recomendados_global.update(recomendados)
        aciertos = np.array([r in relevantes for r in recomendados], dtype=bool)

        precisiones.append(aciertos.sum() / k)
        recalls.append(aciertos.sum() / len(relevantes))
        aps.append(_precision_promedio(aciertos, len(relevantes)))
        ndcgs.append(_ndcg(aciertos, len(relevantes)))

    return {
        f"precision@{k}": float(np.mean(precisiones)) if precisiones else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"map@{k}": float(np.mean(aps)) if aps else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "cobertura_catalogo": len(recomendados_global) / cantidad_juegos,
        "usuarios_evaluados": len(precisiones),
    }


def evaluar(
    interacciones: pd.DataFrame,
    catalogo: Catalogo,
    config_als: ConfiguracionALS = CONFIG_ALS,
    config_evaluacion: ConfiguracionEvaluacion = CONFIG_EVALUACION,
    verboso: bool = True,
) -> dict[str, dict[str, float]]:
    """Entrena sobre la partición de entrenamiento y mide contra la de prueba.

    Args:
        interacciones: tabla consolidada completa.
        catalogo: catálogo del dataset completo.
        config_als: hiperparámetros del modelo.
        config_evaluacion: parámetros del protocolo de validación.
        verboso: si es True, informa el avance.

    Returns:
        Diccionario con dos entradas, ``factorizacion`` y ``popularidad``, cada una
        con sus métricas.
    """
    k = config_evaluacion.k
    particion = particionar(interacciones, catalogo, config_evaluacion)

    if verboso:
        print(
            f"  usuarios en evaluación: {len(particion.prueba):,} · "
            f"interacciones de entrenamiento: {len(particion.entrenamiento):,}"
        )

    matriz = _matriz_desde(particion.entrenamiento, catalogo, config_als)
    modelo = ModeloALS(config_als).entrenar(matriz, verboso=verboso)
    popularidad = RecomendadorPopularidad.entrenar(particion.entrenamiento, catalogo)

    matriz_csr = matriz.tocsr()
    listas_modelo: dict[int, list[int]] = {}
    listas_popularidad: dict[int, list[int]] = {}

    for indice_usuario in particion.prueba:
        consumidos = set(
            matriz_csr.indices[
                matriz_csr.indptr[indice_usuario] : matriz_csr.indptr[indice_usuario + 1]
            ].tolist()
        )

        puntajes = modelo.puntuar(modelo.factores_usuario[indice_usuario])
        if consumidos:
            puntajes = puntajes.copy()
            puntajes[list(consumidos)] = -np.inf
        mejores = np.argpartition(-puntajes, k)[:k]
        listas_modelo[indice_usuario] = [
            int(i) for i in mejores[np.argsort(-puntajes[mejores])]
        ]

        listas_popularidad[indice_usuario] = [
            indice for indice, _ in popularidad.recomendar(k, consumidos)
        ]

    return {
        "factorizacion": _metricas_de_listas(
            listas_modelo, particion.prueba, catalogo.cantidad_juegos, k
        ),
        "popularidad": _metricas_de_listas(
            listas_popularidad, particion.prueba, catalogo.cantidad_juegos, k
        ),
    }
