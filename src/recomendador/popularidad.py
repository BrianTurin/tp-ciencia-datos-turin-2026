"""Recomendador no personalizado por popularidad, usado para arranque en frío.

El punto 3 del enunciado pide recomendar a usuarios nuevos, sin interacciones
previas. La factorización no puede resolverlo: sin ninguna celda observada, la
ecuación del usuario devuelve el vector nulo y todos los juegos empatan en cero.

La respuesta estándar es un ranking no personalizado. Acá se combinan dos señales,
porque cada una sola se desvía hacia un extremo:

- **cantidad de usuarios que compraron el juego**: mide alcance, pero premia a los
  free-to-play que todo el mundo instala y nadie juega;
- **horas medianas entre quienes lo jugaron**: mide involucramiento real, pero se
  dispara en juegos de nicho con dos o tres jugadores obsesivos.

El puntaje final es el producto de ambas, cada una comprimida con logaritmo, y con
un promedio bayesiano sobre las horas para castigar a los títulos con muestra chica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datos import Catalogo


@dataclass
class RecomendadorPopularidad:
    """Ranking global de juegos, independiente del usuario.

    Atributos:
        puntajes: vector de puntajes alineado con las columnas del catálogo.
        orden: índices de los juegos ordenados de mayor a menor puntaje.
        soporte: cantidad de usuarios distintos que poseen cada juego. No interviene
            en el ranking, pero se conserva porque es la medida de cuánta evidencia
            respalda al vector latente de cada ítem, y otras partes del sistema la
            usan para descartar títulos sin sustento estadístico.
    """

    puntajes: np.ndarray
    orden: np.ndarray
    soporte: np.ndarray

    @classmethod
    def entrenar(
        cls,
        interacciones: pd.DataFrame,
        catalogo: Catalogo,
        peso_previo: float = 20.0,
    ) -> "RecomendadorPopularidad":
        """Construye el ranking a partir de las interacciones consolidadas.

        Args:
            interacciones: tabla con columnas id_usuario, titulo, compro y horas.
            catalogo: traductor entre títulos e índices de columna.
            peso_previo: cantidad de jugadores "virtuales" con horas iguales a la
                mediana global que se suman a cada juego. Es el término previo del
                promedio bayesiano: cuanto más alto, más evidencia necesita un
                título poco jugado para escalar posiciones.

        Returns:
            Instancia entrenada de :class:`RecomendadorPopularidad`.
        """
        agregado = interacciones.groupby("titulo").agg(
            usuarios=("id_usuario", "nunique"),
            horas_totales=("horas", "sum"),
            jugadores=("horas", lambda serie: int((serie > 0).sum())),
        )

        jugadas = interacciones.loc[interacciones["horas"] > 0, "horas"]
        horas_globales = float(jugadas.median()) if len(jugadas) else 0.0

        # Promedio bayesiano: mezcla las horas del juego con la mediana global,
        # ponderando por cuántos jugadores reales lo respaldan.
        horas_ajustadas = (agregado["horas_totales"] + peso_previo * horas_globales) / (
            agregado["jugadores"] + peso_previo
        )

        puntaje_por_titulo = np.log1p(agregado["usuarios"]) * np.log1p(horas_ajustadas)

        puntajes = np.zeros(catalogo.cantidad_juegos, dtype=np.float64)
        soporte = np.zeros(catalogo.cantidad_juegos, dtype=np.int64)
        for titulo, puntaje in puntaje_por_titulo.items():
            indice = catalogo.indice_de_juego(titulo)
            if indice is not None:
                puntajes[indice] = float(puntaje)
                soporte[indice] = int(agregado.loc[titulo, "usuarios"])

        return cls(puntajes=puntajes, orden=np.argsort(-puntajes), soporte=soporte)

    def recomendar(
        self, k: int, excluir: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Devuelve los k juegos más populares.

        Args:
            k: cantidad de recomendaciones.
            excluir: índices de juegos que el usuario ya consumió. El enunciado y la
                bibliografía coinciden en que una recomendación debe ser novedosa,
                así que lo ya consumido no se vuelve a ofrecer.

        Returns:
            Lista de pares (índice de juego, puntaje), de mayor a menor.
        """
        excluir = excluir or set()
        resultado: list[tuple[int, float]] = []
        for indice in self.orden:
            indice = int(indice)
            if indice in excluir:
                continue
            resultado.append((indice, float(self.puntajes[indice])))
            if len(resultado) == k:
                break
        return resultado
