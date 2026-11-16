"""Factorización de matrices por mínimos cuadrados alternados (ALS) implícito.

Implementa el algoritmo de Hu, Koren y Volinsky (2008), "Collaborative Filtering
for Implicit Feedback Datasets", que es la variante de ALS pensada para datos
donde no hay puntuaciones explícitas sino señales de uso.

Idea del modelo
---------------
Cada usuario ``u`` y cada ítem ``i`` se representan con un vector latente de ``f``
dimensiones (``x_u`` e ``y_i``). La preferencia estimada es el producto interno:

    p̂_ui = x_u · y_i

A diferencia del ALS clásico, acá el error se calcula sobre *todas* las celdas de
la matriz, no solo sobre las observadas, y cada una pesa según su confianza:

    min Σ_{u,i} c_ui (p_ui − x_u·y_i)² + λ (Σ_u ||x_u||² + Σ_i ||y_i||²)

donde ``p_ui`` vale 1 si hubo interacción y 0 si no, y ``c_ui`` es el peso que
calcula :func:`recomendador.datos.calcular_confianza`. Que las celdas vacías
participen con peso 1 es justamente lo que permite interpretarlas como "probable
desinterés" en lugar de "dato faltante".

Con los factores de un lado fijos, el problema del otro lado es un mínimo cuadrado
regularizado con solución cerrada:

    x_u = (YᵀY + Yᵀ(C_u − I)Y + λI)⁻¹ Yᵀ C_u p(u)

El truco de Hu et al. está en precalcular ``YᵀY`` una sola vez por iteración: como
``C_u − I`` es nula en todas las celdas no observadas, el término restante solo
recorre los pocos ítems con los que el usuario interactuó, y el costo pasa a ser
lineal en la cantidad de interacciones en lugar de en el tamaño de la matriz.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .config import CONFIG_ALS, ConfiguracionALS


def _resolver_factor(
    factores_fijos: np.ndarray,
    gramiana: np.ndarray,
    indices: np.ndarray,
    confianzas: np.ndarray,
    regularizacion: float,
) -> np.ndarray:
    """Resuelve el vector latente de una fila (un usuario o un ítem).

    Aplica la ecuación cerrada de Hu et al. aprovechando que solo las celdas
    observadas modifican el sistema respecto de la gramiana precalculada.

    Args:
        factores_fijos: matriz (n, f) con los factores del lado que queda fijo.
        gramiana: producto ``YᵀY`` (f, f) precalculado para esta iteración.
        indices: posiciones de las celdas observadas de esta fila.
        confianzas: pesos ``c_ui`` de esas celdas.
        regularizacion: coeficiente lambda.

    Returns:
        Vector latente de dimensión f. Si la fila no tiene interacciones devuelve
        ceros, que es la solución exacta del sistema regularizado en ese caso.
    """
    dimension = factores_fijos.shape[1]
    if indices.size == 0:
        return np.zeros(dimension, dtype=np.float64)

    factores_observados = factores_fijos[indices]  # (n_obs, f)

    # A = YᵀY + Yᵀ(C_u − I)Y + λI
    matriz_sistema = gramiana + (
        factores_observados.T * (confianzas - 1.0)
    ) @ factores_observados
    matriz_sistema[np.diag_indices(dimension)] += regularizacion

    # b = Yᵀ C_u p(u); como p(u)=1 en las celdas observadas y 0 en el resto,
    # se reduce a la suma de los factores observados ponderada por su confianza.
    termino_independiente = factores_observados.T @ confianzas

    return np.linalg.solve(matriz_sistema, termino_independiente)


def _actualizar_lado(
    matriz: csr_matrix,
    factores_a_actualizar: np.ndarray,
    factores_fijos: np.ndarray,
    regularizacion: float,
) -> None:
    """Recalcula in-place todos los factores de un lado de la factorización.

    Args:
        matriz: matriz de confianza en formato CSR, orientada de modo que cada fila
            corresponda a una entidad del lado que se actualiza.
        factores_a_actualizar: matriz (n, f) que se sobrescribe.
        factores_fijos: matriz (m, f) del lado opuesto, que permanece constante.
        regularizacion: coeficiente lambda.
    """
    gramiana = factores_fijos.T @ factores_fijos
    indptr, indices, datos = matriz.indptr, matriz.indices, matriz.data

    for fila in range(factores_a_actualizar.shape[0]):
        desde, hasta = indptr[fila], indptr[fila + 1]
        factores_a_actualizar[fila] = _resolver_factor(
            factores_fijos,
            gramiana,
            indices[desde:hasta],
            datos[desde:hasta],
            regularizacion,
        )


class ModeloALS:
    """Modelo de factorización de matrices para feedback implícito.

    Atributos:
        config: hiperparámetros usados en el entrenamiento.
        factores_usuario: matriz (usuarios, f) aprendida.
        factores_juego: matriz (juegos, f) aprendida.
    """

    def __init__(self, config: ConfiguracionALS = CONFIG_ALS) -> None:
        self.config = config
        self.factores_usuario: np.ndarray | None = None
        self.factores_juego: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #

    def entrenar(self, matriz: csr_matrix, verboso: bool = True) -> "ModeloALS":
        """Ajusta los factores latentes alternando entre usuarios e ítems.

        Args:
            matriz: matriz dispersa usuarios x juegos con pesos de confianza.
            verboso: si es True, informa el avance por iteración.

        Returns:
            El propio modelo, para poder encadenar llamadas.
        """
        generador = np.random.default_rng(self.config.semilla)
        cantidad_usuarios, cantidad_juegos = matriz.shape

        # Inicialización aleatoria pequeña: valores grandes hacen que las primeras
        # iteraciones sobrecorrijan y el ajuste tarde más en estabilizarse.
        escala = 0.01
        self.factores_usuario = (
            generador.standard_normal((cantidad_usuarios, self.config.factores))
            * escala
        )
        self.factores_juego = (
            generador.standard_normal((cantidad_juegos, self.config.factores)) * escala
        )

        matriz_usuarios = matriz.tocsr()
        matriz_juegos = matriz.T.tocsr()

        for iteracion in range(1, self.config.iteraciones + 1):
            _actualizar_lado(
                matriz_usuarios,
                self.factores_usuario,
                self.factores_juego,
                self.config.regularizacion,
            )
            _actualizar_lado(
                matriz_juegos,
                self.factores_juego,
                self.factores_usuario,
                self.config.regularizacion,
            )
            if verboso:
                perdida = self.perdida(matriz_usuarios)
                print(
                    f"  iteración {iteracion:>2}/{self.config.iteraciones} "
                    f"· pérdida = {perdida:,.2f}"
                )

        # La última actualización del bucle es la de los ítems, así que los factores
        # de usuario quedarían desfasados respecto de Y. Se los recalcula una vez más
        # para que sean exactamente la solución óptima dado el Y final. Esto importa
        # más allá de la prolijidad: garantiza que el fold-in de un usuario existente
        # reproduzca su factor entrenado, que es la premisa de actualizar perfiles en
        # caliente sin reentrenar.
        _actualizar_lado(
            matriz_usuarios,
            self.factores_usuario,
            self.factores_juego,
            self.config.regularizacion,
        )

        return self

    def perdida(self, matriz: csr_matrix) -> float:
        """Calcula la función objetivo completa, incluidas las celdas no observadas.

        Se apoya en la descomposición que permite evaluar el término de las celdas
        vacías sin recorrerlas una por una::

            Σ_{u,i} c_ui (p_ui − x_u·y_i)²
              = Σ_{u,i} (x_u·y_i)²                        [todas las celdas, c=1]
              + Σ_{observadas} [ c_ui (1 − x_u·y_i)² − (x_u·y_i)² ]

        y el primer término es la traza de ``(XᵀX)(YᵀY)``, de costo O(f²).

        Args:
            matriz: matriz de confianza en formato CSR orientada por usuario.

        Returns:
            Valor de la función objetivo, con la penalización L2 incluida.
        """
        self._verificar_entrenado()
        factores_usuario = self.factores_usuario
        factores_juego = self.factores_juego

        gramiana_usuarios = factores_usuario.T @ factores_usuario
        gramiana_juegos = factores_juego.T @ factores_juego
        total = float(np.sum(gramiana_usuarios * gramiana_juegos))

        indptr, indices, datos = matriz.indptr, matriz.indices, matriz.data
        for usuario in range(factores_usuario.shape[0]):
            desde, hasta = indptr[usuario], indptr[usuario + 1]
            if desde == hasta:
                continue
            predicciones = factores_juego[indices[desde:hasta]] @ factores_usuario[usuario]
            confianzas = datos[desde:hasta]
            total += float(
                np.sum(confianzas * (1.0 - predicciones) ** 2 - predicciones**2)
            )

        penalizacion = self.config.regularizacion * (
            float(np.sum(factores_usuario**2)) + float(np.sum(factores_juego**2))
        )
        return total + penalizacion

    # ------------------------------------------------------------------ #
    # Inferencia
    # ------------------------------------------------------------------ #

    def calcular_factor_usuario(
        self, indices_juegos: np.ndarray, confianzas: np.ndarray
    ) -> np.ndarray:
        """Obtiene el vector latente de un usuario sin reentrenar el modelo.

        Es la técnica de *fold-in*: se mantienen fijos los factores de los ítems y
        se resuelve únicamente la ecuación del usuario. Permite cumplir dos puntos
        del enunciado sin recalcular la factorización completa:

        - actualizar las recomendaciones de un usuario ante nuevas interacciones;
        - personalizar a un usuario nuevo apenas registra su primera interacción.

        Es válido porque el enunciado aclara que no se incorporarán títulos nuevos,
        de modo que el espacio latente de los ítems no se mueve.

        Args:
            indices_juegos: columnas de los juegos con los que interactuó.
            confianzas: pesos de confianza de esas interacciones.

        Returns:
            Vector latente del usuario, de dimensión f.
        """
        self._verificar_entrenado()
        gramiana = self.factores_juego.T @ self.factores_juego
        return _resolver_factor(
            self.factores_juego,
            gramiana,
            np.asarray(indices_juegos, dtype=np.int64),
            np.asarray(confianzas, dtype=np.float64),
            self.config.regularizacion,
        )

    def puntuar(self, factor_usuario: np.ndarray) -> np.ndarray:
        """Calcula la preferencia estimada del usuario sobre todo el catálogo.

        Args:
            factor_usuario: vector latente del usuario.

        Returns:
            Vector de puntajes, uno por juego, en el orden del catálogo.
        """
        self._verificar_entrenado()
        return self.factores_juego @ np.asarray(factor_usuario, dtype=np.float64)

    def juegos_similares(
        self,
        indice_juego: int,
        k: int = 10,
        candidatos: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Devuelve los k juegos más parecidos a uno dado.

        La similitud es el coseno entre los vectores latentes. Sirve tanto para
        validar cualitativamente el modelo (los vecinos de un juego conocido
        deberían resultar razonables) como para recomendar por similitud de ítem.

        Args:
            indice_juego: columna del juego de referencia.
            k: cantidad de vecinos a devolver.
            candidatos: máscara booleana sobre el catálogo que limita qué juegos
                pueden aparecer como vecinos. Si es None, se considera todo el
                catálogo. Sirve para excluir ítems cuyo vector latente no tiene
                suficiente evidencia detrás.

        Returns:
            Lista de pares (índice de juego, similitud), de mayor a menor, sin
            incluir al juego de referencia.
        """
        self._verificar_entrenado()
        normas = np.linalg.norm(self.factores_juego, axis=1)
        normas[normas == 0] = 1e-10
        referencia = self.factores_juego[indice_juego] / normas[indice_juego]
        similitudes = (self.factores_juego @ referencia) / normas

        if candidatos is not None:
            similitudes = np.where(candidatos, similitudes, -np.inf)
        similitudes[indice_juego] = -np.inf

        disponibles = int(np.isfinite(similitudes).sum())
        k = min(k, disponibles)
        if k <= 0:
            return []

        mejores = np.argpartition(-similitudes, k - 1)[:k]
        mejores = mejores[np.argsort(-similitudes[mejores])]
        return [(int(i), float(similitudes[i])) for i in mejores]

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #

    def _verificar_entrenado(self) -> None:
        if self.factores_usuario is None or self.factores_juego is None:
            raise RuntimeError(
                "El modelo no está entrenado. Ejecutá entrenar() o cargá un modelo "
                "persistido antes de pedir recomendaciones."
            )
