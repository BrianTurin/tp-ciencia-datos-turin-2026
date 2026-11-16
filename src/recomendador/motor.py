"""Motor de recomendación: orquesta el modelo, el arranque en frío y el estado.

Esta es la capa que consume la API. Es deliberadamente independiente de FastAPI y
de cualquier framework web: el enunciado pide que el motor funcione de manera
autónoma respecto de los sistemas que lo consumen, así que acá no hay nada de HTTP.

Responsabilidades:

- elegir la estrategia adecuada para cada usuario (personalizada o popularidad);
- mantener el historial de interacciones en memoria;
- recalcular el vector latente de un usuario cuando llegan interacciones nuevas;
- serializar y recuperar todo el estado desde disco.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .als import ModeloALS
from .config import CONFIG_ALS, ConfiguracionALS
from .datos import Catalogo, DatosInteraccion, calcular_confianza
from .popularidad import RecomendadorPopularidad

ESTRATEGIA_PERSONALIZADA = "factorizacion"
ESTRATEGIA_POPULARIDAD = "popularidad"

# Cantidad mínima de usuarios que debe tener un juego para que su vector latente se
# considere confiable al buscar ítems similares.
SOPORTE_MINIMO_SIMILARES = 5


@dataclass(frozen=True)
class Recomendacion:
    """Una entrada de la lista de recomendaciones.

    Atributos:
        titulo: nombre del juego.
        puntaje: preferencia estimada. Solo es comparable dentro de una misma
            respuesta: no es una probabilidad ni una nota sobre un máximo fijo.
        estrategia: qué método produjo la recomendación.
    """

    titulo: str
    puntaje: float
    estrategia: str


@dataclass(frozen=True)
class Interaccion:
    """Interacción de un usuario con un juego, tal como la reporta el cliente.

    Atributos:
        titulo: nombre del juego, debe existir en el catálogo.
        compro: si el usuario adquirió el juego.
        horas: horas jugadas acumuladas. Cero si solo lo compró.
    """

    titulo: str
    compro: bool = True
    horas: float = 0.0


class MotorRecomendacion:
    """Fachada del sistema de recomendación.

    Atributos:
        modelo: modelo ALS entrenado.
        catalogo: traductor entre ids/títulos e índices.
        popularidad: recomendador de respaldo para usuarios sin historial.
        config: hiperparámetros con los que se calculan las confianzas.
    """

    def __init__(
        self,
        modelo: ModeloALS,
        catalogo: Catalogo,
        popularidad: RecomendadorPopularidad,
        historial: dict[int, dict[int, tuple[bool, float]]],
        factores_usuario: dict[int, np.ndarray],
        config: ConfiguracionALS = CONFIG_ALS,
    ) -> None:
        self.modelo = modelo
        self.catalogo = catalogo
        self.popularidad = popularidad
        self.config = config
        self._historial = historial
        self._factores_usuario = factores_usuario
        # Las actualizaciones por nuevas interacciones mutan el estado compartido;
        # el servidor web atiende pedidos concurrentes, así que se serializan.
        self._candado = threading.RLock()

    # ------------------------------------------------------------------ #
    # Construcción
    # ------------------------------------------------------------------ #

    @classmethod
    def desde_entrenamiento(
        cls,
        datos: DatosInteraccion,
        modelo: ModeloALS,
        popularidad: RecomendadorPopularidad,
    ) -> "MotorRecomendacion":
        """Arma el motor a partir de un entrenamiento recién terminado.

        Args:
            datos: salida del preprocesamiento, con matriz y catálogo.
            modelo: modelo ALS ya ajustado sobre ``datos.matriz``.
            popularidad: ranking global ya calculado.

        Returns:
            Motor listo para responder consultas.
        """
        historial: dict[int, dict[int, tuple[bool, float]]] = {}
        for id_usuario, titulo, compro, horas in datos.interacciones.itertuples(
            index=False, name=None
        ):
            indice_juego = datos.catalogo.indice_de_juego(titulo)
            if indice_juego is None:
                continue
            historial.setdefault(int(id_usuario), {})[indice_juego] = (
                bool(compro),
                float(horas),
            )

        factores_usuario = {
            datos.catalogo.usuario_de_indice(i): modelo.factores_usuario[i].copy()
            for i in range(datos.catalogo.cantidad_usuarios)
        }

        return cls(
            modelo=modelo,
            catalogo=datos.catalogo,
            popularidad=popularidad,
            historial=historial,
            factores_usuario=factores_usuario,
            config=modelo.config,
        )

    # ------------------------------------------------------------------ #
    # Consultas
    # ------------------------------------------------------------------ #

    def existe_usuario(self, id_usuario: int) -> bool:
        """Indica si el usuario tiene historial conocido."""
        return id_usuario in self._historial

    def existe_juego(self, titulo: str) -> bool:
        """Indica si el título está en el catálogo."""
        return self.catalogo.indice_de_juego(titulo) is not None

    def recomendar(self, id_usuario: int, k: int = 10) -> list[Recomendacion]:
        """Devuelve las k mejores recomendaciones para un usuario.

        Si el usuario tiene historial se usa la factorización; si no, se cae al
        ranking por popularidad. En ambos casos se excluyen los juegos que el
        usuario ya consumió, por el criterio de novedad.

        Args:
            id_usuario: identificador del usuario.
            k: cantidad de recomendaciones pedidas.

        Returns:
            Lista de :class:`Recomendacion` ordenada de mayor a menor puntaje. Puede
            tener menos de k elementos si el catálogo se agota.

        Raises:
            ValueError: si k no es positivo.
        """
        if k <= 0:
            raise ValueError("k debe ser un entero positivo")

        with self._candado:
            factor = self._factores_usuario.get(id_usuario)
            consumidos = set(self._historial.get(id_usuario, {}).keys())

            # Un usuario sin historial (o cuyo vector quedó en cero porque nunca
            # tuvo interacciones válidas) no admite personalización.
            if factor is None or not np.any(factor):
                return [
                    Recomendacion(
                        titulo=self.catalogo.titulo_de_indice(indice),
                        puntaje=puntaje,
                        estrategia=ESTRATEGIA_POPULARIDAD,
                    )
                    for indice, puntaje in self.popularidad.recomendar(k, consumidos)
                ]

            puntajes = self.modelo.puntuar(factor)

        if consumidos:
            puntajes = puntajes.copy()
            puntajes[list(consumidos)] = -np.inf

        cantidad = min(k, int(np.isfinite(puntajes).sum()))
        if cantidad <= 0:
            return []

        mejores = np.argpartition(-puntajes, cantidad - 1)[:cantidad]
        mejores = mejores[np.argsort(-puntajes[mejores])]

        return [
            Recomendacion(
                titulo=self.catalogo.titulo_de_indice(int(indice)),
                puntaje=float(puntajes[indice]),
                estrategia=ESTRATEGIA_PERSONALIZADA,
            )
            for indice in mejores
        ]

    def historial_usuario(self, id_usuario: int) -> list[Interaccion]:
        """Devuelve las interacciones registradas para un usuario.

        Args:
            id_usuario: identificador del usuario.

        Returns:
            Lista de interacciones ordenada por horas jugadas descendente. Vacía si
            el usuario no tiene historial.
        """
        with self._candado:
            registros = self._historial.get(id_usuario, {})
            interacciones = [
                Interaccion(
                    titulo=self.catalogo.titulo_de_indice(indice),
                    compro=compro,
                    horas=horas,
                )
                for indice, (compro, horas) in registros.items()
            ]
        return sorted(interacciones, key=lambda i: i.horas, reverse=True)

    def buscar_juegos(self, texto: str = "", limite: int = 20) -> list[str]:
        """Busca títulos del catálogo que contengan un texto.

        Existe para que quien consume la API pueda descubrir los nombres exactos
        antes de registrar interacciones, ya que los títulos son la clave de los
        ítems y deben coincidir literalmente.

        Args:
            texto: subcadena a buscar, sin distinguir mayúsculas.
            limite: cantidad máxima de resultados.

        Returns:
            Lista de títulos coincidentes en orden alfabético.
        """
        texto = texto.strip().lower()
        if not texto:
            return [str(t) for t in self.catalogo.juegos[:limite]]
        coincidencias = [
            str(titulo)
            for titulo in self.catalogo.juegos
            if texto in str(titulo).lower()
        ]
        return coincidencias[:limite]

    def juegos_similares(
        self, titulo: str, k: int = 10, soporte_minimo: int = SOPORTE_MINIMO_SIMILARES
    ) -> list[Recomendacion]:
        """Devuelve los juegos más parecidos a uno dado según el espacio latente.

        Se descartan los títulos con muy pocos usuarios. Un tercio del catálogo
        aparece en dos usuarios o menos, y el vector latente de esos ítems queda
        determinado por un puñado de celdas: su dirección es esencialmente ruido y,
        como la similitud coseno ignora la magnitud, cualquiera de ellos puede
        alinearse con la referencia por azar y colarse entre los primeros puestos.
        El filtro exige un mínimo de evidencia antes de considerar a un ítem vecino.

        Args:
            titulo: juego de referencia.
            k: cantidad de vecinos.
            soporte_minimo: cantidad mínima de usuarios que debe tener un juego para
                poder aparecer como vecino.

        Returns:
            Lista de recomendaciones ordenada por similitud coseno descendente.

        Raises:
            KeyError: si el título no está en el catálogo.
        """
        indice = self.catalogo.indice_de_juego(titulo)
        if indice is None:
            raise KeyError(f"El juego '{titulo}' no está en el catálogo")

        candidatos = self.popularidad.soporte >= soporte_minimo
        # El juego de referencia siempre es candidato válido: el propio método lo
        # excluye después, y sin esto un título de nicho no tendría vecinos.
        candidatos[indice] = True

        return [
            Recomendacion(
                titulo=self.catalogo.titulo_de_indice(vecino),
                puntaje=similitud,
                estrategia=ESTRATEGIA_PERSONALIZADA,
            )
            for vecino, similitud in self.modelo.juegos_similares(indice, k, candidatos)
        ]

    # ------------------------------------------------------------------ #
    # Actualización en caliente
    # ------------------------------------------------------------------ #

    def registrar_interacciones(
        self, id_usuario: int, interacciones: list[Interaccion]
    ) -> int:
        """Incorpora interacciones nuevas y recalcula el perfil del usuario.

        Cubre los puntos 2 y 3 del enunciado: las recomendaciones se modifican a
        partir de nuevas interacciones, y un usuario que no existía queda dado de
        alta con su primer registro. No se reentrena el modelo completo, solo se
        resuelve la ecuación de ese usuario contra los factores de ítem, que
        permanecen fijos.

        Si el par usuario-juego ya existía, las horas se reemplazan por el máximo
        entre lo registrado y lo informado, en coherencia con el criterio del
        preprocesamiento: el campo de horas es acumulado, no incremental.

        Args:
            id_usuario: identificador del usuario.
            interacciones: interacciones a registrar.

        Returns:
            Cantidad de interacciones aplicadas.

        Raises:
            KeyError: si algún título no pertenece al catálogo. La validación es
                previa a cualquier modificación, para no dejar el estado a medio
                actualizar.
        """
        if not interacciones:
            return 0

        desconocidos = [
            i.titulo for i in interacciones if self.catalogo.indice_de_juego(i.titulo) is None
        ]
        if desconocidos:
            raise KeyError(
                "Títulos fuera del catálogo: " + ", ".join(sorted(set(desconocidos)))
            )

        with self._candado:
            registros = self._historial.setdefault(id_usuario, {})
            for interaccion in interacciones:
                indice = self.catalogo.indice_de_juego(interaccion.titulo)
                compro_previo, horas_previas = registros.get(indice, (False, 0.0))
                registros[indice] = (
                    compro_previo or interaccion.compro,
                    max(horas_previas, float(interaccion.horas)),
                )

            self._recalcular_factor(id_usuario)

        return len(interacciones)

    def _recalcular_factor(self, id_usuario: int) -> None:
        """Resuelve por fold-in el vector latente de un usuario. Requiere el candado."""
        registros = self._historial.get(id_usuario, {})
        if not registros:
            self._factores_usuario.pop(id_usuario, None)
            return

        indices = np.fromiter(registros.keys(), dtype=np.int64, count=len(registros))
        compro = np.fromiter(
            (valor[0] for valor in registros.values()), dtype=bool, count=len(registros)
        )
        horas = np.fromiter(
            (valor[1] for valor in registros.values()),
            dtype=np.float64,
            count=len(registros),
        )
        confianzas = calcular_confianza(compro, horas, self.config)
        self._factores_usuario[id_usuario] = self.modelo.calcular_factor_usuario(
            indices, confianzas
        )

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #

    def guardar(self, ruta: Path) -> None:
        """Serializa el modelo y el estado completo en un archivo .npz.

        Se guarda el historial además de los factores porque el motor necesita
        poder excluir lo ya consumido y recalcular perfiles después de un reinicio.

        Args:
            ruta: archivo destino. Se crea el directorio si no existe.
        """
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)

        with self._candado:
            usuarios_estado = sorted(self._historial.keys())
            filas, columnas, compras, horas = [], [], [], []
            for id_usuario in usuarios_estado:
                for indice_juego, (compro, hs) in self._historial[id_usuario].items():
                    filas.append(id_usuario)
                    columnas.append(indice_juego)
                    compras.append(compro)
                    horas.append(hs)

            ids_con_factor = sorted(self._factores_usuario.keys())
            matriz_factores = (
                np.vstack([self._factores_usuario[u] for u in ids_con_factor])
                if ids_con_factor
                else np.zeros((0, self.modelo.config.factores))
            )

            np.savez_compressed(
                ruta,
                factores_juego=self.modelo.factores_juego,
                factores_usuario=matriz_factores,
                ids_con_factor=np.asarray(ids_con_factor, dtype=np.int64),
                catalogo_usuarios=self.catalogo.usuarios,
                catalogo_juegos=self.catalogo.juegos,
                popularidad_puntajes=self.popularidad.puntajes,
                popularidad_soporte=self.popularidad.soporte,
                historial_usuario=np.asarray(filas, dtype=np.int64),
                historial_juego=np.asarray(columnas, dtype=np.int64),
                historial_compro=np.asarray(compras, dtype=bool),
                historial_horas=np.asarray(horas, dtype=np.float64),
                configuracion=json.dumps(asdict(self.config)),
            )

    @classmethod
    def cargar(cls, ruta: Path) -> "MotorRecomendacion":
        """Reconstruye un motor desde un archivo generado por :meth:`guardar`.

        Args:
            ruta: archivo .npz con el modelo persistido.

        Returns:
            Motor listo para responder consultas.

        Raises:
            FileNotFoundError: si el archivo no existe.
        """
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No hay un modelo entrenado en {ruta}. "
                "Ejecutá scripts/entrenar.py antes de levantar la API."
            )

        with np.load(ruta, allow_pickle=True) as archivo:
            # Cada acceso por clave a un .npz comprimido descomprime el array
            # completo, así que se extraen todos una única vez. Hacerlo dentro de
            # un bucle convierte una carga de milisegundos en una de varios minutos.
            factores_juego = archivo["factores_juego"]
            matriz_factores = archivo["factores_usuario"]
            ids_con_factor = archivo["ids_con_factor"]
            catalogo_usuarios = archivo["catalogo_usuarios"]
            catalogo_juegos = archivo["catalogo_juegos"]
            popularidad_puntajes = archivo["popularidad_puntajes"]
            popularidad_soporte = archivo["popularidad_soporte"]
            historial_usuario = archivo["historial_usuario"]
            historial_juego = archivo["historial_juego"]
            historial_compro = archivo["historial_compro"]
            historial_horas = archivo["historial_horas"]
            configuracion = str(archivo["configuracion"])

        config = ConfiguracionALS(**json.loads(configuracion))

        modelo = ModeloALS(config)
        modelo.factores_juego = factores_juego
        modelo.factores_usuario = matriz_factores

        catalogo = Catalogo(usuarios=catalogo_usuarios, juegos=catalogo_juegos)
        popularidad = RecomendadorPopularidad(
            puntajes=popularidad_puntajes,
            orden=np.argsort(-popularidad_puntajes),
            soporte=popularidad_soporte,
        )

        historial: dict[int, dict[int, tuple[bool, float]]] = {}
        for id_usuario, indice_juego, compro, horas in zip(
            historial_usuario.tolist(),
            historial_juego.tolist(),
            historial_compro.tolist(),
            historial_horas.tolist(),
        ):
            historial.setdefault(id_usuario, {})[indice_juego] = (compro, horas)

        factores_usuario = {
            int(id_usuario): matriz_factores[posicion]
            for posicion, id_usuario in enumerate(ids_con_factor.tolist())
        }

        return cls(
            modelo=modelo,
            catalogo=catalogo,
            popularidad=popularidad,
            historial=historial,
            factores_usuario=factores_usuario,
            config=config,
        )

    # ------------------------------------------------------------------ #
    # Diagnóstico
    # ------------------------------------------------------------------ #

    def estadisticas(self) -> dict[str, int | float]:
        """Devuelve métricas de estado, para el endpoint de salud y la operación."""
        with self._candado:
            return {
                "usuarios": len(self._historial),
                "juegos": self.catalogo.cantidad_juegos,
                "interacciones": sum(len(v) for v in self._historial.values()),
                "factores": int(self.modelo.config.factores),
            }
