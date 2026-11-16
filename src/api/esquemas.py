"""Esquemas de entrada y salida de la API.

Separar los modelos Pydantic del código de rutas mantiene el contrato de la API en
un solo lugar y permite que FastAPI genere la documentación OpenAPI con ejemplos.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InteraccionEntrada(BaseModel):
    """Interacción de un usuario con un juego, tal como la envía el cliente."""

    titulo: str = Field(
        ...,
        description="Título exacto del juego, tal como figura en el catálogo.",
        examples=["Counter-Strike Global Offensive"],
    )
    comportamiento: str = Field(
        default="purchase",
        description="Tipo de interacción: 'purchase' (compra) o 'play' (juego).",
        examples=["play"],
    )
    horas: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Horas jugadas acumuladas. Se ignora si el comportamiento es 'purchase'. "
            "Es un acumulado, no un incremento: si se envía un valor menor al ya "
            "registrado, se conserva el mayor."
        ),
        examples=[27.5],
    )


class InteraccionesEntrada(BaseModel):
    """Cuerpo del pedido para registrar interacciones de un usuario."""

    id_usuario: int = Field(
        ...,
        description="Identificador del usuario. Si no existe, se da de alta.",
        examples=[151603712],
    )
    interacciones: list[InteraccionEntrada] = Field(
        ...,
        min_length=1,
        description="Interacciones a registrar. Deben referirse a juegos del catálogo.",
    )


class RecomendacionSalida(BaseModel):
    """Una entrada de la lista de recomendaciones."""

    titulo: str = Field(description="Título del juego recomendado.")
    puntaje: float = Field(
        description=(
            "Preferencia estimada. Solo sirve para ordenar dentro de esta misma "
            "respuesta; no es una probabilidad ni una escala fija."
        )
    )
    estrategia: str = Field(
        description="Método que generó la recomendación: 'factorizacion' o 'popularidad'."
    )


class RespuestaRecomendaciones(BaseModel):
    """Respuesta del endpoint de recomendaciones."""

    id_usuario: int
    k: int
    usuario_conocido: bool = Field(
        description="False si el usuario no tenía historial al momento de la consulta."
    )
    estrategia: str = Field(
        description=(
            "Estrategia aplicada. 'popularidad' indica arranque en frío: el usuario "
            "no tiene interacciones suficientes para personalizar."
        )
    )
    recomendaciones: list[RecomendacionSalida]


class RespuestaInteracciones(BaseModel):
    """Respuesta al registrar interacciones nuevas."""

    id_usuario: int
    interacciones_registradas: int
    total_historial: int = Field(
        description="Cantidad de juegos distintos que el usuario tiene registrados."
    )
    usuario_nuevo: bool = Field(
        description="True si el usuario no existía antes de este pedido."
    )
    mensaje: str


class InteraccionSalida(BaseModel):
    """Una entrada del historial de un usuario."""

    titulo: str
    compro: bool
    horas: float


class RespuestaHistorial(BaseModel):
    """Respuesta del endpoint de historial."""

    id_usuario: int
    cantidad: int
    interacciones: list[InteraccionSalida]


class RespuestaSalud(BaseModel):
    """Respuesta del endpoint de salud."""

    estado: str
    modelo_cargado: bool
    usuarios: int
    juegos: int
    interacciones: int
    factores: int


class RespuestaError(BaseModel):
    """Cuerpo estándar de los errores."""

    detalle: str
