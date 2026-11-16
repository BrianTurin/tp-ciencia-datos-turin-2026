"""API REST del sistema de recomendación de videojuegos.

Esta capa es una cáscara delgada sobre :class:`recomendador.motor.MotorRecomendacion`:
traduce HTTP a llamadas del motor y de vuelta. Toda la lógica de recomendación vive
en el paquete ``recomendador``, que no depende de FastAPI y puede usarse desde un
script, un notebook o cualquier otro proceso.

Cómo levantarla:

    python scripts/entrenar.py          # genera modelo/modelo_als.npz
    uvicorn api.main:app --app-dir src --reload

Documentación interactiva en http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Path as ParametroRuta, Query
from fastapi.responses import JSONResponse, RedirectResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.esquemas import (  # noqa: E402
    InteraccionesEntrada,
    InteraccionSalida,
    RecomendacionSalida,
    RespuestaHistorial,
    RespuestaInteracciones,
    RespuestaRecomendaciones,
    RespuestaSalud,
)
from recomendador.config import RUTA_MODELO  # noqa: E402
from recomendador.motor import (  # noqa: E402
    ESTRATEGIA_POPULARIDAD,
    SOPORTE_MINIMO_SIMILARES,
    Interaccion,
    MotorRecomendacion,
)

# Estado del proceso. El motor se carga una sola vez al arrancar: deserializar el
# modelo en cada pedido costaría segundos y no aportaría nada.
_motor: MotorRecomendacion | None = None


def _cargar_motor() -> None:
    """Carga el modelo si todavía no está en memoria.

    En un entorno serverless (Vercel) el evento de arranque puede no ejecutarse,
    así que la carga también se intenta de forma perezosa en el primer pedido.
    """
    global _motor
    if _motor is not None:
        return
    try:
        _motor = MotorRecomendacion.cargar(RUTA_MODELO)
        print(f"Modelo cargado desde {RUTA_MODELO}: {_motor.estadisticas()}")
    except FileNotFoundError as error:
        # No se aborta: /salud tiene que poder responder para que el problema sea
        # diagnosticable desde afuera.
        print(f"ADVERTENCIA: {error}")


def obtener_motor() -> MotorRecomendacion:
    """Devuelve el motor cargado.

    Returns:
        Instancia única de :class:`MotorRecomendacion`.

    Raises:
        HTTPException: 503 si el modelo no llegó a cargarse.
    """
    _cargar_motor()
    if _motor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está cargado. Ejecutá 'python scripts/entrenar.py' y "
                "reiniciá el servicio."
            ),
        )
    return _motor


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carga el modelo al arrancar y persiste el estado al cerrar.

    Guardar en el apagado es lo que hace que las interacciones registradas en
    caliente sobrevivan a un reinicio del servicio.
    """
    _cargar_motor()

    yield

    if _motor is not None:
        try:
            _motor.guardar(RUTA_MODELO)
            print(f"Estado persistido en {RUTA_MODELO}")
        except OSError as error:
            # En Vercel el sistema de archivos es de solo lectura: el estado nuevo
            # vive en memoria mientras la instancia siga activa.
            print(f"No se pudo persistir el estado: {error}")


app = FastAPI(
    title="Sistema recomendador de videojuegos",
    description=(
        "Motor de recomendación basado en factorización de matrices sobre feedback "
        "implícito (compras y horas jugadas)."
    ),
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


@app.get("/", include_in_schema=False)
def inicio() -> RedirectResponse:
    """Lleva a la documentación interactiva, desde donde se prueba la API."""
    return RedirectResponse(url="/docs")


@app.get("/salud", response_model=RespuestaSalud, tags=["operación"])
def salud() -> RespuestaSalud:
    """Informa si el servicio está operativo y con qué volumen de datos trabaja."""
    _cargar_motor()
    if _motor is None:
        return RespuestaSalud(
            estado="degradado",
            modelo_cargado=False,
            usuarios=0,
            juegos=0,
            interacciones=0,
            factores=0,
        )

    estadisticas = _motor.estadisticas()
    return RespuestaSalud(
        estado="operativo",
        modelo_cargado=True,
        usuarios=int(estadisticas["usuarios"]),
        juegos=int(estadisticas["juegos"]),
        interacciones=int(estadisticas["interacciones"]),
        factores=int(estadisticas["factores"]),
    )


@app.get(
    "/recomendaciones/{id_usuario}",
    response_model=RespuestaRecomendaciones,
    tags=["recomendaciones"],
)
def recomendaciones(
    id_usuario: int = ParametroRuta(description="Identificador del usuario."),
    k: int = Query(
        default=10, ge=1, le=100, description="Cantidad de recomendaciones a devolver."
    ),
) -> RespuestaRecomendaciones:
    """Devuelve las k mejores recomendaciones para un usuario.

    Para usuarios con historial usa el modelo de factorización. Para usuarios sin
    interacciones previas responde igual, con el ranking por popularidad: el
    servicio nunca devuelve una lista vacía por desconocer al usuario.

    Los juegos que el usuario ya consumió quedan excluidos de la lista.
    """
    motor = obtener_motor()
    conocido = motor.existe_usuario(id_usuario)
    resultado = motor.recomendar(id_usuario, k)

    estrategia = resultado[0].estrategia if resultado else ESTRATEGIA_POPULARIDAD

    return RespuestaRecomendaciones(
        id_usuario=id_usuario,
        k=k,
        usuario_conocido=conocido,
        estrategia=estrategia,
        recomendaciones=[
            RecomendacionSalida(
                titulo=r.titulo, puntaje=round(r.puntaje, 6), estrategia=r.estrategia
            )
            for r in resultado
        ],
    )


@app.post(
    "/interacciones",
    response_model=RespuestaInteracciones,
    status_code=201,
    tags=["recomendaciones"],
)
def registrar_interacciones(cuerpo: InteraccionesEntrada) -> RespuestaInteracciones:
    """Registra interacciones nuevas y actualiza el perfil del usuario.

    A partir de este pedido, las recomendaciones de ese usuario reflejan las
    interacciones informadas. Si el usuario no existía, queda dado de alta y pasa
    de recibir recomendaciones por popularidad a recibirlas personalizadas.

    No se reentrena el modelo completo: se recalcula únicamente el vector latente
    del usuario contra los factores de ítem, que permanecen fijos.
    """
    motor = obtener_motor()
    era_nuevo = not motor.existe_usuario(cuerpo.id_usuario)

    interacciones = [
        Interaccion(
            titulo=entrada.titulo,
            # Un evento de 'play' implica que el juego está en la biblioteca del
            # usuario, así que también cuenta como compra.
            compro=True,
            horas=entrada.horas if entrada.comportamiento.lower() == "play" else 0.0,
        )
        for entrada in cuerpo.interacciones
    ]

    try:
        registradas = motor.registrar_interacciones(cuerpo.id_usuario, interacciones)
    except KeyError as error:
        raise HTTPException(status_code=422, detail=str(error.args[0])) from error

    total = len(motor.historial_usuario(cuerpo.id_usuario))
    return RespuestaInteracciones(
        id_usuario=cuerpo.id_usuario,
        interacciones_registradas=registradas,
        total_historial=total,
        usuario_nuevo=era_nuevo,
        mensaje=(
            "Usuario dado de alta; las recomendaciones ya son personalizadas."
            if era_nuevo
            else "Perfil actualizado; las recomendaciones reflejan las nuevas interacciones."
        ),
    )


@app.get(
    "/usuarios/{id_usuario}/historial",
    response_model=RespuestaHistorial,
    tags=["consultas"],
)
def historial(
    id_usuario: int = ParametroRuta(description="Identificador del usuario."),
) -> RespuestaHistorial:
    """Devuelve las interacciones registradas de un usuario.

    Raises:
        HTTPException: 404 si el usuario no tiene historial.
    """
    motor = obtener_motor()
    if not motor.existe_usuario(id_usuario):
        raise HTTPException(
            status_code=404, detail=f"El usuario {id_usuario} no tiene historial."
        )

    interacciones = motor.historial_usuario(id_usuario)
    return RespuestaHistorial(
        id_usuario=id_usuario,
        cantidad=len(interacciones),
        interacciones=[
            InteraccionSalida(titulo=i.titulo, compro=i.compro, horas=i.horas)
            for i in interacciones
        ],
    )


@app.get("/juegos", response_model=list[str], tags=["consultas"])
def juegos(
    busqueda: str = Query(default="", description="Filtra por subcadena del título."),
    limite: int = Query(default=20, ge=1, le=200),
) -> list[str]:
    """Busca títulos del catálogo.

    Los títulos son la clave de los ítems y deben coincidir exactamente al
    registrar interacciones, así que este endpoint existe para resolverlos antes.
    """
    return obtener_motor().buscar_juegos(busqueda, limite)


@app.get(
    "/juegos/{titulo}/similares",
    response_model=list[RecomendacionSalida],
    tags=["consultas"],
)
def similares(
    titulo: str = ParametroRuta(description="Título exacto del juego de referencia."),
    k: int = Query(default=10, ge=1, le=50),
    soporte_minimo: int = Query(
        default=SOPORTE_MINIMO_SIMILARES,
        ge=1,
        description=(
            "Cantidad mínima de usuarios que debe tener un juego para poder aparecer "
            "como vecino. Filtra títulos cuyo vector latente es puro ruido por falta "
            "de evidencia."
        ),
    ),
) -> list[RecomendacionSalida]:
    """Devuelve los juegos más parecidos a uno dado según el espacio latente.

    Sirve para recomendar desde la ficha de un producto y para inspeccionar
    cualitativamente qué aprendió el modelo.

    Raises:
        HTTPException: 404 si el título no está en el catálogo.
    """
    motor = obtener_motor()
    try:
        vecinos = motor.juegos_similares(titulo, k, soporte_minimo)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error.args[0])) from error

    return [
        RecomendacionSalida(
            titulo=v.titulo, puntaje=round(v.puntaje, 6), estrategia=v.estrategia
        )
        for v in vecinos
    ]


@app.exception_handler(HTTPException)
async def manejador_http(request, exc: HTTPException) -> JSONResponse:
    """Uniforma el cuerpo de los errores bajo la clave ``detalle``."""
    return JSONResponse(status_code=exc.status_code, content={"detalle": exc.detail})
