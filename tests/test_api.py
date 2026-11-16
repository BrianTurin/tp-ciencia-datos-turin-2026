"""Pruebas de la API REST.

Se levanta la aplicación real contra un modelo entrenado sobre el dataset sintético,
persistido en un directorio temporal. No se usa el modelo de producción: las pruebas
registran interacciones y no deben ensuciar el artefacto real.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main as modulo_api


@pytest.fixture
def cliente(motor, tmp_path, monkeypatch):
    """Cliente HTTP apuntando a la aplicación con el modelo sintético cargado."""
    ruta = tmp_path / "modelo_prueba.npz"
    motor.guardar(ruta)
    monkeypatch.setattr(modulo_api, "RUTA_MODELO", ruta)

    with TestClient(modulo_api.app) as cliente:
        yield cliente


# ---------------------------------------------------------------------- #
# Operación
# ---------------------------------------------------------------------- #


def test_salud_reporta_servicio_operativo(cliente):
    respuesta = cliente.get("/salud")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "operativo"
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["juegos"] > 0


# ---------------------------------------------------------------------- #
# Recomendaciones
# ---------------------------------------------------------------------- #


def test_recomendaciones_de_usuario_conocido(cliente):
    """Punto 2 del enunciado: entra un id de usuario y un k, salen k recomendaciones."""
    respuesta = cliente.get("/recomendaciones/1", params={"k": 3})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["usuario_conocido"] is True
    assert cuerpo["estrategia"] == "factorizacion"
    assert len(cuerpo["recomendaciones"]) == 3


def test_recomendaciones_respetan_el_orden_por_puntaje(cliente):
    cuerpo = cliente.get("/recomendaciones/1", params={"k": 5}).json()
    puntajes = [r["puntaje"] for r in cuerpo["recomendaciones"]]

    assert puntajes == sorted(puntajes, reverse=True)


def test_usuario_desconocido_recibe_popularidad(cliente):
    """Punto 3: un usuario sin interacciones previas igual obtiene recomendaciones."""
    cuerpo = cliente.get("/recomendaciones/888888", params={"k": 4}).json()

    assert cuerpo["usuario_conocido"] is False
    assert cuerpo["estrategia"] == "popularidad"
    assert len(cuerpo["recomendaciones"]) == 4


@pytest.mark.parametrize("k", [0, -1, 101])
def test_k_fuera_de_rango_es_rechazado(cliente, k):
    """La validación del parámetro corre antes de tocar el motor."""
    assert cliente.get("/recomendaciones/1", params={"k": k}).status_code == 422


# ---------------------------------------------------------------------- #
# Registro de interacciones
# ---------------------------------------------------------------------- #


def test_registrar_interacciones_modifica_las_recomendaciones(cliente):
    """Punto 2: el sistema modifica las recomendaciones ante nuevas interacciones."""
    antes = cliente.get("/recomendaciones/777777", params={"k": 3}).json()

    respuesta = cliente.post(
        "/interacciones",
        json={
            "id_usuario": 777777,
            "interacciones": [
                {"titulo": "Shooter A", "comportamiento": "play", "horas": 30.0},
                {"titulo": "Shooter B", "comportamiento": "play", "horas": 12.0},
            ],
        },
    )

    assert respuesta.status_code == 201
    cuerpo = respuesta.json()
    assert cuerpo["usuario_nuevo"] is True
    assert cuerpo["interacciones_registradas"] == 2
    assert cuerpo["total_historial"] == 2

    despues = cliente.get("/recomendaciones/777777", params={"k": 3}).json()
    assert despues["estrategia"] == "factorizacion"
    assert despues["recomendaciones"] != antes["recomendaciones"]


def test_no_recomienda_lo_recien_registrado(cliente):
    """Lo que el usuario acaba de declarar como consumido no puede volver en la lista."""
    cliente.post(
        "/interacciones",
        json={
            "id_usuario": 777778,
            "interacciones": [
                {"titulo": "Estrategia A", "comportamiento": "play", "horas": 20.0}
            ],
        },
    )

    cuerpo = cliente.get("/recomendaciones/777778", params={"k": 5}).json()
    titulos = {r["titulo"] for r in cuerpo["recomendaciones"]}

    assert "Estrategia A" not in titulos


def test_titulo_fuera_del_catalogo_devuelve_422(cliente):
    respuesta = cliente.post(
        "/interacciones",
        json={
            "id_usuario": 777779,
            "interacciones": [{"titulo": "Juego Que No Existe", "comportamiento": "purchase"}],
        },
    )

    assert respuesta.status_code == 422
    assert "Juego Que No Existe" in respuesta.json()["detalle"]


def test_lista_de_interacciones_vacia_es_rechazada(cliente):
    respuesta = cliente.post(
        "/interacciones", json={"id_usuario": 777780, "interacciones": []}
    )
    assert respuesta.status_code == 422


def test_una_compra_sin_horas_igual_registra_la_interaccion(cliente):
    """Comprar sin jugar es una señal válida y debe quedar en el historial."""
    cliente.post(
        "/interacciones",
        json={
            "id_usuario": 777781,
            "interacciones": [{"titulo": "Shooter C", "comportamiento": "purchase"}],
        },
    )

    historial = cliente.get("/usuarios/777781/historial").json()

    assert historial["cantidad"] == 1
    assert historial["interacciones"][0]["titulo"] == "Shooter C"
    assert historial["interacciones"][0]["horas"] == 0.0


# ---------------------------------------------------------------------- #
# Consultas
# ---------------------------------------------------------------------- #


def test_historial_de_usuario_inexistente_devuelve_404(cliente):
    respuesta = cliente.get("/usuarios/888889/historial")

    assert respuesta.status_code == 404
    assert "detalle" in respuesta.json()


def test_busqueda_de_juegos(cliente):
    resultados = cliente.get("/juegos", params={"busqueda": "shooter"}).json()

    assert resultados
    assert all("Shooter" in titulo for titulo in resultados)


def test_juegos_similares(cliente):
    respuesta = cliente.get("/juegos/Shooter A/similares", params={"k": 3})

    assert respuesta.status_code == 200
    titulos = [r["titulo"] for r in respuesta.json()]
    assert "Shooter A" not in titulos
    assert len(titulos) == 3


def test_similares_de_juego_inexistente_devuelve_404(cliente):
    respuesta = cliente.get("/juegos/Juego Que No Existe/similares")
    assert respuesta.status_code == 404


# ---------------------------------------------------------------------- #
# Persistencia entre reinicios
# ---------------------------------------------------------------------- #


def test_las_interacciones_sobreviven_al_reinicio(motor, tmp_path, monkeypatch):
    """Al apagarse, el servicio persiste el estado; al volver, lo recupera."""
    ruta = tmp_path / "modelo_persistencia.npz"
    motor.guardar(ruta)
    monkeypatch.setattr(modulo_api, "RUTA_MODELO", ruta)

    with TestClient(modulo_api.app) as cliente:
        cliente.post(
            "/interacciones",
            json={
                "id_usuario": 654321,
                "interacciones": [
                    {"titulo": "Estrategia B", "comportamiento": "play", "horas": 44.0}
                ],
            },
        )
        esperado = cliente.get("/recomendaciones/654321", params={"k": 3}).json()

    with TestClient(modulo_api.app) as cliente:
        historial = cliente.get("/usuarios/654321/historial").json()
        obtenido = cliente.get("/recomendaciones/654321", params={"k": 3}).json()

    assert historial["cantidad"] == 1
    assert obtenido["recomendaciones"] == esperado["recomendaciones"]
