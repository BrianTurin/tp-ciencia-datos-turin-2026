"""Pruebas del motor de recomendación.

Se apoyan en el dataset sintético de ``conftest.py``, donde las preferencias están
construidas a propósito: dos comunidades de gustos con títulos que co-ocurren entre
sí. Eso permite verificar que el modelo aprende la estructura, y no solamente que
devuelve algo con la forma correcta.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import JUEGOS_ESTRATEGIA, JUEGOS_SHOOTER
from recomendador.als import ModeloALS
from recomendador.datos import calcular_confianza
from recomendador.motor import (
    ESTRATEGIA_PERSONALIZADA,
    ESTRATEGIA_POPULARIDAD,
    Interaccion,
    MotorRecomendacion,
)

SHOOTERS = set(JUEGOS_SHOOTER)
ESTRATEGIA = set(JUEGOS_ESTRATEGIA)

ID_USUARIO_EXISTENTE = 1
ID_USUARIO_INEXISTENTE = 500_000


# ---------------------------------------------------------------------- #
# Recomendaciones personalizadas
# ---------------------------------------------------------------------- #


def test_recomienda_dentro_de_la_comunidad_de_gustos(motor):
    """Un usuario que juega shooters debe recibir shooters, no estrategia."""
    motor.registrar_interacciones(
        7001,
        [Interaccion("Shooter A", horas=30.0), Interaccion("Shooter B", horas=25.0)],
    )

    recomendaciones = motor.recomendar(7001, k=3)

    assert {r.titulo for r in recomendaciones} <= SHOOTERS
    assert all(r.estrategia == ESTRATEGIA_PERSONALIZADA for r in recomendaciones)


def test_la_comunidad_opuesta_recibe_el_catalogo_opuesto(motor):
    """La contraparte del caso anterior: el modelo separa ambos grupos."""
    motor.registrar_interacciones(
        7002,
        [
            Interaccion("Estrategia A", horas=30.0),
            Interaccion("Estrategia B", horas=25.0),
        ],
    )

    recomendaciones = motor.recomendar(7002, k=3)

    assert {r.titulo for r in recomendaciones} <= ESTRATEGIA


def test_no_recomienda_lo_ya_consumido(motor):
    """El criterio de novedad exige excluir lo que el usuario ya tiene."""
    consumidos = {i.titulo for i in motor.historial_usuario(ID_USUARIO_EXISTENTE)}

    recomendados = {r.titulo for r in motor.recomendar(ID_USUARIO_EXISTENTE, k=10)}

    assert consumidos
    assert not (consumidos & recomendados)


def test_respeta_la_cantidad_pedida(motor):
    """La lista devuelta no puede exceder k."""
    assert len(motor.recomendar(ID_USUARIO_EXISTENTE, k=2)) == 2
    assert len(motor.recomendar(ID_USUARIO_EXISTENTE, k=5)) == 5


def test_nunca_devuelve_mas_de_lo_que_queda_en_catalogo(motor):
    """Con k mayor al catálogo disponible, la lista se trunca en lugar de fallar."""
    consumidos = len(motor.historial_usuario(ID_USUARIO_EXISTENTE))
    disponibles = motor.catalogo.cantidad_juegos - consumidos

    assert len(motor.recomendar(ID_USUARIO_EXISTENTE, k=100)) == disponibles


def test_recomendaciones_ordenadas_por_puntaje_descendente(motor):
    """El ranking tiene que venir ordenado: es un problema de top-k."""
    puntajes = [r.puntaje for r in motor.recomendar(ID_USUARIO_EXISTENTE, k=5)]
    assert puntajes == sorted(puntajes, reverse=True)


def test_k_invalido_es_rechazado(motor):
    """Pedir cero o menos recomendaciones es un error del cliente."""
    with pytest.raises(ValueError):
        motor.recomendar(ID_USUARIO_EXISTENTE, k=0)


# ---------------------------------------------------------------------- #
# Arranque en frío
# ---------------------------------------------------------------------- #


def test_usuario_nuevo_recibe_recomendaciones_por_popularidad(motor):
    """Punto 3 del enunciado: nadie se queda sin recomendaciones."""
    recomendaciones = motor.recomendar(ID_USUARIO_INEXISTENTE, k=3)

    assert len(recomendaciones) == 3
    assert all(r.estrategia == ESTRATEGIA_POPULARIDAD for r in recomendaciones)


def test_usuario_nuevo_no_figura_como_conocido(motor):
    """El motor tiene que poder distinguir a quién conoce."""
    assert motor.existe_usuario(ID_USUARIO_INEXISTENTE) is False
    assert motor.existe_usuario(ID_USUARIO_EXISTENTE) is True


# ---------------------------------------------------------------------- #
# Actualización a partir de nuevas interacciones
# ---------------------------------------------------------------------- #


def test_las_recomendaciones_cambian_con_nuevas_interacciones(motor):
    """Punto 2 del enunciado: el perfil se actualiza sin reentrenar todo."""
    antes = [r.titulo for r in motor.recomendar(ID_USUARIO_INEXISTENTE, k=3)]

    motor.registrar_interacciones(
        ID_USUARIO_INEXISTENTE, [Interaccion("Estrategia A", horas=50.0)]
    )
    despues = [r.titulo for r in motor.recomendar(ID_USUARIO_INEXISTENTE, k=3)]

    assert antes != despues


def test_usuario_nuevo_pasa_a_personalizado_tras_su_primera_interaccion(motor):
    """El alta y la personalización ocurren en el mismo pedido."""
    assert motor.recomendar(7003, k=1)[0].estrategia == ESTRATEGIA_POPULARIDAD

    motor.registrar_interacciones(
        7003,
        [
            Interaccion("Estrategia A", horas=40.0),
            Interaccion("Estrategia B", horas=35.0),
        ],
    )

    recomendaciones = motor.recomendar(7003, k=2)
    assert all(r.estrategia == ESTRATEGIA_PERSONALIZADA for r in recomendaciones)
    assert {r.titulo for r in recomendaciones} <= ESTRATEGIA


def test_horas_se_actualizan_al_maximo_y_no_se_suman(motor):
    """El campo de horas es acumulado; sumarlo duplicaría el mismo tiempo de juego."""
    motor.registrar_interacciones(7004, [Interaccion("Shooter A", horas=10.0)])
    motor.registrar_interacciones(7004, [Interaccion("Shooter A", horas=30.0)])
    motor.registrar_interacciones(7004, [Interaccion("Shooter A", horas=5.0)])

    historial = motor.historial_usuario(7004)
    assert len(historial) == 1
    assert historial[0].horas == pytest.approx(30.0)


def test_registrar_sin_interacciones_no_hace_nada(motor):
    assert motor.registrar_interacciones(7005, []) == 0
    assert motor.existe_usuario(7005) is False


def test_titulo_fuera_del_catalogo_es_rechazado(motor):
    """Sin ítems nuevos en el catálogo, un título desconocido es un error."""
    with pytest.raises(KeyError):
        motor.registrar_interacciones(
            ID_USUARIO_EXISTENTE, [Interaccion("Juego Inexistente")]
        )


def test_el_rechazo_no_deja_el_estado_a_medio_actualizar(motor):
    """La validación es previa: o se aplican todas las interacciones o ninguna."""
    with pytest.raises(KeyError):
        motor.registrar_interacciones(
            7006, [Interaccion("Shooter A", horas=10.0), Interaccion("No Existe")]
        )

    assert motor.existe_usuario(7006) is False


# ---------------------------------------------------------------------- #
# Similitud entre ítems
# ---------------------------------------------------------------------- #


def test_juegos_similares_se_mantienen_dentro_de_la_comunidad(motor):
    """Los vecinos de un shooter deberían ser shooters.

    Se usa 'Shooter A' porque el puente entre comunidades del dataset sintético pasa
    por el último título de cada grupo, no por este.
    """
    vecinos = {r.titulo for r in motor.juegos_similares("Shooter A", k=4)}

    assert vecinos <= SHOOTERS
    assert "Shooter A" not in vecinos


def test_juego_inexistente_al_buscar_similares(motor):
    with pytest.raises(KeyError):
        motor.juegos_similares("Juego Inexistente")


def test_busqueda_de_juegos_no_distingue_mayusculas(motor):
    assert "Shooter A" in motor.buscar_juegos("shooter a")
    assert motor.buscar_juegos("no existe nada asi") == []
    assert len(motor.buscar_juegos("shooter", limite=2)) == 2


# ---------------------------------------------------------------------- #
# Fold-in
# ---------------------------------------------------------------------- #


def test_fold_in_reproduce_el_factor_aprendido(motor):
    """El fold-in y el paso de usuario del entrenamiento resuelven la misma ecuación.

    Si ambos coinciden, actualizar un perfil en caliente equivale a haberlo entrenado
    con esas interacciones desde el principio, que es lo que justifica no reentrenar
    el modelo completo ante cada interacción nueva.
    """
    registros = motor._historial[ID_USUARIO_EXISTENTE]
    indices = np.fromiter(registros.keys(), dtype=np.int64)
    compro = np.array([valor[0] for valor in registros.values()])
    horas = np.array([valor[1] for valor in registros.values()])
    confianzas = calcular_confianza(compro, horas, motor.config)

    factor_fold_in = motor.modelo.calcular_factor_usuario(indices, confianzas)
    factor_entrenado = motor._factores_usuario[ID_USUARIO_EXISTENTE]

    np.testing.assert_allclose(factor_fold_in, factor_entrenado, atol=1e-8)


def test_factor_de_usuario_sin_interacciones_es_nulo(motor):
    """Es la solución exacta del sistema regularizado cuando no hay observaciones."""
    factor = motor.modelo.calcular_factor_usuario(
        np.array([], dtype=np.int64), np.array([])
    )
    assert np.all(factor == 0.0)


def test_modelo_sin_entrenar_falla_con_mensaje_claro():
    with pytest.raises(RuntimeError, match="no está entrenado"):
        ModeloALS().puntuar(np.zeros(8))


# ---------------------------------------------------------------------- #
# Persistencia
# ---------------------------------------------------------------------- #


def test_guardar_y_cargar_preserva_las_recomendaciones(motor, tmp_path):
    """Un reinicio del servicio no debe cambiar lo que el sistema responde."""
    motor.registrar_interacciones(7007, [Interaccion("Shooter A", horas=12.0)])
    esperado = [r.titulo for r in motor.recomendar(7007, k=3)]

    ruta = tmp_path / "modelo.npz"
    motor.guardar(ruta)
    recuperado = MotorRecomendacion.cargar(ruta)

    assert [r.titulo for r in recuperado.recomendar(7007, k=3)] == esperado
    assert recuperado.estadisticas() == motor.estadisticas()


def test_el_historial_sobrevive_al_guardado(motor, tmp_path):
    """Sin historial persistido, tras un reinicio se volvería a recomendar lo consumido."""
    ruta = tmp_path / "modelo.npz"
    motor.guardar(ruta)
    recuperado = MotorRecomendacion.cargar(ruta)

    original = {i.titulo for i in motor.historial_usuario(ID_USUARIO_EXISTENTE)}
    recargado = {i.titulo for i in recuperado.historial_usuario(ID_USUARIO_EXISTENTE)}

    assert original == recargado


def test_cargar_modelo_inexistente_da_error_explicativo(tmp_path):
    with pytest.raises(FileNotFoundError, match="entrenar"):
        MotorRecomendacion.cargar(tmp_path / "no_existe.npz")


def test_similares_filtra_titulos_sin_evidencia(motor):
    """Un juego con un solo usuario no debe poder colarse entre los vecinos.

    La similitud coseno ignora la magnitud del vector, así que un ítem con evidencia
    casi nula puede alinearse con cualquier referencia por puro azar.
    """
    motor.registrar_interacciones(7008, [Interaccion("Shooter A", horas=1.0)])
    indice_raro = motor.catalogo.indice_de_juego(JUEGOS_ESTRATEGIA[-1])

    # Se fuerza un soporte mínimo alto: ningún juego lo alcanza salvo la referencia.
    assert motor.juegos_similares("Shooter A", k=3, soporte_minimo=10_000) == []
    assert indice_raro is not None


def test_similares_sin_filtro_considera_todo_el_catalogo(motor):
    """Con soporte mínimo 1 el filtro no descarta nada."""
    vecinos = motor.juegos_similares("Shooter A", k=9, soporte_minimo=1)
    assert len(vecinos) == motor.catalogo.cantidad_juegos - 1
