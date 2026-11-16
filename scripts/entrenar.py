"""Entrena el motor de recomendación y lo deja persistido en disco.

Uso:
    python scripts/entrenar.py
    python scripts/entrenar.py --factores 96 --iteraciones 20 --alfa 30

El artefacto resultante (``modelo/modelo_als.npz``) es lo único que necesita la API
para levantar: contiene los factores latentes, el catálogo, el ranking de
popularidad y el historial de interacciones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recomendador.als import ModeloALS  # noqa: E402
from recomendador.config import (  # noqa: E402
    CONFIG_ALS,
    RUTA_DATASET,
    RUTA_MODELO,
    ConfiguracionALS,
)
from recomendador.datos import (  # noqa: E402
    cargar_dataset,
    construir_matriz,
    consolidar_interacciones,
    resumen_dataset,
)
from recomendador.motor import MotorRecomendacion  # noqa: E402
from recomendador.popularidad import RecomendadorPopularidad  # noqa: E402


def parsear_argumentos() -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena el motor de recomendación.")
    parser.add_argument("--dataset", type=Path, default=RUTA_DATASET)
    parser.add_argument("--salida", type=Path, default=RUTA_MODELO)
    parser.add_argument("--factores", type=int, default=CONFIG_ALS.factores)
    parser.add_argument("--iteraciones", type=int, default=CONFIG_ALS.iteraciones)
    parser.add_argument("--alfa", type=float, default=CONFIG_ALS.alfa)
    parser.add_argument(
        "--regularizacion", type=float, default=CONFIG_ALS.regularizacion
    )
    parser.add_argument("--silencioso", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Ejecuta el pipeline completo de entrenamiento."""
    argumentos = parsear_argumentos()
    verboso = not argumentos.silencioso

    config = ConfiguracionALS(
        factores=argumentos.factores,
        regularizacion=argumentos.regularizacion,
        iteraciones=argumentos.iteraciones,
        alfa=argumentos.alfa,
    )

    print(f"[1/4] Leyendo {argumentos.dataset}")
    crudos = cargar_dataset(argumentos.dataset)
    for clave, valor in resumen_dataset(crudos).items():
        formato = f"{valor:,.4f}" if isinstance(valor, float) else f"{valor:,}"
        print(f"      {clave:.<38} {formato}")

    print("[2/4] Consolidando interacciones y armando la matriz de confianza")
    interacciones = consolidar_interacciones(crudos)
    datos = construir_matriz(interacciones, config)
    print(
        f"      matriz {datos.matriz.shape[0]:,} x {datos.matriz.shape[1]:,} "
        f"con {datos.matriz.nnz:,} celdas observadas"
    )

    print(f"[3/4] Entrenando ALS implícito (f={config.factores}, "
          f"lambda={config.regularizacion}, alfa={config.alfa})")
    inicio = time.perf_counter()
    modelo = ModeloALS(config).entrenar(datos.matriz, verboso=verboso)
    print(f"      entrenamiento completado en {time.perf_counter() - inicio:,.1f} s")

    popularidad = RecomendadorPopularidad.entrenar(datos.interacciones, datos.catalogo)
    motor = MotorRecomendacion.desde_entrenamiento(datos, modelo, popularidad)

    print(f"[4/4] Guardando el modelo en {argumentos.salida}")
    motor.guardar(argumentos.salida)
    tamanio = argumentos.salida.stat().st_size / 1024**2
    print(f"      artefacto de {tamanio:,.1f} MB")

    _mostrar_ejemplos(motor, datos)


def _mostrar_ejemplos(motor: MotorRecomendacion, datos) -> None:
    """Imprime un par de salidas de ejemplo como verificación rápida."""
    print("\nVerificación rápida")
    print("-" * 60)

    id_usuario = datos.catalogo.usuario_de_indice(0)
    historial = motor.historial_usuario(id_usuario)[:5]
    print(f"Usuario {id_usuario} · juegos con más horas:")
    for interaccion in historial:
        print(f"  - {interaccion.titulo} ({interaccion.horas:,.1f} h)")

    print(f"\nRecomendaciones para el usuario {id_usuario}:")
    for recomendacion in motor.recomendar(id_usuario, k=5):
        print(f"  - {recomendacion.titulo} ({recomendacion.puntaje:.4f})")

    print("\nRecomendaciones para un usuario nuevo (arranque en frío):")
    for recomendacion in motor.recomendar(-1, k=5):
        print(f"  - {recomendacion.titulo} ({recomendacion.puntaje:.4f})")


if __name__ == "__main__":
    main()
