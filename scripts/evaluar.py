"""Valida el motor de recomendación con métricas de ranking.

Uso:
    python scripts/evaluar.py
    python scripts/evaluar.py --k 20 --factores 96

Entrena sobre la partición de entrenamiento (no reutiliza el modelo persistido, que
vio el dataset completo) y compara la factorización contra el ranking por
popularidad, que es la referencia mínima que hay que superar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recomendador.config import (  # noqa: E402
    CONFIG_ALS,
    CONFIG_EVALUACION,
    RUTA_DATASET,
    ConfiguracionALS,
    ConfiguracionEvaluacion,
)
from recomendador.datos import (  # noqa: E402
    cargar_dataset,
    construir_matriz,
    consolidar_interacciones,
)
from recomendador.evaluacion import evaluar  # noqa: E402


def parsear_argumentos() -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Evalúa el motor de recomendación.")
    parser.add_argument("--dataset", type=Path, default=RUTA_DATASET)
    parser.add_argument("--k", type=int, default=CONFIG_EVALUACION.k)
    parser.add_argument("--factores", type=int, default=CONFIG_ALS.factores)
    parser.add_argument("--iteraciones", type=int, default=CONFIG_ALS.iteraciones)
    parser.add_argument("--alfa", type=float, default=CONFIG_ALS.alfa)
    parser.add_argument(
        "--regularizacion", type=float, default=CONFIG_ALS.regularizacion
    )
    parser.add_argument(
        "--proporcion-prueba", type=float, default=CONFIG_EVALUACION.proporcion_prueba
    )
    parser.add_argument("--silencioso", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Ejecuta el protocolo de validación e imprime la tabla comparativa."""
    argumentos = parsear_argumentos()

    config_als = ConfiguracionALS(
        factores=argumentos.factores,
        regularizacion=argumentos.regularizacion,
        iteraciones=argumentos.iteraciones,
        alfa=argumentos.alfa,
    )
    config_evaluacion = ConfiguracionEvaluacion(
        k=argumentos.k, proporcion_prueba=argumentos.proporcion_prueba
    )

    crudos = cargar_dataset(argumentos.dataset)
    interacciones = consolidar_interacciones(crudos)
    catalogo = construir_matriz(interacciones, config_als).catalogo

    print(
        f"Evaluación top-{config_evaluacion.k} · f={config_als.factores} · "
        f"lambda={config_als.regularizacion} · alfa={config_als.alfa}"
    )
    resultados = evaluar(
        interacciones,
        catalogo,
        config_als,
        config_evaluacion,
        verboso=not argumentos.silencioso,
    )

    metricas = list(resultados["factorizacion"].keys())
    ancho = max(len(m) for m in metricas) + 2

    print("\n" + "-" * (ancho + 32))
    print(f"{'métrica':<{ancho}}{'factorización':>16}{'popularidad':>16}")
    print("-" * (ancho + 32))
    for metrica in metricas:
        valor_modelo = resultados["factorizacion"][metrica]
        valor_base = resultados["popularidad"][metrica]
        if metrica == "usuarios_evaluados":
            print(f"{metrica:<{ancho}}{valor_modelo:>16,.0f}{valor_base:>16,.0f}")
        else:
            print(f"{metrica:<{ancho}}{valor_modelo:>16.4f}{valor_base:>16.4f}")
    print("-" * (ancho + 32))

    mejora = (
        resultados["factorizacion"][f"map@{config_evaluacion.k}"]
        / resultados["popularidad"][f"map@{config_evaluacion.k}"]
        if resultados["popularidad"][f"map@{config_evaluacion.k}"] > 0
        else float("inf")
    )
    print(f"\nMejora en MAP@{config_evaluacion.k} sobre popularidad: {mejora:,.2f}x")


if __name__ == "__main__":
    main()
