"""Genera las figuras del informe a partir del dataset y del motor.

Uso:
    python scripts/graficos.py                    # figuras rápidas (solo datos)
    python scripts/graficos.py --todas            # incluye las que entrenan modelos
    python scripts/graficos.py --solo horas_jugadas popularidad_juegos
    python scripts/graficos.py --lista            # muestra qué figuras existen
    python scripts/graficos.py --formato pdf --salida otra/carpeta
    python scripts/graficos.py --con-titulos      # título dentro de la imagen

Salida (``outputs/`` por defecto):

- una imagen por figura (300 dpi, escala de grises, números con coma decimal);
- ``pies_de_figura.md`` con el número, el título y la nota de cada figura en el
  formato de APA 7, listos para pegar debajo de la imagen en el informe.

Por defecto las imágenes no llevan título: en APA el título va en el pie de figura,
no dentro de la imagen.

Cómo agregar una figura nueva
-----------------------------
Escribir una función que reciba el contexto y devuelva una ``Figure`` de matplotlib,
y decorarla con ``@figura``. No hay que tocar nada más: aparece sola en ``--lista``,
se puede invocar con ``--solo`` y su pie queda en ``pies_de_figura.md``.

    @figura(
        "mi_grafico",
        justifica="Qué decisión respalda.",
        titulo="Título que va en el pie de figura",
        nota=lambda ctx: "Texto de la nota, con datos si hace falta.",
    )
    def mi_grafico(ctx: Contexto) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.hist(ctx.interacciones["horas"])
        return fig

``ctx`` ofrece ``ctx.crudos`` (dataset original), ``ctx.interacciones`` (un registro
por usuario-juego) y ``ctx.config_als``. Marcar ``costosa=True`` si la figura entrena
modelos, para que no se regenere en cada corrida. Para mostrar números con el
formato del informe usar :func:`numero`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: solo escribe archivos
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
except ImportError:  # pragma: no cover
    sys.exit(
        "Falta matplotlib. Instalarlo con:  pip install matplotlib  "
        "(o  pip install -e .[graficos])"
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recomendador.config import (  # noqa: E402
    CONFIG_ALS,
    CONFIG_EVALUACION,
    RAIZ_PROYECTO,
    RUTA_DATASET,
    ConfiguracionALS,
)
from recomendador.datos import (  # noqa: E402
    cargar_dataset,
    consolidar_interacciones,
    construir_matriz,
)
from recomendador.evaluacion import evaluar  # noqa: E402

DIRECTORIO_SALIDA = RAIZ_PROYECTO / "outputs"

# Estilo sobrio y apto para imprimir en blanco y negro (el informe va todo en negro).
NEGRO = "#000000"
GRIS_OSCURO = "#404040"
GRIS = "#8c8c8c"
GRIS_CLARO = "#cfcfcf"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRIS_CLARO,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# --------------------------------------------------------------------------- #
# Formato de números (coma decimal, punto de miles, como en el informe)
# --------------------------------------------------------------------------- #
def numero(valor: float, decimales: int = 0) -> str:
    """Formatea un número con coma decimal y punto como separador de miles."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "\0").replace(".", ",").replace("\0", ".")


def _formatear(ax, eje: str, decimales: int = 0) -> None:
    """Aplica el formato de :func:`numero` a los rótulos de un eje."""
    formato = FuncFormatter(lambda valor, _: numero(valor, decimales))
    gestor = ax.yaxis if eje == "y" else ax.xaxis
    gestor.set_major_formatter(formato)
    if decimales == 0:
        # Sin decimales, una marca en 2,5 se rotularía "2" (o "3"): solo se
        # permiten marcas en valores enteros para que el rótulo nunca mienta.
        gestor.set_major_locator(MaxNLocator(integer=True))


def _eje_logaritmico(ax, eje: str, marcas: list[float]) -> None:
    """Eje logarítmico con rótulos legibles (1; 10; 100...) en lugar de 10^n."""
    if eje == "x":
        ax.set_xscale("log")
        ax.set_xticks(marcas)
        gestor = ax.xaxis
    else:
        ax.set_yscale("log")
        ax.set_yticks(marcas)
        gestor = ax.yaxis
    gestor.set_major_formatter(
        FuncFormatter(lambda v, _: numero(v, 1 if v < 1 else 0))
    )
    gestor.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------- #
# Infraestructura
# --------------------------------------------------------------------------- #
@dataclass
class Contexto:
    """Datos compartidos por todas las figuras (se cargan una sola vez)."""

    crudos: pd.DataFrame
    interacciones: pd.DataFrame
    config_als: ConfiguracionALS
    con_titulos: bool = False


@dataclass(frozen=True)
class Figura:
    """Entrada del registro de figuras."""

    nombre: str
    justifica: str
    titulo: str
    nota: Callable[[Contexto], str]
    costosa: bool
    funcion: Callable[[Contexto], plt.Figure]


REGISTRO: dict[str, Figura] = {}


def figura(
    nombre: str,
    justifica: str,
    titulo: str,
    nota: Callable[[Contexto], str],
    costosa: bool = False,
):
    """Registra una función generadora de figuras bajo ``nombre``."""

    def decorador(funcion: Callable[[Contexto], plt.Figure]):
        if nombre in REGISTRO:
            raise ValueError(f"Ya existe una figura llamada {nombre!r}")
        REGISTRO[nombre] = Figura(nombre, justifica, titulo, nota, costosa, funcion)
        return funcion

    return decorador


def _titulo(ctx: Contexto, ax, texto: str) -> None:
    """Pone título dentro de la imagen solo si se pidió con --con-titulos."""
    if ctx.con_titulos:
        ax.set_title(texto, fontsize=12, color=NEGRO)


def _etiquetas(ax, x: str, y: str) -> None:
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def _pares_jugados(ctx: Contexto) -> pd.Series:
    inter = ctx.interacciones
    return inter.loc[inter["horas"] > 0, "horas"]


FUENTE = "Elaboración propia a partir del dataset del trabajo práctico."


# --------------------------------------------------------------------------- #
# Figuras de entendimiento de los datos
# --------------------------------------------------------------------------- #
@figura(
    "horas_jugadas",
    justifica="Cola larga de horas: justifica la transformación logarítmica de la confianza.",
    titulo="Distribución acumulada de las horas jugadas",
    nota=lambda ctx: (
        f"Considera los {numero(len(_pares_jugados(ctx)))} pares usuario-juego con "
        "horas mayores a 0 (se excluyen los comprados sin jugar). El eje horizontal "
        f"está en escala logarítmica. {FUENTE}"
    ),
)
def horas_jugadas(ctx: Contexto) -> plt.Figure:
    # Se usa la distribución acumulada y no un histograma: las horas vienen con una
    # sola decimal, y los bins logarítmicos caen de forma despareja sobre esos
    # valores y generan picos y huecos que no existen en los datos.
    horas = np.sort(_pares_jugados(ctx).to_numpy())
    acumulado = np.arange(1, len(horas) + 1) / len(horas) * 100
    mediana, p90, maximo = np.median(horas), np.quantile(horas, 0.9), horas[-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(horas, acumulado, where="post", color=NEGRO, linewidth=1.5)
    _eje_logaritmico(ax, "x", [0.1, 1, 10, 100, 1000, 10000])
    ax.set_xlim(0.08, maximo * 2)
    ax.set_ylim(0, 102)

    for valor, nivel, texto in [
        (mediana, 50, f"mediana: {numero(mediana, 1)} h"),
        (p90, 90, f"percentil 90: {numero(p90)} h"),
    ]:
        ax.vlines(valor, 0, nivel, color=GRIS_OSCURO, linestyle="--", linewidth=1)
        ax.hlines(nivel, 0.08, valor, color=GRIS_OSCURO, linestyle="--", linewidth=1)
        ax.annotate(
            texto,
            (valor, nivel),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=9,
        )
    ax.annotate(
        f"máximo: {numero(maximo)} h",
        (maximo, 100),
        xytext=(-6, -18),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    _formatear(ax, "y")
    _etiquetas(
        ax,
        "Horas jugadas (escala logarítmica)",
        "% de pares usuario-juego con hasta x horas",
    )
    _titulo(ctx, ax, "Distribución acumulada de las horas jugadas")
    return fig


@figura(
    "juegos_por_usuario",
    justifica="Pocos juegos por usuario: explica la dispersión y por qué se evalúa solo a una parte.",
    titulo="Cantidad de juegos por usuario",
    nota=lambda ctx: (
        f"Total: {numero(ctx.interacciones['id_usuario'].nunique())} usuarios. Las "
        "barras claras son los usuarios con menos de 5 juegos, que no entran en la "
        f"validación del sistema. {FUENTE}"
    ),
)
def juegos_por_usuario(ctx: Contexto) -> plt.Figure:
    por_usuario = ctx.interacciones.groupby("id_usuario").size()
    total = len(por_usuario)
    tope = 11  # el último grupo reúne a quienes tienen 11 juegos o más
    cuentas = por_usuario.clip(upper=tope).value_counts()
    cantidades = list(range(1, tope + 1))
    valores = [int(cuentas.get(n, 0)) for n in cantidades]
    etiquetas = [str(n) for n in cantidades[:-1]] + [f"{tope} o más"]
    pocos = por_usuario.lt(5).mean() * 100

    fig, ax = plt.subplots(figsize=(7.5, 4))
    barras = ax.bar(
        range(len(valores)),
        valores,
        color=[GRIS_CLARO if n < 5 else GRIS for n in cantidades],
        edgecolor=NEGRO,
        linewidth=0.6,
    )
    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor,
            f"{numero(valor / total * 100)} %",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(range(len(valores)), etiquetas)
    ax.set_ylim(0, max(valores) * 1.15)
    ax.grid(axis="x", visible=False)
    _formatear(ax, "y")
    ax.legend(
        handles=[
            Patch(
                facecolor=GRIS_CLARO,
                edgecolor=NEGRO,
                label=f"Menos de 5 juegos ({numero(pocos)} % de los usuarios)",
            ),
            Patch(facecolor=GRIS, edgecolor=NEGRO, label="5 juegos o más"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper right",
    )
    _etiquetas(ax, "Juegos por usuario", "Cantidad de usuarios")
    _titulo(ctx, ax, "Cantidad de juegos por usuario")
    return fig


@figura(
    "popularidad_juegos",
    justifica="Concentración en pocos títulos: justifica la popularidad para el arranque en frío.",
    titulo="Concentración de las interacciones en el catálogo",
    nota=lambda ctx: (
        "Cada interacción es un par usuario-juego. En el panel a el eje horizontal "
        "ordena los juegos del más al menos jugado. En el panel b se acumula el "
        f"porcentaje de interacciones. {FUENTE}"
    ),
)
def popularidad_juegos(ctx: Contexto) -> plt.Figure:
    jugadores = (
        ctx.interacciones.groupby("titulo").size().sort_values(ascending=False).to_numpy()
    )
    acumulado = np.cumsum(jugadores) / jugadores.sum() * 100
    porcentaje_catalogo = np.arange(1, len(jugadores) + 1) / len(jugadores) * 100
    corte = int(np.searchsorted(porcentaje_catalogo, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(np.arange(1, len(jugadores) + 1), jugadores, color=NEGRO, linewidth=1.3)
    _eje_logaritmico(ax1, "x", [1, 10, 100, 1000])
    _eje_logaritmico(ax1, "y", [1, 10, 100, 1000])
    ax1.set_title("a) Usuarios por juego", fontsize=11)
    _etiquetas(ax1, "Posición del juego (escala logarítmica)", "Usuarios que lo tienen (escala logarítmica)")

    ax2.plot(porcentaje_catalogo, acumulado, color=NEGRO, linewidth=1.3)
    ax2.axvline(10, color=GRIS, linestyle="--", linewidth=1)
    ax2.annotate(
        f"El 10 % del catálogo concentra\nel {numero(acumulado[corte])} % de las interacciones",
        xy=(10, acumulado[corte]),
        xytext=(25, acumulado[corte] - 25),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": GRIS_OSCURO},
    )
    ax2.set_title("b) Concentración acumulada", fontsize=11)
    _etiquetas(
        ax2,
        "% del catálogo (ordenado por popularidad)",
        "% acumulado de las interacciones",
    )
    ax2.set_ylim(0, 100)
    _formatear(ax2, "x")
    _formatear(ax2, "y")
    fig.tight_layout()
    return fig


@figura(
    "compra_vs_juego",
    justifica="Compras sin jugar: justifica darle peso propio a la compra en la confianza.",
    titulo="Composición de los pares usuario-juego",
    nota=lambda ctx: (
        f"El dataset original tiene {numero(len(ctx.crudos))} filas: un juego comprado "
        "y jugado aparece en dos (una de compra y otra de juego). Al consolidarlas "
        f"quedan {numero(len(ctx.interacciones))} pares usuario-juego. {FUENTE}"
    ),
)
def compra_vs_juego(ctx: Contexto) -> plt.Figure:
    inter = ctx.interacciones
    jugado = inter["horas"] > 0
    categorias = {
        "Comprado y jugado": int((inter["compro"] & jugado).sum()),
        "Comprado, nunca jugado": int((inter["compro"] & ~jugado).sum()),
        "Jugado, sin compra registrada": int((~inter["compro"] & jugado).sum()),
    }
    categorias = {nombre: valor for nombre, valor in categorias.items() if valor > 0}
    total = sum(categorias.values())

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    posiciones = np.arange(len(categorias))
    barras = ax.barh(
        posiciones, list(categorias.values()), color=GRIS, edgecolor=NEGRO, linewidth=0.6
    )
    ax.set_yticks(posiciones, list(categorias.keys()))
    ax.invert_yaxis()
    ax.set_xlim(0, max(categorias.values()) * 1.3)
    for barra, valor in zip(barras, categorias.values()):
        ax.text(
            valor,
            barra.get_y() + barra.get_height() / 2,
            f"  {numero(valor)} ({numero(valor / total * 100, 1)} %)",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="y", visible=False)
    _formatear(ax, "x")
    ax.set_xlabel(
        f"Pares usuario-juego ({numero(len(ctx.crudos))} filas consolidadas en {numero(total)} pares)"
    )
    _titulo(ctx, ax, "Composición de los pares usuario-juego")
    return fig


# --------------------------------------------------------------------------- #
# Figuras de validación (entrenan modelos: se marcan como costosas)
# --------------------------------------------------------------------------- #
NOTA_VALIDACION = (
    "Validación con el 20 % de los juegos de cada usuario ocultos, sobre los "
    "usuarios con al menos 5 juegos, y una sola semilla (42). " + FUENTE
)


def _evaluar(ctx: Contexto, config: ConfiguracionALS) -> dict[str, dict[str, float]]:
    catalogo = construir_matriz(ctx.interacciones, config).catalogo
    return evaluar(ctx.interacciones, catalogo, config, CONFIG_EVALUACION, verboso=False)


def _barrido(
    ctx: Contexto, campo: str, valores: list[float]
) -> tuple[list[float], list[float]]:
    """Entrena y evalúa una vez por valor de un hiperparámetro."""
    k = CONFIG_EVALUACION.k
    mapas, coberturas = [], []
    for valor in valores:
        print(f"    entrenando con {campo} = {valor}...")
        resultados = _evaluar(ctx, replace(ctx.config_als, **{campo: valor}))
        mapas.append(resultados["factorizacion"][f"map@{k}"])
        coberturas.append(resultados["factorizacion"]["cobertura_catalogo"] * 100)
    return mapas, coberturas


def _grafico_barrido(
    ctx: Contexto,
    valores: list[float],
    mapas: list[float],
    coberturas: list[float],
    elegido: float,
    etiqueta_x: str,
    decimales_x: int = 0,
) -> plt.Figure:
    """MAP y cobertura frente a un hiperparámetro, en dos paneles apilados.

    Se evita el doble eje: los marcadores de una serie terminaban tapando las
    etiquetas de la otra. Dos paneles con el mismo eje horizontal dicen lo mismo
    sin ambigüedad.
    """
    k = CONFIG_EVALUACION.k
    posiciones = list(range(len(valores)))

    fig, (ax_map, ax_cob) = plt.subplots(
        2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw={"hspace": 0.12}
    )

    ax_map.plot(posiciones, mapas, color=NEGRO, marker="o", linewidth=1.6)
    ax_cob.plot(
        posiciones, coberturas, color=GRIS_OSCURO, marker="s", linestyle="--", linewidth=1.4
    )

    if elegido in valores:
        indice = valores.index(elegido)
        for ax, serie in ((ax_map, mapas), (ax_cob, coberturas)):
            ax.plot(
                indice, serie[indice], marker="o", markersize=13,
                markerfacecolor="none", color=NEGRO,
            )

    for x, y in zip(posiciones, mapas):
        ax_map.annotate(
            numero(y, 3), (x, y), textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9,
        )
    for x, y in zip(posiciones, coberturas):
        ax_cob.annotate(
            numero(y, 1), (x, y), textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=9,
        )

    ax_map.set_ylim(0, max(mapas) * 1.2)
    ax_cob.set_ylim(0, max(coberturas) * 1.25)
    _formatear(ax_map, "y", 2)
    _formatear(ax_cob, "y")
    ax_map.set_ylabel(f"MAP@{k}")
    ax_cob.set_ylabel("Cobertura de catálogo (%)")
    ax_map.set_title(f"a) MAP@{k}", fontsize=11)
    ax_cob.set_title("b) Cobertura de catálogo", fontsize=11)
    ax_cob.set_xticks(posiciones, [numero(v, decimales_x) for v in valores])
    ax_cob.set_xlabel(etiqueta_x)
    return fig


@figura(
    "sensibilidad_factores",
    justifica="MAP según cantidad de factores: respalda elegir 128 frente a 64 y 192.",
    titulo="Sensibilidad a la cantidad de factores latentes",
    nota=lambda ctx: (
        f"El círculo marca el valor elegido ({ctx.config_als.factores}). "
        f"{NOTA_VALIDACION}"
    ),
    costosa=True,
)
def sensibilidad_factores(ctx: Contexto) -> plt.Figure:
    valores = [32, 64, 128, 192]
    mapas, coberturas = _barrido(ctx, "factores", valores)
    fig = _grafico_barrido(
        ctx, valores, mapas, coberturas, ctx.config_als.factores, "Factores latentes"
    )
    _titulo(ctx, fig.axes[0], "Sensibilidad a la cantidad de factores")
    return fig


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
def escribir_pies(ctx: Contexto, destino: Path) -> None:
    """Escribe los pies de figura de todas las figuras en formato APA 7."""
    lineas = [
        "# Pies de figura",
        "",
        "Pegar debajo de cada imagen. Ajustar el número de figura al orden en que",
        "aparecen en el informe, y citarlas en el texto (por ejemplo, \"Figura 1\").",
        "",
    ]
    for numero_figura, f in enumerate(REGISTRO.values(), start=1):
        lineas += [
            f"**Figura {numero_figura}**  (`{f.nombre}`)",
            "",
            f"*{f.titulo}*",
            "",
            f"*Nota.* {f.nota(ctx)}",
            "",
        ]
    destino.write_text("\n".join(lineas), encoding="utf-8")


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera las figuras del informe.")
    parser.add_argument("--dataset", type=Path, default=RUTA_DATASET)
    parser.add_argument("--salida", type=Path, default=DIRECTORIO_SALIDA)
    parser.add_argument("--formato", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--solo", nargs="+", metavar="NOMBRE", help="generar solo estas figuras")
    parser.add_argument("--todas", action="store_true", help="incluir las figuras costosas")
    parser.add_argument("--lista", action="store_true", help="listar figuras y salir")
    parser.add_argument(
        "--con-titulos", action="store_true", help="poner título dentro de la imagen"
    )
    return parser.parse_args()


def main() -> None:
    argumentos = parsear_argumentos()

    if argumentos.lista:
        for f in REGISTRO.values():
            marca = " [costosa]" if f.costosa else ""
            print(f"{f.nombre}{marca}\n    {f.justifica}")
        return

    if argumentos.solo:
        desconocidas = [n for n in argumentos.solo if n not in REGISTRO]
        if desconocidas:
            sys.exit(f"Figuras inexistentes: {', '.join(desconocidas)}. Ver --lista.")
        elegidas = [REGISTRO[n] for n in argumentos.solo]
    else:
        elegidas = [f for f in REGISTRO.values() if argumentos.todas or not f.costosa]

    crudos = cargar_dataset(argumentos.dataset)
    ctx = Contexto(
        crudos,
        consolidar_interacciones(crudos),
        CONFIG_ALS,
        con_titulos=argumentos.con_titulos,
    )

    argumentos.salida.mkdir(parents=True, exist_ok=True)
    for f in elegidas:
        print(f"- {f.nombre}")
        fig = f.funcion(ctx)
        destino = argumentos.salida / f"{f.nombre}.{argumentos.formato}"
        fig.savefig(destino)
        plt.close(fig)
        print(f"    guardada en {destino}")

    pies = argumentos.salida / "pies_de_figura.md"
    escribir_pies(ctx, pies)
    print(f"- pies de figura en {pies}")


if __name__ == "__main__":
    main()
