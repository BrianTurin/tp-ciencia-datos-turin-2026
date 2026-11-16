# Sistema recomendador de videojuegos

Motor de recomendación sobre feedback implícito (compras y horas jugadas) con API REST.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python scripts/entrenar.py                       # genera modelo/modelo_als.npz (~40 s)
python scripts/evaluar.py                        # métricas de ranking vs. popularidad
python -m pytest tests/ -q                       # suite de pruebas
uvicorn api.main:app --app-dir src --reload      # levanta la API en :8000
```

Documentación interactiva de la API en `http://127.0.0.1:8000/docs`.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/salud` | Estado del servicio y volumen de datos cargado. |
| `GET` | `/recomendaciones/{id_usuario}?k=10` | k recomendaciones para el usuario. |
| `POST` | `/interacciones` | Registra interacciones y actualiza el perfil. |
| `GET` | `/usuarios/{id_usuario}/historial` | Interacciones registradas del usuario. |
| `GET` | `/juegos?busqueda=portal` | Busca títulos del catálogo. |
| `GET` | `/juegos/{titulo}/similares?k=10` | Juegos parecidos según el espacio latente. |

### Ejemplos

```bash
# Recomendaciones para un usuario existente
curl "http://127.0.0.1:8000/recomendaciones/151603712?k=5"

# Usuario sin historial: responde con el ranking por popularidad
curl "http://127.0.0.1:8000/recomendaciones/424242?k=5"

# Registrar interacciones: el usuario queda dado de alta y pasa a personalizado
curl -X POST "http://127.0.0.1:8000/interacciones" \
  -H "Content-Type: application/json" \
  -d '{"id_usuario": 424242,
       "interacciones": [
         {"titulo": "Portal 2", "comportamiento": "play", "horas": 18.5},
         {"titulo": "Portal",   "comportamiento": "purchase"}
       ]}'
```

## Estructura

```
src/recomendador/     motor, independiente de cualquier framework web
  config.py           hiperparámetros y rutas
  datos.py            carga y preparación del dataset
  als.py              factorización de matrices (ALS implícito)
  popularidad.py      ranking no personalizado (arranque en frío)
  motor.py            fachada: estrategia, estado y persistencia
  evaluacion.py       validación offline con métricas de ranking
src/api/              capa HTTP (FastAPI)
scripts/              entrenamiento y evaluación por línea de comandos
tests/                pruebas del motor y de la API
datos/                dataset de interacciones
modelo/               artefacto entrenado (no versionado)
```

## Configuración

Los hiperparámetros por defecto están en `src/recomendador/config.py` y se pueden
sobrescribir por línea de comandos:

```bash
python scripts/entrenar.py --factores 192 --alfa 5 --iteraciones 20
python scripts/evaluar.py --k 20
```

Las rutas admiten las variables de entorno `RUTA_DATASET` y `DIRECTORIO_MODELO`.
