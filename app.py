"""Punto de entrada para Vercel: expone la aplicación FastAPI de src/api/main.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from api.main import app  # noqa: E402,F401
