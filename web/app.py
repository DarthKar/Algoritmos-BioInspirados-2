"""Punto de entrada cloud: dependencias ligeras en web/requirements.txt."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent.parent / "app.py"), run_name="__main__")