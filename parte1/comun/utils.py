"""Rutas del proyecto, archivo único de análisis y control de semillas."""
import json
import os
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]   # raíz del repositorio
PART1_ROOT = REPO_ROOT / "parte1"
DATA_RAW = REPO_ROOT / "data" / "raw"             # loan.csv, compartido con la Parte 2
DATA_PROCESSED = PART1_ROOT / "datos" / "procesados"

# Carpeta de salida de modelos, figuras y resultados. El modo demo la cambia
# (PD_SALIDA=salidas_demo) para no sobrescribir los resultados completos.
OUTPUT_ROOT = PART1_ROOT / os.environ.get("PD_SALIDA", "salidas")
MODELS_DIR = OUTPUT_ROOT / "modelos"
FIGURES_DIR = OUTPUT_ROOT / "figuras"
RESULTS_DIR = OUTPUT_ROOT / "resultados"
ANALYSIS_FILE = RESULTS_DIR / "pd_analisis.json"

RAW_LOANS_CSV = DATA_RAW / "loan.csv"
DATA_DICTIONARY = DATA_RAW / "LCDataDictionary.xlsx"

SEED = 42


# Fija todas las semillas (Python, NumPy, PyTorch) para que los resultados se repitan.
def set_seed(seed: int = SEED) -> None:
    """Fija las semillas de Python, NumPy y PyTorch (si está instalado)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# Convierte un DataFrame en lista de dicts con tipos nativos (apto para JSON).
def records(df) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


# Lee una sección de results/pd_analisis.json (EDA, SHAP, comparación, scorecard...).
def read_analysis(section: str):
    data = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8")) if ANALYSIS_FILE.exists() else {}
    if section not in data:
        raise KeyError(f"Falta la sección '{section}' en {ANALYSIS_FILE}; ejecute antes esa etapa.")
    return data[section]


# Escribe o reemplaza una sección de results/pd_analisis.json sin tocar las demás.
def update_analysis(section: str, value) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8")) if ANALYSIS_FILE.exists() else {}
    data[section] = value
    ANALYSIS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
