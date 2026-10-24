"""Rutas del proyecto y control de semillas."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

RAW_LOANS_CSV = DATA_RAW / "loan.csv"
DATA_DICTIONARY = DATA_RAW / "LCDataDictionary.xlsx"

SEED = 42


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
