# ============ CONFIGURACIÓN (todo se cambia aquí, sin tocar el resto del código) ============
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Rutas absolutas: el script funciona igual sin importar desde qué carpeta se ejecute
PARTE2_DIR = Path(__file__).resolve().parent
REPO_DIR = PARTE2_DIR.parent

CFG = dict(
    DEMO=False,            # True: muestra de los datos + pocas semillas/épocas (< 2 min)
    DEMO_FRAC=0.10,
    SEED=42,
    N_SEEDS=30,            # semillas para la configuración final de la red
    DEMO_N_SEEDS=3,
    EPOCHS=100, DEMO_EPOCHS=25,
    BATCH=256, PATIENCE=8,
    TRAIN_END="2013-01-01",   # entrenamiento: issue_d <  TRAIN_END
    VAL_END="2013-07-01",     # validación:    TRAIN_END <= issue_d < VAL_END ; prueba: issue_d >= VAL_END
    CSV_LOCAL=str(REPO_DIR / "data" / "raw" / "loan.csv"),   # el mismo archivo que usa la Parte 1
    OUT_DIR=str(PARTE2_DIR / "outputs"),
    PROC_DIR=str(PARTE2_DIR / "data" / "processed"),
)


SEED = CFG["SEED"]
N_SEEDS = CFG["DEMO_N_SEEDS"] if CFG["DEMO"] else CFG["N_SEEDS"]
EPOCHS = CFG["DEMO_EPOCHS"] if CFG["DEMO"] else CFG["EPOCHS"]
np.random.seed(SEED); tf.random.set_seed(SEED); keras.utils.set_random_seed(SEED)
os.makedirs(CFG["OUT_DIR"], exist_ok=True); os.makedirs(CFG["PROC_DIR"], exist_ok=True)


# Ruta de loan.csv; falla con un mensaje claro si no se ha descargado.
def find_csv():
    if os.path.exists(CFG["CSV_LOCAL"]):
        return CFG["CSV_LOCAL"]
    raise FileNotFoundError(f"No se encontró {CFG['CSV_LOCAL']}. Descárgalo de Kaggle "
                            "(ver README, sección Datos) y cópialo en data/raw/.")
