# ============ CONFIGURACIÓN (todo se cambia aquí, sin tocar el resto del código) ============import os
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
    CSV_LOCAL="loan.csv",  # si existe se usa; si no, se descarga con kagglehub
    OUT_DIR="outputs",
    PROC_DIR="data/processed",
)


SEED = CFG["SEED"]
N_SEEDS = CFG["DEMO_N_SEEDS"] if CFG["DEMO"] else CFG["N_SEEDS"]
EPOCHS = CFG["DEMO_EPOCHS"] if CFG["DEMO"] else CFG["EPOCHS"]
np.random.seed(SEED); tf.random.set_seed(SEED); keras.utils.set_random_seed(SEED)
os.makedirs(CFG["OUT_DIR"], exist_ok=True); os.makedirs(CFG["PROC_DIR"], exist_ok=True)

def find_csv():
    if os.path.exists(CFG["CSV_LOCAL"]):
        return CFG["CSV_LOCAL"]



