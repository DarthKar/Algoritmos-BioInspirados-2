# ============ CONFIGURACIÓN DE LA PARTE 2 ============
from pathlib import Path

from src.utils import RESULTS_DIR

PART2_ROOT = Path(__file__).resolve().parent

CFG = {
    "DEMO": False,          # True reduce épocas y semillas; no reduce ni cambia las particiones.
    "SEED": 42,
    "N_SEEDS": 30,
    "DEMO_N_SEEDS": 3,
    "EPOCHS": 100,
    "DEMO_EPOCHS": 25,
    "BATCH": 256,
    "PATIENCE": 8,
    "PD_MODE": "matured",
    "OUT_DIR": PART2_ROOT / "outputs",
    "PROC_DIR": PART2_ROOT / "data" / "processed",
    "CACHE_VERSION": 2,
    "PD_PREDICTIONS": {
        "logistic": RESULTS_DIR / "pd_logistic_predicciones.parquet",
        "mlp": RESULTS_DIR / "pd_mlp_predicciones.parquet",
    },
}

CFG["N_SEEDS"] = CFG["DEMO_N_SEEDS"] if CFG["DEMO"] else CFG["N_SEEDS"]
CFG["EPOCHS"] = CFG["DEMO_EPOCHS"] if CFG["DEMO"] else CFG["EPOCHS"]
