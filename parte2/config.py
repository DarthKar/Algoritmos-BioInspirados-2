# ============ CONFIGURACIÓN DE LA PARTE 2 ============
from pathlib import Path

from parte1.comun.utils import RESULTS_DIR

PART2_ROOT = Path(__file__).resolve().parent

CFG = {
    "SEED": 42,
    "N_SEEDS": 30,
    "EPOCHS": 100,
    "BATCH": 256,
    "PATIENCE": 8,
    "PD_MODE": "matured",
    "OUT_DIR": PART2_ROOT / "outputs",
    "FULL_OUT_DIR": PART2_ROOT / "outputs",
    "DEMO_OUT_DIR": PART2_ROOT / "outputs_demo",
    "PROC_DIR": PART2_ROOT / "data" / "processed",
    "CACHE_VERSION": 2,
    "MLP_GRID": [
        ((32,), 0.0, 1e-3),
        ((64,), 0.0, 1e-3),
        ((64, 32), 0.1, 1e-3),
        ((128, 64), 0.2, 1e-3),
        ((128, 64, 32), 0.2, 5e-4),
        ((256, 128), 0.3, 5e-4),
    ],
    "MLP_DEFAULT": ((256, 128), 0.3, 5e-4),
    "LINEAR_FIT_INTERCEPT": True,
    "LINEAR_POSITIVE": False,
    "DEMO_EPOCHS": 10,
    "DEMO_N_SEEDS": 2,
    "DEMO_BATCH": 256,
    "DEMO_PATIENCE": 5,
    "DEMO_SEED": 42,
    "DEMO_ROWS_PER_SPLIT": 800,
    "DEMO_MLP_GRID": [
        ((64, 32), 0.1, 1e-3),
        ((128, 64), 0.2, 1e-3),
    ],
    "PD_PREDICTIONS": {
        "logistic": RESULTS_DIR / "pd_logistic_predicciones.parquet",
        "mlp": RESULTS_DIR / "pd_mlp_predicciones.parquet",
    },
}
