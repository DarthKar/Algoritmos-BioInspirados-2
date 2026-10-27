"""Modelo de referencia de la Parte 1: regresión logística con penalización L2.

Búsqueda de C (inverso de la fuerza de regularización) en una rejilla pequeña,
eligiendo por log-loss en validación. Se reporta cuántas configuraciones se
entrenaron y el tiempo total (regla de comparación justa).
"""
import json
import time
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from src.features import DEFAULT_LC_FEATURES
from src.pd_model.dataset import load_pd_dataset, make_design_matrices
from src.pd_model.metrics import calibration_in_the_large, calibration_table, evaluate
from src.pd_model.models import fit_logistic, logistic_n_params
from src.utils import MODELS_DIR, RESULTS_DIR, SEED, records, set_seed


# Parámetros del experimento de la logística (rejilla de C, variables, semilla).
@dataclass
class LogisticConfig:
    mode: str = "matured"
    lc_features: tuple[str, ...] = DEFAULT_LC_FEATURES
    use_geography: bool = False
    c_grid: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    max_iter: int = 2000
    calibration_bins: int = 10
    seed: int = SEED
    name: str = "pd_logistic"


# Busca C en la rejilla, elige por log-loss de validación, evalúa y guarda todo.
def run_logistic_baseline(cfg: LogisticConfig = LogisticConfig(), save: bool = True) -> dict:
    set_seed(cfg.seed)
    start = time.perf_counter()
    data = make_design_matrices(load_pd_dataset(cfg.mode), cfg.lc_features, cfg.use_geography)
    X, y = data["X"], data["y"]

    search, best, best_loss = [], None, np.inf
    for c in cfg.c_grid:
        t0 = time.perf_counter()
        model = fit_logistic(X["train"], y["train"], C=c, max_iter=cfg.max_iter, seed=cfg.seed)
        p_val = model.predict_proba(X["val"])[:, 1]
        loss = log_loss(y["val"], p_val)
        search.append({"C": c, "val_logloss": loss, "val_auc": roc_auc_score(y["val"], p_val),
                       "segundos": time.perf_counter() - t0})
        if loss < best_loss:
            best, best_loss = model, loss

    probs = {s: best.predict_proba(X[s])[:, 1] for s in X}
    results = {
        "config": asdict(cfg),
        "variables_numericas": data["numeric"],
        "variables_categoricas": data["categorical"],
        "n_columnas_X": len(data["feature_names"]),
        "n_parametros": logistic_n_params(best),
        "busqueda": {"n_configuraciones": len(cfg.c_grid), "criterio": "log-loss en validación",
                     "resultados": search, "mejor_C": float(best.C),
                     "segundos_totales": time.perf_counter() - start},
        "metricas": {s: {**evaluate(y[s], probs[s]), "log_loss": float(log_loss(y[s], probs[s])),
                         **calibration_in_the_large(y[s], probs[s]), "n": int(len(y[s]))} for s in X},
        "calibracion_test": calibration_table(y["test"], probs["test"], n_bins=cfg.calibration_bins).to_dict("records"),
    }

    if save:
        _save(cfg, data, best, probs, results)
    return results


# Tabla de coeficientes estandarizados y odds ratios, ordenada por magnitud.
def coefficient_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Coeficientes sobre variables estandarizadas: comparables entre sí en magnitud."""
    coef = pd.DataFrame({"variable": feature_names, "coeficiente": model.coef_.ravel()})
    coef["odds_ratio"] = np.exp(coef["coeficiente"])
    coef = coef.reindex(coef["coeficiente"].abs().sort_values(ascending=False).index)
    intercept = pd.DataFrame({"variable": ["(intercepto)"], "coeficiente": model.intercept_, "odds_ratio": np.exp(model.intercept_)})
    return pd.concat([intercept, coef], ignore_index=True)


# Guarda modelo (con preprocesador), predicciones y un JSON con métricas y coeficientes.
def _save(cfg: LogisticConfig, data: dict, model, probs: dict, results: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Preprocesador + modelo juntos: la app los carga y predice sin reentrenar
    pipeline = Pipeline([("preprocessor", data["preprocessor"]), ("model", model)])
    joblib.dump(pipeline, MODELS_DIR / f"{cfg.name}.joblib")

    preds = pd.concat([
        pd.DataFrame({"id": data["ids"][s], "split": s, "target": data["y"][s], "pd": probs[s]}) for s in probs
    ], ignore_index=True)
    preds.to_parquet(RESULTS_DIR / f"{cfg.name}_predicciones.parquet", index=False)
    results["coeficientes"] = records(coefficient_table(model, data["feature_names"]))
    (RESULTS_DIR / f"{cfg.name}_metricas.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
