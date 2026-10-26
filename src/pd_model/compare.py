"""Comparación justa LR vs MLP en test, con referencia de tasa base y bootstrap de la diferencia."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.utils import RESULTS_DIR, SEED

METRICS = ["auc", "gini", "ks", "brier", "log_loss"]


# Lee el JSON de métricas de un modelo.
def _load(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}_metricas.json").read_text(encoding="utf-8"))


# Tabla LR vs MLP contra la tasa base, costos y bootstrap de la diferencia en test.
def run_comparison(n_boot: int = 1000, save: bool = True) -> dict:
    lr, mlp = _load("pd_logistic"), _load("pd_mlp")
    seeds = {r["metrica"]: r for r in mlp["semillas"]["resumen_test"]}

    preds_lr = pd.read_parquet(RESULTS_DIR / "pd_logistic_predicciones.parquet")
    preds_mlp = pd.read_parquet(RESULTS_DIR / "pd_mlp_predicciones.parquet")
    t = (preds_lr[preds_lr["split"] == "test"][["id", "target", "pd"]].rename(columns={"pd": "pd_lr"})
         .merge(preds_mlp[preds_mlp["split"] == "test"][["id", "pd"]].rename(columns={"pd": "pd_mlp"}), on="id"))
    y, a, b = t["target"].to_numpy(), t["pd_lr"].to_numpy(), t["pd_mlp"].to_numpy()

    # Referencia: predecir siempre la tasa de malos de train
    p0 = lr["metricas"]["train"]["tasa_observada"]
    ref = {"auc": 0.5, "gini": 0.0, "ks": 0.0,
           "brier": float(brier_score_loss(y, np.full(len(y), p0))),
           "log_loss": float(log_loss(y, np.full(len(y), p0)))}

    metrics = pd.DataFrame({
        "metrica": METRICS,
        "referencia_tasa_base": [ref[m] for m in METRICS],
        "logistica": [lr["metricas"]["test"][m] for m in METRICS],
        "mlp_media_30_semillas": [seeds[m]["mean"] for m in METRICS],
        "mlp_std_semillas": [seeds[m]["std"] for m in METRICS],
        "mlp_modelo_guardado": [mlp["metricas_modelo_guardado"]["test"][m] for m in METRICS],
    })
    for col in ("logistica", "mlp_media_30_semillas"):
        metrics[f"mejora_brier_vs_base_{col}"] = np.nan
    brier_i = METRICS.index("brier")
    base_b = ref["brier"]
    metrics.loc[brier_i, "mejora_brier_vs_base_logistica"] = 1 - metrics.loc[brier_i, "logistica"] / base_b
    metrics.loc[brier_i, "mejora_brier_vs_base_mlp_media_30_semillas"] = 1 - metrics.loc[brier_i, "mlp_media_30_semillas"] / base_b

    costs = pd.DataFrame([
        {"concepto": "parametros", "logistica": lr["n_parametros"], "mlp": mlp["n_parametros"]},
        {"concepto": "configuraciones_evaluadas", "logistica": lr["busqueda"]["n_configuraciones"], "mlp": mlp["busqueda"]["n_configuraciones"]},
        {"concepto": "semillas", "logistica": 1, "mlp": mlp["semillas"]["n"]},
        {"concepto": "segundos_totales", "logistica": lr["busqueda"]["segundos_totales"], "mlp": mlp["segundos_totales"]},
        {"concepto": "interpretabilidad", "logistica": "coeficientes y scorecard directa", "mlp": "SHAP (aproximado)"},
    ])

    # Bootstrap pareado sobre test: MLP guardado menos logística
    rng = np.random.default_rng(SEED)
    d_auc, d_brier = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d_auc.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
        d_brier.append(brier_score_loss(y[i], b[i]) - brier_score_loss(y[i], a[i]))
    boot = {
        "n_boot": len(d_auc),
        "dif_auc_mlp_menos_lr": {"media": float(np.mean(d_auc)), "ic95": np.percentile(d_auc, [2.5, 97.5]).tolist()},
        "dif_brier_mlp_menos_lr": {"media": float(np.mean(d_brier)), "ic95": np.percentile(d_brier, [2.5, 97.5]).tolist()},
        "nota": "El bootstrap recoge la variabilidad de la muestra de test, no la de las semillas.",
    }

    if save:
        metrics.to_csv(RESULTS_DIR / "pd_comparacion_metricas.csv", index=False)
        costs.to_csv(RESULTS_DIR / "pd_comparacion_costos.csv", index=False)
        (RESULTS_DIR / "pd_comparacion_bootstrap.json").write_text(json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.round(4).to_string(index=False))
    print(costs.to_string(index=False))
    print(json.dumps(boot, indent=2, ensure_ascii=False))
    return {"metricas": metrics, "costos": costs, "bootstrap": boot}


if __name__ == "__main__":
    run_comparison()