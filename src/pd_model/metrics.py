"""Métricas de discriminación y calibración."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


# KS: máxima separación entre las distribuciones de buenos y malos.
def ks_statistic(y_true, y_prob) -> float:
    """KS = max |F_malos(s) - F_buenos(s)| = max (TPR - FPR) sobre los umbrales."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# Calcula AUC, Gini, KS y Brier de un conjunto.
def evaluate(y_true, y_prob) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_prob)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


# PD media predicha contra tasa observada por decil.
def calibration_table(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Probabilidad media predicha vs. tasa observada de malos por bin."""
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return pd.DataFrame({"prob_predicha": predicted, "tasa_observada": observed})


# Compara la PD media predicha con la tasa de malos total.
def calibration_in_the_large(y_true, y_prob) -> dict[str, float]:
    """PD media predicha vs. tasa observada de malos en todo el conjunto."""
    observed, predicted = float(np.mean(y_true)), float(np.mean(y_prob))
    return {"pd_media": predicted, "tasa_observada": observed, "diferencia": predicted - observed}


# Media, desviación, mejor y peor de cada métrica entre semillas.
def summarize_seeds(results: list[dict[str, float]]) -> pd.DataFrame:
    """Media, desviación estándar, mejor y peor de cada métrica entre semillas.

    Para Brier y log-loss "mejor" es el mínimo; para el resto, el máximo.
    """
    df = pd.DataFrame(results)
    summary = df.agg(["mean", "std", "max", "min"]).T
    lower_is_better = summary.index.isin(["brier", "log_loss"])
    summary["mejor"] = np.where(lower_is_better, summary["min"], summary["max"])
    summary["peor"] = np.where(lower_is_better, summary["max"], summary["min"])
    return summary[["mean", "std", "mejor", "peor"]]
