"""Transformación de probabilidad a puntaje (scorecard).

Convención: puntaje más alto = menor riesgo. Con odds = P(bueno)/P(malo):

    score = offset + factor · ln(odds)
    factor = PDO / ln(2)
    offset = base_score - factor · ln(base_odds)

Así, un cliente con odds = base_odds recibe base_score, y cada PDO puntos las
odds se duplican.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScorecardParams:
    base_score: float = 600.0
    base_odds: float = 7.0    # ~ odds promedio del portafolio en train (12,8 % de malos)
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)


def prob_to_score(pd_bad, params: ScorecardParams, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(pd_bad, dtype=float), eps, 1 - eps)
    return params.offset + params.factor * np.log((1 - p) / p)


def score_to_prob(score, params: ScorecardParams) -> np.ndarray:
    log_odds = (np.asarray(score, dtype=float) - params.offset) / params.factor
    return 1 / (1 + np.exp(log_odds))


def logistic_points(coef: np.ndarray, intercept: float, x: np.ndarray, feature_names: list[str],
                    params: ScorecardParams) -> pd.DataFrame:
    """Puntos por variable para una regresión logística con variables continuas.

    El modelo da logit(PD) = β0 + Σ βj xj, y ln(odds) = -logit(PD), entonces

        score = (offset - factor·β0) + Σ (-factor · βj · xj)

    El primer término son los puntos base; el resto, los puntos que aporta cada
    variable (en la escala ya preprocesada). Devuelve una fila por cliente.
    """
    x = np.atleast_2d(x)
    points = -params.factor * x * np.asarray(coef).ravel()
    df = pd.DataFrame(points, columns=feature_names)
    df.insert(0, "puntos_base", params.offset - params.factor * float(intercept))
    df["score"] = df.sum(axis=1)
    return df

def scorecard_table(coef, intercept, preprocessor, params: ScorecardParams):
    """Puntos por variable en unidades originales (después de log1p/topes, antes de estandarizar).

    Como x_std = (x - media) / desv, el modelo queda
        score = puntos_base + sum_j puntos_por_unidad_j * x_j
    con puntos_por_unidad_j = -factor * beta_j / desv_j, y las medias absorbidas
    en los puntos base.
    Numéricas: puntos por unidad de la variable (en log1p si es de LOG_FEATURES).
    Categóricas: puntos si la categoría está presente (la de referencia vale 0).
    """
    coef = np.asarray(coef).ravel()
    names = np.asarray(preprocessor.get_feature_names_out())
    mean, scale = np.zeros(len(names)), np.ones(len(names))
    for name, sl in preprocessor.output_indices_.items():
        trans = preprocessor.named_transformers_[name]
        if hasattr(trans, "named_steps") and "scale" in trans.named_steps:
            mean[sl] = trans.named_steps["scale"].mean_
            scale[sl] = trans.named_steps["scale"].scale_
    per_unit = -params.factor * coef / scale
    base_points = params.offset - params.factor * (float(intercept) - np.sum(coef * mean / scale))
    table = pd.DataFrame({"variable": names, "puntos_por_unidad": per_unit, "desv_est": scale, "media": mean})
    return base_points, table