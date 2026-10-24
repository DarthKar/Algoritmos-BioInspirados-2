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
    base_odds: float = 50.0   # 50:1 buenos por cada malo en el puntaje base
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
