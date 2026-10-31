"""Puntos de la scorecard por variable para cualquier modelo, a partir de SHAP.

El puntaje es lineal en el logit de la PD:

    puntaje = offset - factor · logit(PD)

y SHAP reparte el logit de cada cliente de forma exacta y aditiva:

    logit = φ0 + Σ_j φ_j        (φ0 = logit medio del fondo de referencia)

Por lo tanto  puntaje = (offset - factor·φ0) + Σ_j (-factor·φ_j):
el primer término son los puntos base y cada variable aporta -factor·φ_j puntos.
Así la scorecard puede representar la red neuronal, con variables continuas,
y no solo la regresión logística. Se usa el estimador por permutaciones de SHAP,
que cumple la aditividad exacta para cada cliente.
"""
import numpy as np
import pandas as pd
import shap
import torch

from parte1.puntaje.transform import ScorecardParams


# Variable original a la que pertenece una columna preprocesada (one-hot o indicador de faltante).
def original_variable(name: str, categorical: list[str]) -> str:
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):]
    for c in categorical:
        if name.startswith(c + "_"):
            return c
    return name


# Preprocesador del modelo y función que devuelve el logit de la PD desde la matriz preprocesada.
def logit_function(model):
    """Acepta el MLPAdapter de la app (preprocesador + red) o el Pipeline de la logística."""
    if hasattr(model, "network"):
        network = model.network

        # Logit de la red para un lote de filas preprocesadas.
        def f(Z):
            with torch.no_grad():
                return network(torch.as_tensor(np.asarray(Z, dtype=np.float32))).numpy().astype(float)
        return model.preprocessor, f
    return model.named_steps["preprocessor"], model.named_steps["model"].decision_function


# Puntos base y puntos por variable de cada fila de `rows`.
def points_by_variable(model, rows: pd.DataFrame, background: pd.DataFrame, params: ScorecardParams,
                       max_evals: int = 200, seed: int = 42) -> tuple[float, pd.DataFrame]:
    """Devuelve (puntos base, DataFrame filas × variables originales con los puntos de cada una).

    Los puntos base más la suma de cada fila reproducen el puntaje del modelo.
    """
    pre, f = logit_function(model)
    names = list(pre.get_feature_names_out())
    categorical = next((cols for n, _, cols in pre.transformers_ if n == "cat"), [])
    groups = [original_variable(n, categorical) for n in names]

    Z_bg, Z = pre.transform(background), pre.transform(rows)
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(Z_bg, max_samples=len(Z_bg)), seed=seed)
    explanation = explainer(Z, max_evals=max(max_evals, 2 * Z.shape[1] + 1), silent=True)
    phi = pd.DataFrame(explanation.values, columns=names, index=rows.index)
    grouped = phi.T.groupby(groups, sort=False).sum().T          # suma las columnas de una misma variable
    base_points = params.offset - params.factor * float(np.mean(explanation.base_values))
    return base_points, -params.factor * grouped
