"""Scorecard de la Parte 1: la del MLP (modelo propuesto) y la de la logística como comparación.

Para cada modelo: puntaje de cada crédito (PD -> puntaje con base, odds y PDO),
bandas de riesgo en prueba y puntos por variable obtenidos con SHAP
(parte1/puntaje/shap_points.py) sobre una muestra de toda la base, para que
aparezcan también los créditos a 60 meses. Para la logística se guarda además la tabla
exacta de puntos por unidad de cada variable, posible porque es lineal.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from parte1.comun.utils import MODELS_DIR, SEED, records, update_analysis
from parte1.inference import load_registered_model
from parte1.preparacion.dataset import load_pd_dataset
from parte1.puntaje.shap_points import points_by_variable
from parte1.puntaje.transform import ScorecardParams, prob_to_score, scorecard_table

N_TRAMOS = 5   # tramos por variable continua en la tabla global de puntos


# Carga los dos modelos guardados con la misma interfaz que usa la app.
def load_models() -> dict:
    return {"pd_mlp": load_registered_model(str(MODELS_DIR / "pd_mlp.pt"), "pytorch_mlp",
                                            str(MODELS_DIR / "pd_mlp_preprocessor.joblib")),
            "pd_logistic": load_registered_model(str(MODELS_DIR / "pd_logistic.joblib"))}


# Puntaje de cada crédito y bandas por decil en prueba (banda 1 = más riesgo).
def scores_and_bands(model, parts: dict, params: ScorecardParams) -> tuple[dict, pd.DataFrame]:
    scores = {}
    for s, p in parts.items():
        pd_bad = model.predict_proba(p)[:, 1]
        scores[s] = pd.DataFrame({"target": p["target"].to_numpy(), "pd": pd_bad, "score": prob_to_score(pd_bad, params)})
    t = scores["test"].copy()
    t["banda"] = pd.qcut(t["score"], 10, labels=False, duplicates="drop") + 1
    bands = (t.groupby("banda").agg(n=("target", "size"), score_min=("score", "min"), score_max=("score", "max"),
                                    pd_media=("pd", "mean"), tasa_malos=("target", "mean")).reset_index())
    return scores, bands


# Puntos medios por tramo de cada variable: la tabla global de la scorecard del MLP.
def points_by_bin(rows: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    fmt = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"   # noqa: E731
    out = []
    for var in points.columns:
        x = rows[var]
        if not pd.api.types.is_numeric_dtype(x) or x.nunique() <= 6:
            keys = x.astype(str).where(x.notna(), "faltante")
            labels = {k: k for k in keys.unique()}
        else:   # tramos por quintiles, etiquetados con el mínimo y máximo reales de cada uno
            code = pd.qcut(x, N_TRAMOS, labels=False, duplicates="drop")
            keys = code.astype("Int64").astype(str).where(x.notna(), "faltante")
            labels = {str(int(c)): f"{fmt(x[code == c].min())} a {fmt(x[code == c].max())}" for c in code.dropna().unique()}
            labels["faltante"] = "faltante"
        g = points[var].groupby(keys).agg(["size", "mean"])
        order = sorted(g.index, key=lambda k: (k == "faltante", float(k) if k.replace(".", "", 1).isdigit() else k))
        out += [{"variable": var, "tramo": labels[k], "n": int(g.loc[k, "size"]), "puntos_medios": float(g.loc[k, "mean"])}
                for k in order]
    return pd.DataFrame(out)


# Construye las dos scorecards, las guarda en pd_analisis.json y devuelve el resumen del MLP.
def run_scorecard(params: ScorecardParams = ScorecardParams(), mode: str = "matured", n_explicar: int = 2000,
                  n_fondo: int = 100, max_evals: int = 200, save: bool = True) -> dict:
    df = load_pd_dataset(mode)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    background = parts["train"].sample(n_fondo, random_state=SEED)
    # Créditos cuyos puntos se describen: de toda la muestra, para incluir los de 60 meses (solo hay en entrenamiento)
    rows = df.sample(min(n_explicar, len(df)), random_state=SEED)
    models = load_models()

    summary = {}
    for name, model in models.items():
        scores, bands = scores_and_bands(model, parts, params)
        base, points = points_by_variable(model, rows, background, params, max_evals=max_evals, seed=SEED)
        score_rows = prob_to_score(model.predict_proba(rows)[:, 1], params)
        additivity = float(np.abs(base + points.sum(axis=1).to_numpy() - score_rows).max())
        summary[name] = {
            "puntos_base": float(base),
            "error_aditividad_max": additivity,
            "auc_test_con_puntaje": float(roc_auc_score(scores["test"]["target"], -scores["test"]["score"])),
            "percentiles_puntaje": {s: scores[s]["score"].quantile([.01, .25, .5, .75, .99]).round(1).tolist() for s in scores},
            "bandas_test": records(bands),
            "importancia_puntos": points.abs().mean().sort_values(ascending=False).rename("puntos_abs_medios"),
            "tabla_tramos": points_by_bin(rows, points),
        }

    # Tabla exacta de la logística: puntos por unidad de cada variable (posible porque es lineal)
    lr = models["pd_logistic"]
    pre, clf = lr.named_steps["preprocessor"], lr.named_steps["model"]
    _, unit_table = scorecard_table(clf.coef_, clf.intercept_[0], pre, params)
    std_orig = pre.transform(parts["train"]).std(axis=0) * unit_table["desv_est"].to_numpy()
    unit_table["impacto_1sd"] = unit_table["puntos_por_unidad"].abs() * std_orig
    unit_table = unit_table.sort_values("impacto_1sd", ascending=False).reset_index(drop=True)

    imp = pd.DataFrame({"mlp": summary["pd_mlp"]["importancia_puntos"],
                        "logistica": summary["pd_logistic"]["importancia_puntos"]}).sort_values("mlp", ascending=False)
    tasa_train = float(parts["train"]["target"].mean())
    common = {"parametros": {"base_score": params.base_score, "base_odds": params.base_odds, "pdo": params.pdo,
                             "factor": float(params.factor), "offset": float(params.offset)},
              "justificacion_base_odds": {"tasa_malos_train": tasa_train, "odds_train": (1 - tasa_train) / tasa_train},
              "n_explicados": len(rows), "n_fondo": n_fondo}
    mlp = summary["pd_mlp"]
    result = {
        **common, "modelo": "pd_mlp",
        **{k: v for k, v in mlp.items() if k not in ("importancia_puntos", "tabla_tramos")},
        "tabla_tramos": records(mlp["tabla_tramos"]),
        "importancia_puntos": records(imp.reset_index(names="variable")),
    }
    lr_sum = summary["pd_logistic"]
    comparison = {
        **common, "modelo": "pd_logistic",
        **{k: v for k, v in lr_sum.items() if k not in ("importancia_puntos", "tabla_tramos")},
        "tabla_por_unidad": records(unit_table),
        "tabla_tramos": records(lr_sum["tabla_tramos"]),
    }
    if save:
        update_analysis("scorecard", result)
        update_analysis("scorecard_logistica", comparison)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tabla_tramos", "importancia_puntos", "bandas_test")},
                     indent=2, ensure_ascii=False))
    print("\nPuntos medios absolutos por variable (MLP y logística):")
    print(imp.round(2).head(12).to_string())
    print("\nBandas en prueba con el puntaje del MLP:")
    print(pd.DataFrame(result["bandas_test"]).round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    run_scorecard()
