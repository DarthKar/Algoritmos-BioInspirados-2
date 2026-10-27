"""Contraste de las hipótesis del EDA con lo que aprendieron los modelos.

El EDA (src/pd_model/eda.py) mira cada variable sola. Aquí se pregunta lo mismo a
los modelos ya entrenados, pero controlando el resto de variables: a cada cliente
de entrenamiento se le pone la variable en un valor "bajo" y luego en uno "alto",
dejando todo lo demás igual, y se promedia el cambio en la PD (efecto ceteris
paribus, similar a una dependencia parcial; Friedman, 2001). Si el signo coincide
con el esperado, el modelo confirma la hipótesis una vez descontado el resto.
No reentrena nada: usa los modelos guardados.
"""
import joblib
import numpy as np
import pandas as pd

from src.pd_model.dataset import load_pd_dataset
from src.pd_model.mlp import load_mlp
from src.pd_model.models import predict_proba
from src.utils import MODELS_DIR, read_analysis, records, update_analysis

MIN_EFFECT_PP = 0.5   # cambio mínimo en la PD media (puntos porcentuales) para no llamarlo despreciable


# Escenarios "bajo" y "alto" de cada hipótesis: columnas a fijar y valores (cuartiles de train).
def scenarios(train: pd.DataFrame) -> list[dict]:
    q = lambda c, p: float(train[c].quantile(p))   # noqa: E731
    clean_record = {"delinq_2yrs": 0, "pub_rec": 0, "never_delinquent": 1, "mths_since_last_delinq": None}
    one_delinq = {"delinq_2yrs": 1, "pub_rec": 0, "never_delinquent": 0, "mths_since_last_delinq": 12}
    num = lambda c: {"bajo": {c: q(c, .25)}, "alto": {c: q(c, .75)}, "detalle": f"{c}: P25 {q(c, .25):g} -> P75 {q(c, .75):g}"}   # noqa: E731
    return [
        {"id": "H1", "esperado": +1, "shap": ["sub_grade_num"], **num("sub_grade_num")},
        {"id": "H2", "esperado": +1, "shap": ["revol_util"], **num("revol_util")},
        {"id": "H3", "esperado": -1, "shap": ["annual_inc"], **num("annual_inc")},
        {"id": "H4", "esperado": +1, "shap": ["inq_last_6mths"], "bajo": {"inq_last_6mths": 0}, "alto": {"inq_last_6mths": 2},
         "detalle": "inq_last_6mths: 0 -> 2"},
        {"id": "H5", "esperado": +1, "shap": ["loan_amnt"], **num("loan_amnt")},
        {"id": "H6", "esperado": +1, "shap": ["dti"], **num("dti")},
        {"id": "H7", "esperado": +1, "shap": ["delinq_2yrs", "pub_rec", "never_delinquent", "mths_since_last_delinq"],
         "bajo": clean_record, "alto": one_delinq, "detalle": "sin moras -> 1 mora hace 12 meses"},
        {"id": "H8", "esperado": -1, "shap": ["credit_history_months"], **num("credit_history_months")},
        {"id": "H9", "esperado": +1, "shap": ["emp_length"], "bajo": {"emp_length": q("emp_length", .5)},
         "alto": {"emp_length": np.nan}, "detalle": f"emp_length: mediana {q('emp_length', .5):g} -> faltante"},
        {"id": "H10", "esperado": -1, "shap": ["emp_length"], **num("emp_length")},
        {"id": "H11", "esperado": -1, "shap": ["verification_status"], "bajo": {"verification_status": "Not Verified"},
         "alto": {"verification_status": "Verified"}, "detalle": "Not Verified -> Verified"},
        {"id": "H12", "esperado": +1, "shap": ["purpose"], "bajo": {"purpose": "debt_consolidation"},
         "alto": {"purpose": "small_business"}, "detalle": "debt_consolidation -> small_business"},
    ]


# Copia de los datos con las columnas fijadas; mantiene coherente "meses desde la última mora".
def _apply(df: pd.DataFrame, values: dict) -> pd.DataFrame:
    out = df.copy()
    for col, v in values.items():
        out[col] = out["credit_history_months"] if v is None else v
    if "credit_history_months" in values:   # quien nunca tuvo mora: meses desde la última = historia
        never = out["never_delinquent"] == 1
        out.loc[never, "mths_since_last_delinq"] = out.loc[never, "credit_history_months"]
    return out


# Veredicto del modelo según signo y tamaño del efecto.
def _verdict(delta_pp: float, expected: int) -> str:
    if abs(delta_pp) < MIN_EFFECT_PP:
        return "Efecto despreciable"
    return "Confirmada" if np.sign(delta_pp) == expected else "Contraria"


# Efecto controlado de cada hipótesis en LR y MLP, unido con el EDA y SHAP (en pd_analisis.json).
def run_hypothesis_check(mode: str = "matured", save: bool = True) -> pd.DataFrame:
    df = load_pd_dataset(mode)
    train = df[df["split"] == "train"]
    lr = joblib.load(MODELS_DIR / "pd_logistic.joblib")
    pre_mlp, mlp = load_mlp("pd_mlp")
    models = {"lr": lambda d: lr.predict_proba(d)[:, 1], "mlp": lambda d: predict_proba(mlp, pre_mlp.transform(d))}

    eda = pd.DataFrame(read_analysis("eda_hipotesis")).set_index("id")
    shap_rank = pd.DataFrame(read_analysis("shap_importancia")).set_index("variable")["rango_mlp"]

    rows = []
    for s in scenarios(train):
        low, high = _apply(train, s["bajo"]), _apply(train, s["alto"])
        row = {"id": s["id"], "hipotesis": eda.loc[s["id"], "hipotesis"], "escenario": s["detalle"],
               "veredicto_eda": eda.loc[s["id"], "veredicto"],
               "eda_tasa_grupo_bajo": eda.loc[s["id"], "tasa_menor_grupo"],
               "eda_tasa_grupo_alto": eda.loc[s["id"], "tasa_mayor_grupo"]}
        for name, f in models.items():
            delta = 100 * float(np.mean(f(high)) - np.mean(f(low)))
            row[f"delta_pd_{name}_pp"] = delta
            row[f"veredicto_{name}"] = _verdict(delta, s["esperado"])
        row["mejor_rango_shap_mlp"] = int(min(shap_rank.get(v, 99) for v in s["shap"]))
        rows.append(row)
    table = pd.DataFrame(rows)

    if save:
        update_analysis("hipotesis_vs_modelos", records(table))
    cols = ["id", "escenario", "veredicto_eda", "delta_pd_lr_pp", "veredicto_lr", "delta_pd_mlp_pp", "veredicto_mlp",
            "mejor_rango_shap_mlp"]
    print(table[cols].round(2).to_string(index=False))
    return table


if __name__ == "__main__":
    run_hypothesis_check()
