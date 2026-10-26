"""Análisis descriptivo de la Parte 1: hipótesis explícitas contrastadas SOLO con entrenamiento."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from src.pd_model.dataset import load_pd_dataset
from src.utils import FIGURES_DIR, RESULTS_DIR

ALPHA = 0.01          # nivel de significancia (Mann-Whitney)
MIN_EFFECT = 0.01     # variables continuas: |AUC univariado - 0,5|
MIN_DIFF_BIN = 0.01   # indicadores sí/no: diferencia absoluta de tasas de malos (1 punto porcentual)
FECHA_DESCARGA = "AAAA-MM-DD"   # COMPLETAR: fecha en que se descargó loan.csv de Kaggle


# Lista de hipótesis: variable, dirección esperada del riesgo y tipo (numérica o sí/no).
def hypotheses(df: pd.DataFrame) -> list[tuple]:
    """(id, enunciado, serie, dirección esperada del riesgo: +1 sube / -1 baja, tipo)."""
    derog = ((df["delinq_2yrs"] > 0) | (df["pub_rec"] > 0)).astype(float)
    return [
        ("H1", "Peor subcalificación de Lending Club (sub_grade_num alto) -> más riesgo", df["sub_grade_num"], +1, "num"),
        ("H2", "Mayor utilización del crédito rotativo (revol_util) -> más riesgo", df["revol_util"], +1, "num"),
        ("H3", "Mayor ingreso anual (annual_inc) -> menos riesgo", df["annual_inc"], -1, "num"),
        ("H4", "Más consultas recientes (inq_last_6mths) -> más riesgo", df["inq_last_6mths"], +1, "num"),
        ("H5", "Mayor monto del crédito (loan_amnt) -> más riesgo", df["loan_amnt"], +1, "num"),
        ("H6", "Mayor endeudamiento sobre ingreso (dti) -> más riesgo", df["dti"], +1, "num"),
        ("H7", "Tener moras o registros públicos previos (delinq_2yrs o pub_rec > 0) -> más riesgo", derog, +1, "bin"),
        ("H8", "Historial crediticio más largo (credit_history_months) -> menos riesgo", df["credit_history_months"], -1, "num"),
        ("H9", "No informar antigüedad laboral (emp_length faltante) -> más riesgo", df["emp_length"].isna().astype(float), +1, "bin"),
        ("H10", "Mayor antigüedad laboral (emp_length) -> menos riesgo", df["emp_length"], -1, "num"),
        ("H11", "Ingreso verificado -> menos riesgo", (df["verification_status"] != "Not Verified").astype(float), -1, "bin"),
        ("H12", "Crédito para pequeño negocio (purpose = small_business) -> más riesgo", (df["purpose"] == "small_business").astype(float), +1, "bin"),
    ]


# Contrasta una hipótesis en train: AUC univariado, Mann-Whitney y veredicto.
def contrast(x: pd.Series, y: pd.Series, expected: int, kind: str) -> dict:
    ok = x.notna()
    x, y = x[ok].astype(float).to_numpy(), y[ok].to_numpy()
    auc = roc_auc_score(y, x)
    p = mannwhitneyu(x[y == 1], x[y == 0], method="asymptotic").pvalue
    observed = 1 if auc > 0.5 else -1
    if kind == "bin":
        groups = x.astype(int)
        labels, rates = ["no", "sí"], [y[groups == 0].mean(), y[groups == 1].mean()]
        effect, min_effect = abs(rates[1] - rates[0]), MIN_DIFF_BIN
    else:
        q = pd.qcut(pd.Series(x), 5, labels=False, duplicates="drop").to_numpy()
        labels = [f"Q{int(i) + 1}" for i in np.unique(q)]
        rates = [y[q == i].mean() for i in np.unique(q)]
        effect, min_effect = abs(auc - 0.5), MIN_EFFECT
    if p >= ALPHA or effect < min_effect:
        verdict = "Sin evidencia suficiente"
    elif observed == expected:
        verdict = "Confirmada"
    else:
        verdict = "Refutada (efecto contrario)"
    return {"direccion_esperada": expected, "direccion_observada": observed, "auc_univariado": auc,
            "efecto": effect, "p_valor": p, "n": int(len(y)), "veredicto": verdict,
            "labels": labels, "rates": rates}


# Evalúa todas las hipótesis, guarda la tabla, el resumen de datos y las figuras.
def run_eda(mode: str = "matured", save: bool = True) -> pd.DataFrame:
    df = load_pd_dataset(mode)
    train = df[df["split"] == "train"]
    y = train["target"]

    rows = []
    for hid, text, x, expected, kind in hypotheses(train):
        res = contrast(x, y, expected, kind)
        rows.append({"id": hid, "hipotesis": text, **{k: v for k, v in res.items() if k not in ("labels", "rates")},
                     "tasa_menor_grupo": res["rates"][0], "tasa_mayor_grupo": res["rates"][-1],
                     "_labels": res["labels"], "_rates": res["rates"]})
    table = pd.DataFrame(rows)

    summary = {
        "n_registros": int(len(df)), "n_variables_guardadas": int(df.shape[1]),
        "periodo_originacion": [str(df["issue_d"].min().date()), str(df["issue_d"].max().date())],
        "particiones": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "tasa_malos": {s: float(df.loc[df["split"] == s, "target"].mean()) for s in ("train", "val", "test")},
        "fecha_descarga": FECHA_DESCARGA,
    }

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        table.drop(columns=["_labels", "_rates"]).to_csv(RESULTS_DIR / "pd_eda_hipotesis.csv", index=False)
        (RESULTS_DIR / "pd_eda_resumen_datos.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
        base = float(y.mean())
        for ax, (_, r) in zip(axes.ravel(), table.iterrows()):
            ax.bar(r["_labels"], r["_rates"], color="#4c72b0")
            ax.axhline(base, color="gray", ls="--", lw=1)
            ax.set_title(f"{r['id']}: {r['veredicto']}", fontsize=9)
        fig.supylabel("tasa de malos (train)")
        fig.suptitle("Tasa de malos por grupo (quintiles o sí/no); línea punteada = tasa global de train")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "pd_eda_hipotesis.png", dpi=150)
        plt.close(fig)

        q = df.groupby(df["issue_d"].dt.to_period("Q"))["target"].agg(["mean", "size"])
        q = q[q["size"] >= 100]   # se omiten trimestres con muy pocos créditos (p. ej. 2007Q2)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(q.index.astype(str), q["mean"], marker="o")
        ax.set_ylabel("tasa de malos")
        ax.set_title("Tasa de malos por trimestre de originación (todas las particiones)")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "pd_eda_tasa_por_trimestre.png", dpi=150)
        plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(table.drop(columns=["_labels", "_rates", "hipotesis"]).round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    run_eda()