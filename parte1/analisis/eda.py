"""Análisis descriptivo de la Parte 1: hipótesis explícitas contrastadas SOLO con entrenamiento."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from parte1.preparacion.dataset import load_pd_dataset
from parte1.preparacion.features import select_features
from parte1.comun.plotting import MUTED, SERIES, apply_style
from parte1.comun.utils import FIGURES_DIR, records, update_analysis

ALPHA = 0.01          # nivel de significancia (Mann-Whitney)
MIN_EFFECT = 0.01     # variables continuas: |AUC univariado - 0,5|
MIN_DIFF_BIN = 0.01   # indicadores sí/no: diferencia absoluta de tasas de malos (1 punto porcentual)
FECHA_DESCARGA = "2026-10-24"   # fecha en que se descargó loan.csv de Kaggle


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
        ("H13", "Plazo de 60 meses (term) -> más riesgo", (df["term"] == 60).astype(float), +1, "bin"),
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


# Calidad de cada variable numérica de entrada en entrenamiento: faltantes, ceros, negativos y colas.
def data_quality(train: pd.DataFrame) -> pd.DataFrame:
    """Resumen para justificar la limpieza: qué falta, qué es casi siempre cero y dónde hay colas largas."""
    numeric, _ = select_features(train)
    rows = []
    for col in numeric:
        x = train[col]
        q = x.quantile([0.01, 0.5, 0.99])
        rows.append({"variable": col, "faltantes": float(x.isna().mean()), "ceros": float((x == 0).mean()),
                     "negativos": int((x < 0).sum()), "minimo": float(x.min()), "p1": float(q[0.01]),
                     "mediana": float(q[0.5]), "p99": float(q[0.99]), "maximo": float(x.max())})
    return pd.DataFrame(rows)


# Evalúa todas las hipótesis; guarda datos e hipótesis en pd_analisis.json y las figuras.
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
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        apply_style()
        update_analysis("datos", summary)
        update_analysis("eda_hipotesis", records(table.drop(columns=["_labels", "_rates"])))
        update_analysis("calidad_datos", records(data_quality(train)))

        nrows = -(-len(table) // 4)   # filas necesarias para 4 paneles por fila
        fig, axes = plt.subplots(nrows, 4, figsize=(16, 3 * nrows), sharey=True)
        base = float(y.mean())
        for ax, (_, r) in zip(axes.ravel(), table.iterrows()):
            ax.bar(r["_labels"], r["_rates"], color=SERIES[0])
            ax.axhline(base, color=MUTED, ls="--", lw=1)
            ax.set_axisbelow(True)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{r['id']}: {r['veredicto']}", fontsize=9)
        for ax in axes.ravel()[len(table):]:
            ax.axis("off")
        fig.supylabel("tasa de malos (train)")
        fig.suptitle("Tasa de malos por grupo (quintiles o sí/no); línea punteada = tasa global de train")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "pd_eda_hipotesis.png", dpi=150)
        plt.close(fig)

        q = df.groupby(df["issue_d"].dt.to_period("Q"))["target"].agg(["mean", "size"])
        q = q[q["size"] >= 100]   # se omiten trimestres con muy pocos créditos (p. ej. 2007Q2)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(q.index.astype(str), q["mean"], marker="o", color=SERIES[0])
        ax.set_ylabel("tasa de malos")
        ax.set_title("Tasa de malos por trimestre de originación (todas las particiones)")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "pd_eda_tasa_por_trimestre.png", dpi=150)
        plt.close(fig)

    print(summary)
    print(table.drop(columns=["_labels", "_rates", "hipotesis"]).round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    run_eda()