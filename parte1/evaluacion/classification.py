"""Métricas de clasificación (accuracy, F1, matriz de confusión) para LR y MLP.

El modelo entrega una probabilidad; para hablar de accuracy o F1 hay que fijar un
umbral y decir "malo" si PD >= umbral. Se reportan dos umbrales:
- 0,5: el valor por defecto. Con 13 % de malos casi nadie lo supera, así que el
  accuracy se parece al de predecir "todos buenos" (paradoja del accuracy).
- el que maximiza F1 en validación, aplicado sin cambios a prueba.
El umbral nunca se elige con prueba. Se calcula desde las predicciones guardadas,
sin reentrenar.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score

from parte1.evaluacion.plots import load_predictions
from parte1.comun.plotting import MUTED, SERIES, TEXT_PRIMARY, TEXT_SECONDARY, apply_style
from parte1.comun.utils import FIGURES_DIR, records, update_analysis

MODELS = {"pd_logistic": "Regresión logística", "pd_mlp": "MLP (semilla guardada)"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]   # rampa secuencial azul


# Umbral que maximiza F1 en un conjunto (se usa con validación).
def best_f1_threshold(y_true, y_prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])   # el último punto de la curva no tiene umbral


# Accuracy, precisión, sensibilidad, especificidad, F1 y celdas de la matriz para un umbral.
def threshold_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "umbral": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensibilidad": float(recall_score(y_true, y_pred, zero_division=0)),   # malos detectados
        "especificidad": float(tn / (tn + fp)),                                  # buenos aceptados
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tasa_rechazo": float(y_pred.mean()),                                    # créditos marcados como malos
        "vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp),
    }


# Tabla de métricas por modelo, criterio de umbral y partición, más la referencia "todos buenos".
def classification_table(names=tuple(MODELS)) -> pd.DataFrame:
    rows = []
    for name in names:
        preds = load_predictions(name)
        val, test = preds[preds["split"] == "val"], preds[preds["split"] == "test"]
        thresholds = {"0.5": 0.5, "max F1 en validación": best_f1_threshold(val["target"], val["pd"])}
        for criterion, t in thresholds.items():
            for split, part in (("val", val), ("test", test)):
                rows.append({"modelo": name, "criterio_umbral": criterion, "split": split,
                             **threshold_metrics(part["target"], part["pd"], t)})
    # Referencia: aprobar a todos (umbral por encima de cualquier PD)
    test = load_predictions(names[0]).query("split == 'test'")
    rows.append({"modelo": "referencia_todos_buenos", "criterio_umbral": "ninguno", "split": "test",
                 **threshold_metrics(test["target"], np.zeros(len(test)), 1.0)})
    return pd.DataFrame(rows)


# Matrices de confusión en prueba (conteos y % por fila) para el umbral de máximo F1.
def plot_confusion(table: pd.DataFrame) -> plt.Figure:
    apply_style()
    sel = table[(table["criterio_umbral"] == "max F1 en validación") & (table["split"] == "test")]
    fig, axes = plt.subplots(1, len(sel), figsize=(5.2 * len(sel), 4.6))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("azules", BLUES)
    for ax, (_, r) in zip(np.atleast_1d(axes), sel.iterrows()):
        m = np.array([[r["vn"], r["fp"]], [r["fn"], r["vp"]]])
        share = m / m.sum(axis=1, keepdims=True)                  # % de cada clase real
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if share[i, j] > 0.55 else TEXT_PRIMARY
                ax.text(j, i, f"{m[i, j]:,}\n{share[i, j]:.0%} de la fila", ha="center", va="center",
                        color=color, fontsize=10)
        ax.set_xticks([0, 1], ["Predice bueno", "Predice malo"])
        ax.set_yticks([0, 1], ["Real bueno", "Real malo"])
        ax.grid(False)
        ax.set_title(f"{MODELS[r['modelo']]}\numbral {r['umbral']:.3f}: accuracy {r['accuracy']:.1%}, F1 {r['f1']:.3f}",
                     fontsize=10.5)
    fig.suptitle("Matriz de confusión en prueba (2012 T4), umbral elegido en validación", fontsize=12)
    fig.tight_layout()
    return fig


# Histograma de la PD en prueba según la clase real, con el umbral de máximo F1.
def plot_pd_by_class(table: pd.DataFrame) -> plt.Figure:
    """Cuánto se solapan las PD de buenos y malos: lo que el AUC resume en un número."""
    apply_style()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 4), sharey=True)
    for ax, (name, label) in zip(np.atleast_1d(axes), MODELS.items()):
        test = load_predictions(name).query("split == 'test'")
        bins = np.linspace(0, test["pd"].quantile(0.995), 40)
        for cls, color, text in ((0, SERIES[0], "Buenos"), (1, SERIES[1], "Malos")):
            ax.hist(test.loc[test["target"] == cls, "pd"], bins=bins, density=True, alpha=0.55,
                    color=color, edgecolor="white", linewidth=0.4, label=text)
        cut = table.query("modelo == @name and criterio_umbral == 'max F1 en validación' and split == 'test'")["umbral"].iloc[0]
        ax.axvline(cut, color=MUTED, linestyle="--", linewidth=1)
        ax.text(cut, 0.95, f"  umbral {cut:.3f}", transform=ax.get_xaxis_transform(), fontsize=8.5, color=TEXT_SECONDARY)
        ax.set(title=label, xlabel="PD estimada", ylabel="Densidad")
        ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Distribución de la PD en prueba según la clase real", fontsize=12)
    fig.tight_layout()
    return fig


# Calcula la tabla, la guarda en pd_analisis.json y genera las figuras de umbral.
def run_classification(save: bool = True) -> pd.DataFrame:
    table = classification_table()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        update_analysis("clasificacion", records(table))
        for fname, fig in (("pd_matriz_confusion.png", plot_confusion(table)),
                           ("pd_distribucion_por_clase.png", plot_pd_by_class(table))):
            fig.savefig(FIGURES_DIR / fname)
            plt.close(fig)
    cols = ["modelo", "criterio_umbral", "split", "umbral", "accuracy", "precision", "sensibilidad",
            "especificidad", "f1", "tasa_rechazo"]
    print(table[cols].round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    run_classification()
