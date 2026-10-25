"""Figuras de discriminación y calibración de un modelo de PD.

Se generan desde el archivo de predicciones guardado, así que no requieren
reentrenar.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.pd_model.metrics import calibration_table, ks_statistic
from src.plotting import MUTED, SERIES, TEXT_SECONDARY, apply_style
from src.utils import FIGURES_DIR, RESULTS_DIR

SPLIT_LABELS = {"val": "Validación (2012 T3)", "test": "Prueba (2012 T4)"}


def load_predictions(name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{name}_predicciones.parquet")


def plot_roc(preds: pd.DataFrame, title: str, splits=("val", "test")) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Azar")
    for color, split in zip(SERIES, splits):
        part = preds[preds["split"] == split]
        fpr, tpr, _ = roc_curve(part["target"], part["pd"])
        auc = roc_auc_score(part["target"], part["pd"])
        ks = ks_statistic(part["target"], part["pd"])
        ax.plot(fpr, tpr, color=color, label=f"{SPLIT_LABELS.get(split, split)}: AUC {auc:.3f}, KS {ks:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Tasa de falsos positivos (buenos rechazados)",
           ylabel="Tasa de verdaderos positivos (malos detectados)", title=title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8.5)
    return fig


def plot_calibration(preds: pd.DataFrame, title: str, splits=("val", "test"), n_bins: int = 10) -> plt.Figure:
    """Curva de calibración (deciles de PD predicha) e histograma de PD en prueba."""
    apply_style()
    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(5.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    parts = {s: preds[preds["split"] == s] for s in splits}
    upper = max(p["pd"].quantile(0.995) for p in parts.values())
    upper = min(1.0, round(upper + 0.05, 1))

    ax.plot([0, upper], [0, upper], color=MUTED, linewidth=1, linestyle="--", label="Calibración perfecta")
    for color, (split, part) in zip(SERIES, parts.items()):
        table = calibration_table(part["target"], part["pd"], n_bins=n_bins)
        ax.plot(table["prob_predicha"], table["tasa_observada"], color=color, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1, label=SPLIT_LABELS.get(split, split))
    ax.set(xlim=(0, upper), ylim=(0, upper), ylabel="Tasa observada de malos", title=title)
    ax.legend(loc="upper left", fontsize=8.5)

    test = parts.get("test", next(iter(parts.values())))
    ax_hist.hist(test["pd"], bins=40, range=(0, upper), color=SERIES[0], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_hist.set(xlabel="PD predicha (cada punto arriba = un decil)", ylabel="Créditos")
    ax_hist.text(0.99, 0.9, "Distribución en prueba", transform=ax_hist.transAxes, ha="right", va="top",
                 fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    return fig


def save_figures(name: str, model_label: str) -> list[Path]:
    """Genera y guarda las figuras de ROC y calibración de `name`."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_predictions(name)
    paths = []
    for kind, fig in (("roc", plot_roc(preds, f"Curva ROC: {model_label}")),
                      ("calibracion", plot_calibration(preds, f"Calibración: {model_label}"))):
        path = FIGURES_DIR / f"{name}_{kind}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths
