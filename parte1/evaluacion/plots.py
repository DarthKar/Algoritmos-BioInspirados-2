"""Figuras de discriminación y calibración de un modelo de PD.

Se generan desde el archivo de predicciones guardado, así que no requieren
reentrenar.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from parte1.evaluacion.metrics import calibration_table, ks_statistic
from parte1.comun.plotting import MUTED, SERIES, TEXT_SECONDARY, apply_style
from parte1.comun.utils import FIGURES_DIR, RESULTS_DIR

SPLIT_LABELS = {"val": "Validación (2012 T3)", "test": "Prueba (2012 T4)"}


# Lee las predicciones guardadas de un modelo.
def load_predictions(name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{name}_predicciones.parquet")


# Curva ROC de validación y prueba con AUC y KS.
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


# Curva de calibración por deciles e histograma de PD en prueba.
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


LOSS_LABELS = {"train_loss": "Entrenamiento", "val_loss": "Validación (2012 T3)", "test_loss": "Prueba (2012 T4)"}


# Pérdida por época en train, val y test para todas las semillas.
def plot_loss_curves(histories: pd.DataFrame, highlight_seed: int, title: str) -> plt.Figure:
    """Entropía cruzada por época en train, val y test.

    Líneas tenues: cada una de las semillas; línea gruesa: la semilla guardada.
    La pérdida de entrenamiento se mide en modo evaluación (sin dropout) y sin la
    penalización L2, para que sea comparable con val y test.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for color, (col, label) in zip(SERIES, LOSS_LABELS.items()):
        for seed, h in histories.groupby("semilla"):
            if seed != highlight_seed:
                ax.plot(h["epoca"], h[col], color=color, alpha=0.15, linewidth=1)
        h = histories[histories["semilla"] == highlight_seed]
        ax.plot(h["epoca"], h[col], color=color, label=label)

    h = histories[histories["semilla"] == highlight_seed]
    best_epoch = int(h.loc[h["val_loss"].idxmin(), "epoca"])
    ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle="--")
    values = histories[list(LOSS_LABELS)].to_numpy().ravel()
    # El eje llega al máximo de la semilla resaltada para no cortar sus primeras épocas
    low = values.min()
    high = max(pd.Series(values).quantile(0.98), h[list(LOSS_LABELS)].to_numpy().max())
    pad = (high - low) * 0.08
    ax.set_ylim(low - pad, high + pad)
    ax.text(best_epoch, 0.42, f"pesos guardados (época {best_epoch})  ", color=TEXT_SECONDARY, fontsize=8.5,
            ha="right", va="center", transform=ax.get_xaxis_transform())
    n_seeds = histories["semilla"].nunique()
    ax.set(xlabel="Época", ylabel="Entropía cruzada binaria (log-loss)",
           title=f"{title}\nLínea gruesa: semilla {highlight_seed}; líneas tenues: las otras {n_seeds - 1} semillas")
    ax.title.set_fontsize(11)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig


# Genera y guarda la figura de pérdidas desde el historial.
def save_loss_figure(name: str, model_label: str) -> Path:
    """Guarda la figura de pérdidas a partir de results/{name}_historial.csv."""
    import json

    histories = pd.read_csv(RESULTS_DIR / f"{name}_historial.csv")
    seed = json.loads((RESULTS_DIR / f"{name}_metricas.json").read_text(encoding="utf-8"))["semillas"]["semilla_guardada"]
    fig = plot_loss_curves(histories, seed, f"Pérdida por época: {model_label}")
    path = FIGURES_DIR / f"{name}_perdidas.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# Genera y guarda las figuras ROC y de calibración.
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
