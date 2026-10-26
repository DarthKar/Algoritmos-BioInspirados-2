"""Experimento del perceptrón multicapa de la Parte 1.

1. Búsqueda aleatoria de arquitectura e hiperparámetros (Bergstra y Bengio, 2012):
   con el mismo presupuesto explora mejor que una rejilla cuando pocos
   hiperparámetros importan. Se elige por log-loss en validación, igual que la
   regresión logística.
2. La mejor configuración se reentrena con `n_seeds` semillas y se reportan media,
   desviación estándar, mejor y peor valor de cada métrica en prueba.

Usa exactamente las mismas variables, preprocesamiento y particiones que la
regresión logística (make_design_matrices).
"""
import json
import time
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

from src.features import DEFAULT_LC_FEATURES
from src.pd_model.dataset import load_pd_dataset, make_design_matrices
from src.pd_model.metrics import calibration_in_the_large, calibration_table, evaluate, summarize_seeds
from src.pd_model.models import MLP, MLPConfig, n_trainable_params, predict_proba, train_mlp
from src.utils import MODELS_DIR, RESULTS_DIR, SEED, set_seed

SEARCH_SPACE = {
    "n_layers": [1, 2, 3],
    "width": [16, 32, 64, 128, 256],
    "activation": ["relu", "leaky_relu", "elu", "tanh", "gelu"],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    "l2_log10": (-7.0, -3.0),            # λ ~ log-uniforme; con prob. 0.2 se usa λ = 0
    "optimizer": ["adam", "rmsprop", "sgd"],
    "lr_log10": {"adam": (-4.0, -2.3), "rmsprop": (-4.0, -2.3), "sgd": (-3.0, -1.0)},
    "batch_size": [256, 512, 1024],
}


# Parámetros del experimento del MLP (presupuesto de búsqueda, semillas, épocas).
@dataclass
class MLPExperimentConfig:
    mode: str = "matured"
    lc_features: tuple[str, ...] = DEFAULT_LC_FEATURES
    use_geography: bool = False
    n_trials: int = 25
    n_seeds: int = 30
    max_epochs: int = 100
    patience: int = 8
    seed: int = SEED
    name: str = "pd_mlp"


# Sortea una configuración aleatoria del espacio de búsqueda.
def sample_config(rng: np.random.Generator, max_epochs: int, patience: int, seed: int) -> MLPConfig:
    """Una configuración aleatoria del espacio de búsqueda (capas en embudo)."""
    s = SEARCH_SPACE
    n_layers = int(rng.choice(s["n_layers"]))
    first = int(rng.choice(s["width"]))
    hidden = [max(8, first // (2 ** i)) for i in range(n_layers)]
    optimizer = str(rng.choice(s["optimizer"]))
    lr = float(10 ** rng.uniform(*s["lr_log10"][optimizer]))
    l2 = 0.0 if rng.random() < 0.2 else float(10 ** rng.uniform(*s["l2_log10"]))
    return MLPConfig(hidden=hidden, activation=str(rng.choice(s["activation"])),
                     dropout=float(rng.choice(s["dropout"])), l2=l2, optimizer=optimizer, lr=lr,
                     batch_size=int(rng.choice(s["batch_size"])), max_epochs=max_epochs,
                     patience=patience, seed=seed)


# Métricas de discriminación y calibración más log-loss.
def _metrics(y, p) -> dict:
    return {**evaluate(y, p), "log_loss": float(log_loss(y, p))}


# Entrena una configuración con varias semillas y guarda métricas e historial de pérdidas.
def train_seeds(mcfg: MLPConfig, X: dict, y: dict, n_seeds: int, verbose: bool = True):
    """Entrena `mcfg` con las semillas 0..n_seeds-1.

    Devuelve (métricas en prueba por semilla, pérdidas por época de cada semilla,
    corrida con menor log-loss en validación).
    """
    runs, histories, best_run = [], [], None
    for seed in range(n_seeds):
        mcfg.seed = seed
        model, hist = train_mlp(X["train"], y["train"], X["val"], y["val"], mcfg, X_test=X["test"], y_test=y["test"])
        p = {s: predict_proba(model, X[s]) for s in X}
        val_loss = float(log_loss(y["val"], p["val"]))
        runs.append({"semilla": seed, "mejor_epoca": hist["best_epoch"], "segundos": hist["seconds"],
                     "val_log_loss": val_loss, **_metrics(y["test"], p["test"])})
        histories.append(pd.DataFrame({"semilla": seed, "epoca": range(1, len(hist["train_loss"]) + 1),
                                       "train_loss": hist["train_loss"], "val_loss": hist["val_loss"],
                                       "test_loss": hist["test_loss"]}))
        if best_run is None or val_loss < best_run["val_loss"]:
            best_run = {"val_loss": val_loss, "seed": seed, "model": model, "probs": p}
        if verbose:
            print(f"semilla {seed:2d}: test auc={runs[-1]['auc']:.4f} brier={runs[-1]['brier']:.4f}", flush=True)
    return pd.DataFrame(runs), pd.concat(histories, ignore_index=True), best_run


# Rehace las curvas de pérdida de la configuración guardada sin repetir la búsqueda.
def regenerate_histories(name: str = "pd_mlp") -> pd.DataFrame:
    """Reentrena la configuración guardada con sus semillas para obtener las pérdidas
    por época, sin repetir la búsqueda. El entrenamiento es determinista en CPU."""
    results = json.loads((RESULTS_DIR / f"{name}_metricas.json").read_text(encoding="utf-8"))
    c = results["config"]
    data = make_design_matrices(load_pd_dataset(c["mode"]), tuple(c["lc_features"]), c["use_geography"])
    mcfg = MLPConfig(**results["mejor_config"])
    _, histories, _ = train_seeds(mcfg, data["X"], data["y"], c["n_seeds"])
    histories.to_csv(RESULTS_DIR / f"{name}_historial.csv", index=False)
    return histories


# Búsqueda aleatoria + 30 semillas de la mejor configuración; guarda modelo y resultados.
def run_mlp_experiment(cfg: MLPExperimentConfig = MLPExperimentConfig(), save: bool = True, verbose: bool = True) -> dict:
    set_seed(cfg.seed)
    start = time.perf_counter()
    data = make_design_matrices(load_pd_dataset(cfg.mode), cfg.lc_features, cfg.use_geography)
    X, y = data["X"], data["y"]
    rng = np.random.default_rng(cfg.seed)

    # 1. Búsqueda aleatoria
    trials = []
    for i in range(cfg.n_trials):
        mcfg = sample_config(rng, cfg.max_epochs, cfg.patience, cfg.seed)
        model, hist = train_mlp(X["train"], y["train"], X["val"], y["val"], mcfg)
        p_val = predict_proba(model, X["val"])
        trials.append({"trial": i, **asdict(mcfg), "n_parametros": n_trainable_params(model),
                       "mejor_epoca": hist["best_epoch"], "segundos": hist["seconds"], **{f"val_{k}": v for k, v in _metrics(y["val"], p_val).items()}})
        if verbose:
            t = trials[-1]
            print(f"[{i + 1}/{cfg.n_trials}] {t['hidden']} {t['activation']} {t['optimizer']} lr={t['lr']:.1e} "
                  f"drop={t['dropout']} l2={t['l2']:.1e} -> val_logloss={t['val_log_loss']:.4f} auc={t['val_auc']:.4f} ({t['segundos']:.0f}s)", flush=True)
    trials_df = pd.DataFrame(trials).sort_values("val_log_loss")
    search_seconds = time.perf_counter() - start
    best = json.loads(trials_df.iloc[[0]].to_json(orient="records"))[0]  # tipos nativos de Python
    best_cfg = MLPConfig(**{k: best[k] for k in asdict(MLPConfig()).keys()})

    # 2. Configuración final con varias semillas
    runs_df, histories_df, best_run = train_seeds(best_cfg, X, y, cfg.n_seeds, verbose)
    seed_summary = summarize_seeds(runs_df[["auc", "gini", "ks", "brier", "log_loss"]].to_dict("records"))

    probs = best_run["probs"]
    best_cfg.seed = best_run["seed"]
    results = {
        "config": asdict(cfg),
        "variables_numericas": data["numeric"],
        "variables_categoricas": data["categorical"],
        "n_columnas_X": len(data["feature_names"]),
        "mejor_config": asdict(best_cfg),
        "n_parametros": n_trainable_params(best_run["model"]),
        "busqueda": {"metodo": "búsqueda aleatoria", "n_configuraciones": cfg.n_trials,
                     "criterio": "log-loss en validación", "segundos": search_seconds},
        "semillas": {"n": cfg.n_seeds, "resumen_test": seed_summary.reset_index(names="metrica").to_dict("records"),
                     "semilla_guardada": best_run["seed"], "segundos": float(runs_df["segundos"].sum())},
        "segundos_totales": time.perf_counter() - start,
        "metricas_modelo_guardado": {s: {**_metrics(y[s], probs[s]), **calibration_in_the_large(y[s], probs[s]), "n": int(len(y[s]))} for s in X},
        "calibracion_test": calibration_table(y["test"], probs["test"]).to_dict("records"),
    }
    if save:
        _save(cfg, data, best_run, trials_df, runs_df, histories_df, results)
    return results


# Guarda red, preprocesador, predicciones, búsqueda, semillas, historial y métricas.
def _save(cfg, data, best_run, trials_df, runs_df, histories_df, results) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = best_run["model"]
    torch.save({"state_dict": model.state_dict(), "config": results["mejor_config"],
                "n_inputs": len(data["feature_names"]), "feature_names": data["feature_names"]},
               MODELS_DIR / f"{cfg.name}.pt")
    joblib.dump(data["preprocessor"], MODELS_DIR / f"{cfg.name}_preprocessor.joblib")

    preds = pd.concat([pd.DataFrame({"id": data["ids"][s], "split": s, "target": data["y"][s], "pd": p})
                       for s, p in best_run["probs"].items()], ignore_index=True)
    preds.to_parquet(RESULTS_DIR / f"{cfg.name}_predicciones.parquet", index=False)
    trials_df.to_csv(RESULTS_DIR / f"{cfg.name}_busqueda.csv", index=False)
    runs_df.to_csv(RESULTS_DIR / f"{cfg.name}_semillas.csv", index=False)
    histories_df.to_csv(RESULTS_DIR / f"{cfg.name}_historial.csv", index=False)
    (RESULTS_DIR / f"{cfg.name}_metricas.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# Carga el preprocesador y la red guardados, listos para predecir.
def load_mlp(name: str = "pd_mlp"):
    """Carga (preprocesador, red) guardados. PD = predict_proba(red, preprocesador.transform(df))."""
    ckpt = torch.load(MODELS_DIR / f"{name}.pt", weights_only=False)
    c = ckpt["config"]
    model = MLP(ckpt["n_inputs"], c["hidden"], c["activation"], c["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return joblib.load(MODELS_DIR / f"{name}_preprocessor.joblib"), model
