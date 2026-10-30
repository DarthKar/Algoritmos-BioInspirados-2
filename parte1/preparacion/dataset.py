"""Construcción del dataset procesado de la Parte 1 desde loan.csv.

Orden: tipos -> variable objetivo -> muestra (A/B) -> limpieza fija -> partición.
El preprocesamiento que usa estadísticas (imputar, escalar, one-hot) NO se hace
aquí: se ajusta con entrenamiento al momento de entrenar (parte1/preparacion/features.py).
"""
import json
from pathlib import Path

import pandas as pd

from parte1.preparacion.cleaning import apply_fixed_cleaning
from parte1.preparacion.load import clean, load_raw
from parte1.preparacion.features import DEFAULT_LC_FEATURES, NON_PREDICTIVE, POST_ORIGINATION, build_preprocessor, select_features
from parte1.preparacion.sample import SNAPSHOT, select_sample
from parte1.preparacion.target import build_target
from parte1.preparacion.splits import assign_split, split_summary
from parte1.comun.utils import DATA_PROCESSED, RAW_LOANS_CSV

# Partición temporal por fecha de originación, según el modo de muestra.
# B (matured): train 2007-06..2012-06, val 2012 T3, test 2012 T4.
DEFAULT_SPLITS = {
    "matured": {"val_start": "2012-07-01", "test_start": "2012-10-01"},
    "resolved": {"val_start": "2014-01-01", "test_start": "2015-01-01"},
}

ID_COLUMNS = ["id", "issue_d", "loan_status", "target", "split"]


# Construye el dataset de la Parte 1 desde loan.csv: target, muestra, limpieza y partición.
def build_pd_dataset(
    mode: str = "matured",
    terms: tuple[int, ...] | None = None,
    snapshot: str = SNAPSHOT,
    val_start: str | None = None,
    test_start: str | None = None,
    raw_path: Path = RAW_LOANS_CSV,
) -> tuple[pd.DataFrame, dict]:
    """Devuelve (dataset, reporte con conteos de cada paso)."""
    val_start = val_start or DEFAULT_SPLITS[mode]["val_start"]
    test_start = test_start or DEFAULT_SPLITS[mode]["test_start"]

    raw = load_raw(raw_path)
    report = {"modo": mode, "snapshot": snapshot, "val_start": val_start, "test_start": test_start,
              "filas_originales": len(raw), "columnas_originales": int(raw.shape[1])}
    df = clean(raw)
    del raw

    df = build_target(df)
    report["filas_con_desenlace"] = len(df)

    df = select_sample(df, mode=mode, terms=terms, snapshot=snapshot)
    report["filas_muestra"] = len(df)
    report["plazos"] = sorted(int(t) for t in df["term"].unique())

    df, cleaning_report = apply_fixed_cleaning(df)
    report["limpieza"] = cleaning_report

    df["split"] = assign_split(df, val_start, test_start)

    # Se guardan solo columnas disponibles al fondear (sin fuga ni texto libre)
    drop = [c for c in POST_ORIGINATION + NON_PREDICTIVE if c in df and c != "id"]
    df = df.drop(columns=drop)
    df = df[ID_COLUMNS + [c for c in df.columns if c not in ID_COLUMNS]].reset_index(drop=True)

    summary = split_summary({s: df[df["split"] == s] for s in ("train", "val", "test")})
    report["particiones"] = json.loads(summary.to_json(orient="records", date_format="iso"))
    report["columnas_guardadas"] = int(df.shape[1])
    return df, report


# Ruta del parquet procesado para un modo de muestra.
def dataset_path(mode: str) -> Path:
    return DATA_PROCESSED / f"pd_{mode}.parquet"


# Guarda el dataset procesado (parquet) y su reporte de conteos (json).
def save_pd_dataset(df: pd.DataFrame, report: dict, mode: str) -> Path:
    path = dataset_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    path.with_name(f"pd_{mode}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# Lee el dataset procesado ya guardado.
def load_pd_dataset(mode: str = "matured") -> pd.DataFrame:
    return pd.read_parquet(dataset_path(mode))


# Ajusta el preprocesador con train y devuelve X, y, ids por partición (igual para LR y MLP).
def make_design_matrices(
    df: pd.DataFrame,
    lc_features: tuple[str, ...] = DEFAULT_LC_FEATURES,
    use_geography: bool = False,
) -> dict:
    """Selecciona variables y ajusta el preprocesador SOLO con entrenamiento.

    Lo usan la regresión logística y la red, para que ambas vean exactamente las
    mismas variables, el mismo preprocesamiento y las mismas particiones.
    """
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    numeric, categorical = select_features(parts["train"], lc_features=lc_features, use_geography=use_geography)
    preprocessor = build_preprocessor(numeric, categorical).fit(parts["train"])
    return {
        "preprocessor": preprocessor,
        "numeric": numeric,
        "categorical": categorical,
        "feature_names": list(preprocessor.get_feature_names_out()),
        "X": {s: preprocessor.transform(p) for s, p in parts.items()},
        "y": {s: p["target"].to_numpy() for s, p in parts.items()},
        "ids": {s: p["id"].to_numpy() for s, p in parts.items()},
    }
