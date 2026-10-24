"""Carga y limpieza básica del archivo original de Lending Club (loan.csv).

Esta limpieza es común a la Parte 1 (PD) y a la Parte 2 (LGD): solo convierte
tipos, no elimina filas ni decide qué variables usar.
"""
from pathlib import Path

import pandas as pd

from src.utils import RAW_LOANS_CSV

DATE_COLUMNS = ["issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"]


def load_raw(path: Path = RAW_LOANS_CSV, nrows: int | None = None) -> pd.DataFrame:
    """Lee loan.csv tal como viene de Kaggle."""
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _parse_month_year(s: pd.Series) -> pd.Series:
    # Las fechas vienen como "Dec-2011"
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def _parse_emp_length(s: pd.Series) -> pd.Series:
    # "< 1 year" -> 0, "1 year" -> 1, ..., "10+ years" -> 10, "n/a" -> NaN
    s = s.astype("string").str.strip()
    out = s.str.extract(r"(\d+)")[0].astype("float")
    out[s.str.startswith("<", na=False)] = 0.0
    return out


def _parse_percent(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.rstrip("%").str.strip(), errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte fechas, plazo, antigüedad laboral y porcentajes a numérico."""
    df = df.copy()
    for col in DATE_COLUMNS:
        if col in df:
            df[col] = _parse_month_year(df[col])
    if "term" in df:
        df["term"] = pd.to_numeric(df["term"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
    if "emp_length" in df:
        df["emp_length"] = _parse_emp_length(df["emp_length"])
    for col in ("int_rate", "revol_util"):
        if col in df:
            df[col] = _parse_percent(df[col])
    if {"issue_d", "earliest_cr_line"} <= set(df.columns):
        # Antigüedad del historial crediticio al momento de la solicitud, en meses
        issue, first = df["issue_d"], df["earliest_cr_line"]
        df["credit_history_months"] = (issue.dt.year - first.dt.year) * 12 + (issue.dt.month - first.dt.month)
    return df
