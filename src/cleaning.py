"""Limpieza fija: reglas que se aplican fila por fila, sin estadísticas del conjunto.

Por eso puede ir antes de la partición sin filtrar información entre train, val y
test. Lo que sí depende de los datos (imputar, escalar, recortar) va en el
preprocesador de src/features.py, ajustado solo con entrenamiento.
"""
import numpy as np
import pandas as pd

# Variables de buró sin las cuales la fila no es usable (29 filas de 2007)
CORE_FIELDS = ["annual_inc", "delinq_2yrs", "pub_rec", "inq_last_6mths", "open_acc", "total_acc", "earliest_cr_line"]

# "Meses desde el último evento": vacío significa que el evento nunca ocurrió.
# En 2007-2008 ese "nunca" se codificó como 0 (57 % de ceros en 2007, 0 % desde 2009).
MONTHS_SINCE = ["mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog"]

SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=[c for c in CORE_FIELDS if c in df])


def fix_months_since_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el 0 usado como 'nunca' en 2007-2008 a NaN, igual que el resto de años."""
    df = df.copy()
    for col in MONTHS_SINCE:
        if col in df:
            df.loc[df[col] == 0, col] = np.nan
    return df


def add_never_delinquent(df: pd.DataFrame) -> pd.DataFrame:
    """Indicador de 'nunca moroso' y meses desde la última mora sin faltantes.

    Si nunca hubo mora, el tiempo desde la última es al menos toda la historia
    crediticia, así que el vacío se llena con credit_history_months.
    """
    df = df.copy()
    never = df["mths_since_last_delinq"].isna()
    df["never_delinquent"] = never.astype("int8")
    df["mths_since_last_delinq"] = df["mths_since_last_delinq"].where(~never, df["credit_history_months"])
    return df


def consolidate_home_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """NONE y ANY (unas decenas de filas) se unen con OTHER."""
    df = df.copy()
    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})
    return df


def add_revol_limit(df: pd.DataFrame) -> pd.DataFrame:
    """Límite de crédito rotativo = revol_bal / (revol_util / 100).

    total_rev_hi_lim solo existe desde sep-2012; esta reconstrucción coincide con
    él (error relativo mediano 0,05 %, 96 % de filas con error < 5 %) y está
    disponible para todos los años. Queda NaN si revol_util es 0 o falta.
    """
    df = df.copy()
    util = df["revol_util"].where(df["revol_util"] > 0)
    df["revol_limit"] = df["revol_bal"] / (util / 100)
    return df


def add_sub_grade_num(df: pd.DataFrame) -> pd.DataFrame:
    """sub_grade ordinal: A1 = 1, ..., G5 = 35 (más alto = más riesgoso según LC)."""
    df = df.copy()
    df["sub_grade_num"] = df["sub_grade"].map({s: i for i, s in enumerate(SUB_GRADES, start=1)}).astype("float")
    return df


def apply_fixed_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica todas las reglas fijas. Devuelve (df limpio, resumen de filas)."""
    report = {"filas_entrada": len(df)}
    df = drop_incomplete_rows(df)
    report["filas_sin_buro_eliminadas"] = report["filas_entrada"] - len(df)
    df = fix_months_since_zero(df)
    df = add_never_delinquent(df)
    df = consolidate_home_ownership(df)
    df = add_revol_limit(df)
    df = add_sub_grade_num(df)
    report["filas_salida"] = len(df)
    return df, report
