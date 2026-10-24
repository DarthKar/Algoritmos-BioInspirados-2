"""Variable objetivo de la Parte 1 a partir de loan_status (tabla del enunciado)."""
import pandas as pd

# 1 = malo (incumple), 0 = bueno. Los estados sin desenlace definitivo quedan en NA.
LOAN_STATUS_TO_TARGET = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Current": pd.NA,
    "Issued": pd.NA,
    "In Grace Period": pd.NA,
    "Late (16-30 days)": pd.NA,
}


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna `target` y elimina los casos NA.

    Falla si aparece un loan_status que no está en la tabla, para no codificarlo
    en silencio.
    """
    status = df["loan_status"].astype("string").str.strip()
    unknown = set(status.dropna().unique()) - set(LOAN_STATUS_TO_TARGET)
    if unknown:
        raise ValueError(f"loan_status sin codificación definida: {sorted(unknown)}")

    out = df.copy()
    out["target"] = status.map(LOAN_STATUS_TO_TARGET).astype("Int8")
    out = out[out["target"].notna()].copy()
    out["target"] = out["target"].astype("int8")
    return out


def target_report(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo por loan_status y su codificación (sirve como tabla para el reporte)."""
    status = df["loan_status"].astype("string").str.strip()
    counts = status.value_counts(dropna=False).rename("n").to_frame()
    counts["target"] = [LOAN_STATUS_TO_TARGET.get(s, "sin definir") for s in counts.index]
    counts["pct"] = counts["n"] / counts["n"].sum()
    return counts
