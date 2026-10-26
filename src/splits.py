"""Particiones de entrenamiento / validación / prueba."""
import pandas as pd


# Parte el dataset en train/val/test por fecha de originación.
def temporal_split(
    df: pd.DataFrame,
    val_start: str,
    test_start: str,
    date_col: str = "issue_d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición fuera de tiempo según la fecha de originación.

    train: issue_d < val_start
    val:   val_start <= issue_d < test_start
    test:  issue_d >= test_start
    """
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if val_start >= test_start:
        raise ValueError("val_start debe ser anterior a test_start")
    dates = df[date_col]
    train = df[dates < val_start]
    val = df[(dates >= val_start) & (dates < test_start)]
    test = df[dates >= test_start]
    return train, val, test


# Etiqueta cada fila como train, val o test según su fecha.
def assign_split(df: pd.DataFrame, val_start: str, test_start: str, date_col: str = "issue_d") -> pd.Series:
    """Etiqueta cada fila como 'train', 'val' o 'test' con las mismas reglas de temporal_split."""
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if val_start >= test_start:
        raise ValueError("val_start debe ser anterior a test_start")
    dates = df[date_col]
    split = pd.Series("train", index=df.index, dtype="string")
    split[dates >= val_start] = "val"
    split[dates >= test_start] = "test"
    return split


# Tabla con tamaño, tasa de malos y fechas de cada partición.
def split_summary(splits: dict[str, pd.DataFrame], target: str = "target", date_col: str = "issue_d") -> pd.DataFrame:
    """Tabla con tamaño, tasa de incumplimiento y rango de fechas de cada partición."""
    rows = []
    for name, part in splits.items():
        rows.append({
            "particion": name,
            "n": len(part),
            "tasa_malos": part[target].mean() if len(part) else float("nan"),
            "desde": part[date_col].min(),
            "hasta": part[date_col].max(),
        })
    return pd.DataFrame(rows)
