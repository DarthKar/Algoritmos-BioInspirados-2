"""Selección de la muestra de modelamiento.

Al excluir los créditos sin desenlace (Current, etc.), en las cohortes recientes
solo quedan los que terminaron rápido: impagos tempranos y prepagos. Eso infla la
tasa de malos (sesgo de selección sobre el resultado). Dos modos:

- "matured" (B, por defecto): solo créditos cuyo plazo completo terminó antes de
  la fecha de corte del dataset. Su desenlace es conocido en ~100 % de los casos.
- "resolved" (A): todos los créditos con desenlace, sin importar la cohorte. Se
  deja para cuantificar el sesgo como comparación.
"""
import pandas as pd

# Último mes con pagos registrados en loan.csv (max(last_pymnt_d) = ene-2016)
SNAPSHOT = "2016-01-01"

DEFAULT_TERMS = {
    "matured": (36,),       # de 60 meses solo maduran 3.372 créditos (cohorte 2010)
    "resolved": (36, 60),
}


# True si el crédito ya cumplió todo su plazo antes de la fecha de corte.
def is_matured(df: pd.DataFrame, snapshot: str = SNAPSHOT) -> pd.Series:
    """True si issue_d + term (meses) es anterior a la fecha de corte."""
    snap = pd.Timestamp(snapshot)
    end_month = df["issue_d"].dt.year * 12 + df["issue_d"].dt.month + df["term"]
    return end_month < snap.year * 12 + snap.month


# Filtra la muestra B (maduros de 36 meses) o A (todos los resueltos).
def select_sample(
    df: pd.DataFrame,
    mode: str = "matured",
    terms: tuple[int, ...] | None = None,
    snapshot: str = SNAPSHOT,
) -> pd.DataFrame:
    """Filtra la muestra según el modo. `df` ya debe tener `target` (sin NA)."""
    if mode not in DEFAULT_TERMS:
        raise ValueError(f"mode debe ser uno de {list(DEFAULT_TERMS)}")
    terms = DEFAULT_TERMS[mode] if terms is None else terms
    mask = df["term"].isin(terms)
    if mode == "matured":
        mask &= is_matured(df, snapshot)
    return df[mask].copy()
