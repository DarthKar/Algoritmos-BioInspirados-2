"""Selección de variables y preprocesamiento.

Caso de uso (Parte 1): un inversionista decide si fondear un crédito publicado en
la plataforma. Solo pueden entrar variables conocidas en ese momento, incluida la
calificación que asigna Lending Club. Todo lo que describe el comportamiento de
pago posterior es fuga de información (leakage).

OJO: esta clasificación hay que verificarla contra LCDataDictionary.xlsx y
citarla desde ahí en el reporte. En el diccionario, verification_status aparece
como is_inc_v.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Información que solo existe después de la originación (pagos, saldos, recuperaciones)
POST_ORIGINATION = [
    "funded_amnt_inv", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "pymnt_plan",
]

# Identificadores, texto libre, columnas constantes o redundantes
NON_PREDICTIVE = [
    "id", "member_id", "url", "desc", "title", "emp_title", "zip_code", "policy_code",
    "funded_amnt",  # igual a loan_amnt en el 99,3 % de los casos
]

# Variables que existen solo desde cierta fecha: en una partición temporal estarían
# vacías en entrenamiento y llenas en prueba (o su faltante delataría la época).
ERA_DEPENDENT = [
    "tot_cur_bal", "tot_coll_amt",       # desde sep-2012
    "total_rev_hi_lim",                  # desde sep-2012; se reemplaza por revol_limit
    "mths_since_last_major_derog",       # desde ago-2012
    "initial_list_status",               # "w" solo aparece desde 2012
]

# Campos nuevos de dic-2015 y de solicitudes conjuntas (oct-2015): >97 % vacíos
SPARSE_RECENT = [
    "annual_inc_joint", "dti_joint", "verification_status_joint", "application_type",
    "open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "inq_fi", "total_cu_tl", "inq_last_12m",
]

# Reemplazadas por una versión derivada en parte1/preparacion/cleaning.py o redundantes con otra
REPLACED = [
    "earliest_cr_line",          # -> credit_history_months
    "mths_since_last_record",    # 95 % vacío; la información está en pub_rec
    "revol_bal",                 # colineal exacta: revol_limit = revol_bal / (revol_util/100)
]

# Evaluación de riesgo de la propia Lending Club. Para el inversionista es
# información legítima, pero son redundantes entre sí: corr(int_rate, sub_grade)
# = 0,97, installment es función de monto, tasa y plazo, e int_rate además cambia
# con la política de precios de cada año. Por defecto entra solo sub_grade_num.
LC_RISK_ASSESSMENT = ["grade", "sub_grade", "sub_grade_num", "int_rate", "installment"]
DEFAULT_LC_FEATURES = ("sub_grade_num",)

# Geografía: proxy de características protegidas y muchas categorías pequeñas
GEOGRAPHIC = ["addr_state"]

TARGET_SOURCE = ["loan_status", "target", "split"]
DATE_FEATURES = ["issue_d"]  # se usa para particionar, no como predictor

LOG_FEATURES = ["annual_inc", "revol_limit"]

WINSOR_FEATURES = ["revol_util"]   # las de LOG_FEATURES también se topan

STATIC_EXCLUDED = POST_ORIGINATION + NON_PREDICTIVE + ERA_DEPENDENT + SPARSE_RECENT + REPLACED + TARGET_SOURCE + DATE_FEATURES


# log(1 + x) que nunca falla con negativos (los recorta a 0).
def _safe_log1p(x):
    # Función de módulo (no lambda) para que el preprocesador se pueda guardar con joblib
    return np.log1p(np.clip(x, 0, None))

# Topa valores extremos en los percentiles 1 y 99 aprendidos solo en train.
class Winsorizer(BaseEstimator, TransformerMixin):
    """Topa en los percentiles de entrenamiento (se ajusta solo con train)."""

    # Guarda los percentiles inferior y superior a usar.
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    # Aprende los topes (percentiles) de cada columna con los datos de entrenamiento.
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    # Recorta cada columna a los topes aprendidos.
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    # Mantiene los nombres de columna (necesario dentro de ColumnTransformer).
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

# Decide qué columnas entran al modelo (sin fuga, sin casi vacías ni casi constantes).
def select_features(
    df: pd.DataFrame,
    lc_features: tuple[str, ...] = DEFAULT_LC_FEATURES,
    use_geography: bool = False,
    max_missing: float = 0.5,
    max_dominant_share: float = 0.995,
) -> tuple[list[str], list[str]]:
    """Devuelve (numéricas, categóricas) disponibles al momento de fondear.

    Además de las listas fijas, descarta columnas con más de `max_missing` de
    faltantes o cuyo valor más frecuente supera `max_dominant_share` (casi
    constantes, p. ej. term en la muestra B). Ambos filtros se calculan sobre `df`:
    pasar solo el conjunto de entrenamiento.
    """
    unknown = set(lc_features) - set(LC_RISK_ASSESSMENT)
    if unknown:
        raise ValueError(f"lc_features no reconocidas: {sorted(unknown)}")
    excluded = set(STATIC_EXCLUDED) | (set(LC_RISK_ASSESSMENT) - set(lc_features))
    if not use_geography:
        excluded |= set(GEOGRAPHIC)

    candidates = [c for c in df.columns if c not in excluded and not pd.api.types.is_datetime64_any_dtype(df[c])]
    missing = df[candidates].isna().mean()
    dominant = {c: df[c].value_counts(normalize=True, dropna=False).iloc[0] for c in candidates}
    candidates = [c for c in candidates if missing[c] <= max_missing and dominant[c] <= max_dominant_share]

    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numeric]
    return numeric, categorical


# Arma el preprocesamiento: log/topes, imputación, escalado y one-hot.
def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    log_features: list[str] = LOG_FEATURES,
    winsor_features: list[str] = WINSOR_FEATURES,
    min_category_frequency: float = 0.005,
) -> ColumnTransformer:
    """Imputación + log1p y/o topes en variables extremas + estandarización; one-hot en categóricas.

    El mismo preprocesador se usa para la regresión logística y para la red, de
    acuerdo con la regla de comparación justa.
    """
    logged = [c for c in numeric if c in log_features]
    winsor = [c for c in numeric if c in winsor_features and c not in logged]
    plain = [c for c in numeric if c not in logged and c not in winsor]

    # Pasos para un grupo de numéricas: log y/o topes, imputar con indicador, estandarizar.
    def numeric_pipeline(log: bool, clip: bool) -> Pipeline:
        steps = []
        if log:
            steps.append(("log", FunctionTransformer(_safe_log1p, feature_names_out="one-to-one")))
        if clip:
            steps.append(("winsor", Winsorizer()))
        steps += [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
        return Pipeline(steps)

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist", min_frequency=min_category_frequency, sparse_output=False)),
    ])

    transformers = []
    if plain:
        transformers.append(("num", numeric_pipeline(False, False), plain))
    if winsor:
        transformers.append(("num_win", numeric_pipeline(False, True), winsor))
    if logged:
        transformers.append(("num_log", numeric_pipeline(True, True), logged))
    if categorical:
        transformers.append(("cat", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)