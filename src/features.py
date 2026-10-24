"""Selección de variables y preprocesamiento.

Caso de uso (Parte 1): el modelo se usa al momento de la SOLICITUD del crédito,
así que solo pueden entrar variables conocidas antes de desembolsar. Todo lo que
describe el comportamiento de pago posterior es fuga de información (leakage).

OJO: esta clasificación hay que verificarla contra LCDataDictionary.xlsx y
citarla desde ahí en el reporte.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Información que solo existe después de la originación (pagos, saldos, recuperaciones)
POST_ORIGINATION = [
    "funded_amnt_inv", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "pymnt_plan",
]

# Identificadores, texto libre y columnas sin variación o de cardinalidad muy alta
NON_PREDICTIVE = [
    "id", "member_id", "url", "desc", "title", "emp_title", "zip_code", "policy_code",
]

# Evaluación de riesgo de la propia Lending Club. Usarlas es casi "copiar" su modelo;
# se dejan como opción para el caso de uso que se defina.
LC_RISK_ASSESSMENT = ["grade", "sub_grade", "int_rate", "installment"]

TARGET_SOURCE = ["loan_status"]
DATE_FEATURES = ["issue_d", "earliest_cr_line"]  # issue_d se usa para particionar, no como predictor

LOG_FEATURES = ["annual_inc", "revol_bal", "tot_cur_bal", "total_rev_hi_lim", "tot_coll_amt"]


def _safe_log1p(x):
    # Función de módulo (no lambda) para que el preprocesador se pueda guardar con joblib
    return np.log1p(np.clip(x, 0, None))


def select_features(
    df: pd.DataFrame,
    use_lc_grade: bool = False,
    max_missing: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Devuelve (numéricas, categóricas) disponibles al momento de la solicitud.

    Se descartan columnas con más de `max_missing` de faltantes, calculado sobre
    `df` (pasar solo el conjunto de entrenamiento para no mirar validación/prueba).
    """
    excluded = set(POST_ORIGINATION + NON_PREDICTIVE + TARGET_SOURCE + DATE_FEATURES + ["target"])
    if not use_lc_grade:
        excluded |= set(LC_RISK_ASSESSMENT)

    candidates = [c for c in df.columns if c not in excluded]
    missing = df[candidates].isna().mean()
    candidates = [c for c in candidates if missing[c] <= max_missing]

    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numeric and not pd.api.types.is_datetime64_any_dtype(df[c])]
    return numeric, categorical


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    log_features: list[str] = LOG_FEATURES,
    min_category_frequency: float = 0.005,
) -> ColumnTransformer:
    """Imputación + log1p en variables sesgadas + estandarización; one-hot en categóricas.

    El mismo preprocesador se usa para la regresión logística y para la red, de
    acuerdo con la regla de comparación justa.
    """
    logged = [c for c in numeric if c in log_features]
    plain = [c for c in numeric if c not in log_features]

    def numeric_pipeline(log: bool) -> Pipeline:
        steps = [("log", FunctionTransformer(_safe_log1p, feature_names_out="one-to-one"))] if log else []
        steps += [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
        return Pipeline(steps)

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_category_frequency, sparse_output=False)),
    ])

    transformers = []
    if plain:
        transformers.append(("num", numeric_pipeline(log=False), plain))
    if logged:
        transformers.append(("num_log", numeric_pipeline(log=True), logged))
    if categorical:
        transformers.append(("cat", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
