import json
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parte2 import config
from src.pd_model.dataset import build_pd_dataset
from src.pd_model.target import LOAN_STATUS_TO_TARGET
from src.utils import RAW_LOANS_CSV

# ======================================================================
# Constantes
# ======================================================================
CFG = config.CFG

NUM = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
       "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]
CAT = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
NUM_ALL = NUM + ["term_m", "emp_len"]
FEATURES = NUM_ALL + CAT

LGD_STATUS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]
PD_BAD_STATUSES = [
    status for status, target in LOAN_STATUS_TO_TARGET.items()
    if pd.notna(target) and target == 1
]
LGD_SOURCE_COLUMNS = ("funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee")

# (unidades por capa, dropout, learning rate)
GRID = [((32,), 0.0, 1e-3),
        ((64,), 0.0, 1e-3),
        ((64, 32), 0.1, 1e-3),
        ((128, 64), 0.2, 1e-3),
        ((128, 64, 32), 0.2, 5e-4),
        ((256, 128), 0.3, 5e-4)]

COLS_SAVE = list(dict.fromkeys(
    ["id", "issue_d", "loan_status", "target", "split"] + FEATURES +
    list(LGD_SOURCE_COLUMNS) + ["issue_dt", "default", "EAD_obs", "rec_net",
                                "lgd_ok", "LGD_raw", "LGD"]))
CACHE_REQUIRED_COLUMNS = set(COLS_SAVE)


def titulo(texto):
    print(f"\n========={texto}=========\n")


# ======================================================================
# Datos
# ======================================================================
def ruta_cache():
    return Path(CFG["PROC_DIR"]) / f"lending_{CFG['PD_MODE']}_lgd.parquet"


def ruta_metadata_cache():
    return ruta_cache().with_suffix(".json")


def validar_cache(df, metadata):
    """Rechaza cachés incompletos o de otra versión/modo antes de modelar."""
    if metadata.get("version") != CFG["CACHE_VERSION"]:
        raise ValueError(
            f"Versión de caché no compatible en {ruta_metadata_cache()}; "
            "reconstrúyelo con --rebuild-cache y el CSV original."
        )
    if metadata.get("mode") != CFG["PD_MODE"]:
        raise ValueError(f"El caché es del modo {metadata.get('mode')}, no {CFG['PD_MODE']}.")
    missing = CACHE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Al caché {ruta_cache()} le faltan columnas {sorted(missing)}; "
            "reconstrúyelo con --rebuild-cache y el CSV original."
        )
    if metadata.get("rows") != len(df):
        raise ValueError(f"El conteo de filas de {ruta_cache()} no coincide con su metadata.")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError(f"Los identificadores del caché {ruta_cache()} están vacíos o duplicados.")
    if df["split"].isna().any() or not set(df["split"].unique()) <= {"train", "val", "test"}:
        raise ValueError(f"El caché {ruta_cache()} tiene particiones inválidas.")


def cargar_datos(rebuild_cache=False):
    """Carga el caché local o lo crea una vez usando el constructor de Parte 1."""
    cache = ruta_cache()
    metadata_path = ruta_metadata_cache()
    if cache.exists() and not rebuild_cache:
        if not metadata_path.exists():
            raise ValueError(
                f"Existe {cache} sin metadata de caché. No se usará porque no se puede "
                "garantizar su alineación; ejecuta `python parte2\\analisis.py --rebuild-cache` "
                "con el CSV original disponible."
            )
        df = pd.read_parquet(cache)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validar_cache(df, metadata)
        print("Caché cargado:", cache)
        print("Creado desde:", metadata.get("source", "origen no registrado"))
        print("Modo compartido con Parte 1:", metadata["mode"])
        print("Registros alineados:", len(df), "| particiones:", df["split"].value_counts().to_dict())
        df["issue_dt"] = pd.to_datetime(df["issue_dt"])
        return df, df["LGD_raw"].copy()

    if not RAW_LOANS_CSV.is_file():
        action = "reconstruir el caché" if rebuild_cache else "crear el caché inicial"
        raise FileNotFoundError(
            f"No se puede {action}: falta el CSV original en {RAW_LOANS_CSV}. "
            "Colócalo ahí para generar una vez el caché comprimido."
        )

    df, report = build_pd_dataset(
        mode=CFG["PD_MODE"],
        raw_path=RAW_LOANS_CSV,
        keep_columns=LGD_SOURCE_COLUMNS,
    )
    required_features = set(NUM + CAT + ["term", "emp_length"])
    missing_features = required_features - set(df.columns)
    if missing_features:
        raise ValueError(f"Faltan variables requeridas para LGD: {sorted(missing_features)}")
    df["issue_dt"] = pd.to_datetime(df["issue_d"])
    df, lgd_raw = agregar_variables(df)
    df["LGD_raw"] = lgd_raw
    print("Modo compartido con Parte 1:", report["modo"])
    print("Registros originales:", report["filas_originales"], "| registros alineados:", len(df),
          "| variables del dataset PD más auxiliares LGD:", df.shape[1])
    print("Periodo de originación:", df.issue_dt.min().date(), "->", df.issue_dt.max().date())
    print("Particiones de Parte 1:", df["split"].value_counts().to_dict())
    guardar_procesados(df, report)
    return df, lgd_raw


def agregar_variables(df):
    """Calcula LGD como pérdida neta observada sobre el monto financiado al origen."""
    df = df.copy()
    df["default"] = df["target"].astype(int)
    df["term_m"] = pd.to_numeric(df["term"], errors="coerce")
    df["emp_len"] = pd.to_numeric(df["emp_length"], errors="coerce")

    funded_amnt = pd.to_numeric(df["funded_amnt"], errors="coerce")
    df["EAD_obs"] = (funded_amnt -
                     pd.to_numeric(df["total_rec_prncp"], errors="coerce"))
    df["rec_net"] = (pd.to_numeric(df["recoveries"], errors="coerce") -
                     pd.to_numeric(df["collection_recovery_fee"], errors="coerce"))
    df["lgd_ok"] = (
        df["loan_status"].isin(LGD_STATUS)
        & funded_amnt.gt(0)
        & np.isfinite(funded_amnt)
        & df["EAD_obs"].gt(0)
        & np.isfinite(df["EAD_obs"])
        & np.isfinite(df["rec_net"])
    )

    lgd_raw = pd.Series(np.nan, index=df.index, dtype=float)
    loss_observed = df["EAD_obs"] - df["rec_net"]
    lgd_raw.loc[df["lgd_ok"]] = (
        loss_observed.loc[df["lgd_ok"]] / funded_amnt.loc[df["lgd_ok"]]
    )
    df["LGD"] = lgd_raw.clip(0, 1)
    return df, lgd_raw


def guardar_procesados(df, report):
    """Persiste el dataset completo para Parte 2 en Parquet comprimido Zstandard."""
    cache = ruta_cache()
    metadata_path = ruta_metadata_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    saved = df[COLS_SAVE].copy()
    temp_cache = cache.with_name(f"{cache.stem}.tmp{cache.suffix}")
    temp_metadata = metadata_path.with_name(f"{metadata_path.stem}.tmp{metadata_path.suffix}")
    saved.to_parquet(temp_cache, index=False, compression="zstd")
    metadata = {
        "version": CFG["CACHE_VERSION"],
        "mode": CFG["PD_MODE"],
        "rows": len(saved),
        "columns": COLS_SAVE,
        "source": str(RAW_LOANS_CSV),
        "source_size_bytes": RAW_LOANS_CSV.stat().st_size,
        "source_modified_ns": RAW_LOANS_CSV.stat().st_mtime_ns,
        "part1_report": report,
        "compression": "zstd",
    }
    temp_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temp_cache.replace(cache)
    temp_metadata.replace(metadata_path)
    print("Caché comprimido guardado:", cache)


def resumen_lgd(df, lgd_raw):
    d = df[df.lgd_ok]
    print("Créditos resueltos:", len(df), "| casos malos según target PD:", df.default.sum(),
          "| con LGD definida:", len(d))
    print("Casos malos sin LGD observada (no Charged Off, EAD<=0 o datos incompletos):",
          int(((df.default == 1) & ~df.lgd_ok).sum()))
    print("LGD cruda < 0:", round((lgd_raw[df.lgd_ok] < 0).mean(), 3),
          "| > 1:", round((lgd_raw[df.lgd_ok] > 1).mean(), 3), "(se recortan)")
    print(d.LGD.describe())
    print("LGD == 1:", (d.LGD == 1).mean().round(3), "| LGD == 0:", (d.LGD == 0).mean().round(3))


def particionar_datos(df):
    """Respeta las etiquetas train/val/test asignadas por la Parte 1."""
    if df["split"].isna().any() or not set(df["split"].unique()) <= {"train", "val", "test"}:
        raise ValueError("El dataset de Parte 1 contiene particiones faltantes o desconocidas.")
    parts = {name: df[df["split"] == name].copy() for name in ("train", "val", "test")}
    if any(part.empty for part in parts.values()):
        raise ValueError(f"Falta al menos una partición en el dataset: {[k for k, v in parts.items() if v.empty]}")

    print(pd.DataFrame({
        "n": [len(parts[s]) for s in ("train", "val", "test")],
        "tasa_target_malo": [parts[s].default.mean().round(4) for s in ("train", "val", "test")],
        "n_con_LGD": [int(parts[s].lgd_ok.sum()) for s in ("train", "val", "test")],
        "LGD_media": [parts[s].LGD.mean().round(3) for s in ("train", "val", "test")],
    }, index=["train", "val", "test"]))
    return parts


def construir_preprocesador():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_ALL),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


def matrices_lgd(tr, va, te):
    """Ajusta el preprocesador solo con entrenamiento y devuelve (X, y) de LGD por partición."""
    for name, part in (("train", tr), ("validación", va), ("prueba", te)):
        if not part["lgd_ok"].any():
            raise ValueError(f"No hay casos con LGD observada en la partición de {name}.")
    pre = construir_preprocesador()
    X = {"tr": pre.fit_transform(tr[FEATURES]),
         "va": pre.transform(va[FEATURES]),
         "te": pre.transform(te[FEATURES])}
    datos = {}
    for k, part in zip(("tr", "va", "te"), (tr, va, te)):
        m = part["lgd_ok"].to_numpy()
        datos[k] = (X[k][m], part.loc[m, "LGD"].to_numpy(dtype=float))
    print("Matrices LGD:", datos["tr"][0].shape, datos["va"][0].shape, datos["te"][0].shape)
    datos["X_all"] = X
    datos["parts"] = {"tr": tr, "va": va, "te": te}
    return datos


# ======================================================================
# Modelos
# ======================================================================
def metricas(y, p):
    return mean_absolute_error(y, p), float(np.sqrt(mean_squared_error(y, p)))


def modelo_lineal(datos):
    (Xtr, ytr), _, (Xte, yte) = datos["tr"], datos["va"], datos["te"]

    t0 = time.time()
    lin = LinearRegression().fit(Xtr, ytr)
    t_lin = time.time() - t0
    p_lin_all = lin.predict(datos["X_all"]["te"]).clip(0, 1)
    n_par = lin.coef_.size + 1

    p_media_all = np.full(len(p_lin_all), ytr.mean())

    titulo("Modelo de regresion lineal")
    mask = datos["parts"]["te"]["lgd_ok"].to_numpy()
    print("Media constante  MAE=%.4f RMSE=%.4f" % metricas(yte, p_media_all[mask]))
    print("Regresión lineal MAE=%.4f RMSE=%.4f | parámetros=%d | tiempo=%.2fs"
          % (*metricas(yte, p_lin_all[mask]), n_par, t_lin))
    return p_media_all, p_lin_all, n_par, t_lin


def build_mlp(units, dropout, lr, n_in):
    m = keras.Sequential([keras.Input(shape=(n_in,))])
    for u in units:
        m.add(layers.Dense(u, activation="relu"))
        if dropout > 0:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mean_squared_error", metrics=["mae"])
    return m


def entrenar(cfg, seed, datos):
    units, dropout, lr = cfg
    (Xtr, ytr), (Xva, yva), _ = datos["tr"], datos["va"], datos["te"]

    keras.utils.set_random_seed(seed)
    m = build_mlp(units, dropout, lr, Xtr.shape[1])
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=CFG["EPOCHS"], batch_size=CFG["BATCH"], verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=CFG["PATIENCE"],
                                                       restore_best_weights=True)])
    return m, h


def buscar_configuracion(datos):
    titulo("Perceptron multicapa")
    t0 = time.time()
    resultados = []
    for cfg in GRID:
        m, h = entrenar(cfg, CFG["SEED"], datos)
        res = (cfg, min(h.history["val_loss"]), min(h.history["val_mae"]), m.count_params())
        resultados.append(res)
        print(cfg, "val_loss=%.4f val_mae=%.4f params=%d" % res[1:])

    mejor = min(resultados, key=lambda r: r[1])[0]
    print("Mejor configuración:", mejor, "| configuraciones entrenadas:", len(GRID),
          "| tiempo búsqueda: %.1fs" % (time.time() - t0))
    return mejor, time.time() - t0


def evaluar_semillas(mejor_cfg, datos):
    """Reentrena con varias semillas y promedia sus predicciones para el conjunto completo."""
    yte = datos["te"][1]
    Xte = datos["te"][0]
    Xte_all = datos["X_all"]["te"]

    t0 = time.time()
    filas, predicciones, n_par = [], [], None
    for s in range(CFG["N_SEEDS"]):
        m, _ = entrenar(mejor_cfg, CFG["SEED"] + s, datos)
        p_labeled = m.predict(Xte, verbose=0).ravel().clip(0, 1)
        p_all = m.predict(Xte_all, verbose=0).ravel().clip(0, 1)
        filas.append(metricas(yte, p_labeled))
        predicciones.append(p_all)
        n_par = m.count_params()
    t_sem = time.time() - t0

    S = pd.DataFrame(filas, columns=["MAE", "RMSE"])
    p_mlp_all = np.mean(predicciones, axis=0)

    resumen = pd.DataFrame({
        "MAE":  [S.MAE.mean(), S.MAE.std(), S.MAE.min(), S.MAE.max()],
        "RMSE": [S.RMSE.mean(), S.RMSE.std(), S.RMSE.min(), S.RMSE.max()],
    }, index=["media", "desv. est.", "mejor", "peor"]).round(4)

    print("MLP (%d semillas), config=%s, parámetros=%d, tiempo=%.1fs"
          % (CFG["N_SEEDS"], mejor_cfg, n_par, t_sem))
    print(resumen)
    return S, p_mlp_all, n_par, t_sem


# ======================================================================
# Reporte
# ======================================================================
def tabla_comparativa(yte, p_media, p_lin, S, n_par_lin, n_par_mlp):
    tabla = pd.DataFrame(
        [metricas(yte, p_media), metricas(yte, p_lin), (S.MAE.mean(), S.RMSE.mean())],
        index=["Media constante", "Regresión lineal", "MLP (media de semillas)"],
        columns=["MAE", "RMSE"]).round(4)
    tabla["Parámetros"] = [1, n_par_lin, n_par_mlp]
    tabla["Configs entrenadas"] = [0, 1, len(GRID)]
    print(tabla)


def graficar_distribuciones(yte, p_lin, p_mlp):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, (nombre, p) in zip(ax, [("Regresión lineal", p_lin), ("MLP", p_mlp)]):
        a.hist(yte, bins=30, range=(0, 1), density=True, alpha=0.5, label="Observada")
        a.hist(p, bins=30, range=(0, 1), density=True, alpha=0.5, label="Predicha")
        a.set_title(nombre)
        a.set_xlabel("LGD")
        a.legend()
    ax[0].set_ylabel("Densidad")
    plt.tight_layout()
    output_dir = Path(CFG["OUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "lgd_distribuciones.png", dpi=150)
    plt.close(fig)

    print("Media observada: %.3f | lineal: %.3f | MLP: %.3f" % (yte.mean(), p_lin.mean(), p_mlp.mean()))
    print("Desv. observada: %.3f | lineal: %.3f | MLP: %.3f" % (yte.std(), p_lin.std(), p_mlp.std()))


def guardar_predicciones_lgd(datos, p_lineal, p_mlp):
    """Guarda las predicciones LGD para todos los créditos del test, no solo los castigados."""
    test = datos["parts"]["te"]
    if len(test) != len(p_lineal) or len(test) != len(p_mlp):
        raise ValueError("Las predicciones LGD no corresponden a todas las filas del test.")
    predictions = test[["id", "split", "loan_status", "target", "funded_amnt",
                        "EAD_obs", "LGD", "lgd_ok"]].copy()
    predictions["LGD_pred_lineal"] = p_lineal
    predictions["LGD_pred_mlp"] = p_mlp
    output_dir = Path(CFG["OUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lgd_predicciones_test.parquet"
    predictions.to_parquet(path, index=False)
    print("Predicciones LGD de prueba:", path)
    return predictions


def calcular_perdida_esperada(pred_lgd):
    """Calcula un proxy por crédito como PD de Parte 1 × LGD predicha × monto de origen.

    No es una E[L] estándar a la fecha de default: PD incluye estados Late y LGD
    se observa solo en Charged Off. La severidad y el monto están referidos al origen.
    """
    base = pred_lgd[["id", "split", "loan_status", "target", "funded_amnt",
                     "LGD", "lgd_ok", "LGD_pred_lineal", "LGD_pred_mlp"]].copy()
    base = base.rename(columns={"LGD": "LGD_observada"})
    if base["id"].duplicated().any():
        raise ValueError("Hay identificadores duplicados en las predicciones LGD.")
    if not base["split"].eq("test").all():
        raise ValueError("El proxy de pérdida solo se calcula para créditos del conjunto test.")
    base["EAD_origin"] = pd.to_numeric(base.pop("funded_amnt"), errors="coerce")
    if not np.isfinite(base["EAD_origin"]).all() or (base["EAD_origin"] < 0).any():
        raise ValueError("funded_amnt (monto de origen) debe ser numérico y no negativo.")
    for col in ("LGD_pred_lineal", "LGD_pred_mlp"):
        if not np.isfinite(base[col]).all() or not base[col].between(0, 1).all():
            raise ValueError(f"{col} debe contener valores finitos entre 0 y 1.")

    for model_name, path in CFG["PD_PREDICTIONS"].items():
        if not path.is_file():
            raise FileNotFoundError(
                f"No se encontró {path}. Ejecuta primero `python run.py --etapas logistica mlp`."
            )
        pd_predictions = pd.read_parquet(path)
        if not {"id", "split", "pd"} <= set(pd_predictions.columns):
            raise ValueError(f"El archivo de PD {path} debe contener id, split y pd.")
        pd_test = pd_predictions.loc[pd_predictions["split"] == "test", ["id", "pd"]]
        if pd_test["id"].duplicated().any():
            raise ValueError(f"Hay identificadores duplicados en las predicciones PD de {model_name}.")
        if set(pd_test["id"]) != set(base["id"]):
            raise ValueError(
                f"Los ids del test de PD ({model_name}) y LGD no coinciden; "
                "revisa que Parte 1 y Parte 2 usen el mismo modo y partición."
            )
        base = base.merge(pd_test, on="id", how="left", validate="one_to_one")
        base = base.rename(columns={"pd": f"PD_{model_name}"})
        if base[f"PD_{model_name}"].isna().any():
            raise ValueError(f"Hay PD faltantes después de unir las predicciones de {model_name}.")
        if (not np.isfinite(base[f"PD_{model_name}"]).all()
                or not base[f"PD_{model_name}"].between(0, 1).all()):
            raise ValueError(f"PD_{model_name} debe contener valores finitos entre 0 y 1.")
        for lgd_name in ("lineal", "mlp"):
            base[f"EL_proxy_{model_name}_{lgd_name}"] = (
                base[f"PD_{model_name}"] * base[f"LGD_pred_{lgd_name}"] * base["EAD_origin"]
            )

    output_dir = Path(CFG["OUT_DIR"])
    path = output_dir / "perdida_esperada_test.parquet"
    base.to_parquet(path, index=False)
    print("Proxy de pérdida de prueba:", path)
    expected_loss = base.filter(regex=r"^EL_proxy_")
    summary = pd.DataFrame({
        "proxy_perdida_total_test": expected_loss.sum(),
        "proxy_perdida_media_por_credito": expected_loss.mean(),
    }).round(2)
    print(summary)
    return base


def guardar_metricas(parts, yte, p_media, p_lineal, predicciones_mlp, seed_metrics,
                     best_cfg, n_params_linear, n_params_mlp, linear_seconds,
                     search_seconds, seed_seconds):
    """Guarda métricas, semillas, configuración y presupuesto del experimento."""
    test = parts["test"]
    pred_labeled = predicciones_mlp[test["lgd_ok"].to_numpy()]

    def summary(frame, column):
        return {
            "media": float(frame[column].mean()),
            "desv_est": float(frame[column].std(ddof=1)),
            "mejor": float(frame[column].min()),
            "peor": float(frame[column].max()),
        }

    test_mask = test["lgd_ok"].to_numpy()
    results = {
        "modo": CFG["PD_MODE"],
        "n_por_particion": {name: int(len(part)) for name, part in parts.items()},
        "n_test_con_lgd_observada": int(test["lgd_ok"].sum()),
        "lgd_definition": (
            "(funded_amnt - total_rec_prncp - recoveries + "
            "collection_recovery_fee) / funded_amnt; pérdida observada acumulada "
            "sobre el principal financiado al origen, recortada a [0, 1]"
        ),
        "lgd_observed_statuses": LGD_STATUS,
        "pd_bad_statuses": PD_BAD_STATUSES,
        "ead_loss_projection": "funded_amnt (monto financiado en originación)",
        "loss_projection_interpretation": (
            "Proxy indicativo, no pérdida esperada estándar a la fecha de default: "
            "el target PD incluye estados Late (31-120 days), la LGD observada solo "
            "incluye Charged Off, y las recuperaciones pueden estar censuradas."
        ),
        "modelo_referencia": {
            "tipo": "Regresión lineal",
            "n_parametros": int(n_params_linear),
            "segundos_entrenamiento": float(linear_seconds),
            "test": dict(zip(("mae", "rmse"), map(float, metricas(yte, p_lineal[test_mask])))),
        },
        "mlp": {
            "configuracion": {
                "unidades": list(best_cfg[0]),
                "dropout": best_cfg[1],
                "learning_rate": best_cfg[2],
            },
            "funcion_perdida": "mean_squared_error",
            "funcion_salida": "sigmoid",
            "n_configuraciones": len(GRID),
            "segundos_busqueda": float(search_seconds),
            "n_semillas": int(CFG["N_SEEDS"]),
            "n_parametros": int(n_params_mlp),
            "segundos_entrenamiento_semillas": float(seed_seconds),
            "segundos_totales_busqueda_y_semillas": float(search_seconds + seed_seconds),
            "test_por_semilla": {
                "mae": summary(seed_metrics, "MAE"),
                "rmse": summary(seed_metrics, "RMSE"),
            },
            "test_ensamble_semillas": dict(
                zip(("mae", "rmse"), map(float, metricas(yte, pred_labeled)))
            ),
        },
        "referencia_media_constante_test": dict(
            zip(("mae", "rmse"), map(float, metricas(yte, p_media[test_mask])))
        ),
    }
    output_dir = Path(CFG["OUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lgd_metricas.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    seed_metrics.to_csv(output_dir / "lgd_mlp_semillas.csv", index=False)
    return results


# ======================================================================
# Pipeline principal
# ======================================================================
def ejecutar_analisis(rebuild_cache=False):
    df, lgd_cruda = cargar_datos(rebuild_cache=rebuild_cache)

    titulo("Analisis estadistico")
    resumen_lgd(df, lgd_cruda)

    titulo("Datos del entrenamiento")
    particiones = particionar_datos(df)
    matrices = matrices_lgd(
        particiones["train"], particiones["val"], particiones["test"]
    )

    pred_media_todas, pred_lineal_todas, n_parametros_lineal, tiempo_lineal = modelo_lineal(matrices)
    configuracion_mejor, tiempo_busqueda = buscar_configuracion(matrices)

    titulo("Evaluacion del modelo")
    metricas_semillas, pred_mlp_todas, n_parametros_mlp, tiempo_semillas = evaluar_semillas(
        configuracion_mejor, matrices
    )

    lgd_test_observada = matrices["te"][1]
    mascara_lgd_test = particiones["test"]["lgd_ok"].to_numpy()
    pred_media = pred_media_todas[mascara_lgd_test]
    pred_lineal = pred_lineal_todas[mascara_lgd_test]
    pred_mlp = pred_mlp_todas[mascara_lgd_test]
    tabla_comparativa(
        lgd_test_observada, pred_media, pred_lineal, metricas_semillas,
        n_parametros_lineal, n_parametros_mlp
    )
    graficar_distribuciones(lgd_test_observada, pred_lineal, pred_mlp)

    predicciones_lgd = guardar_predicciones_lgd(matrices, pred_lineal_todas, pred_mlp_todas)
    guardar_metricas(
        particiones, lgd_test_observada, pred_media_todas, pred_lineal_todas,
        pred_mlp_todas, metricas_semillas, configuracion_mejor,
        n_parametros_lineal, n_parametros_mlp, tiempo_lineal, tiempo_busqueda,
        tiempo_semillas
    )
    titulo("Proxy de pérdida = PD × severidad × monto financiado al origen")
    calcular_perdida_esperada(predicciones_lgd)


def main():
    parser = argparse.ArgumentParser(description="Entrena los modelos LGD y calcula pérdida esperada.")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="reconstruye el caché desde data/raw/loan.csv en vez de cargarlo",
    )
    ejecutar_analisis(rebuild_cache=parser.parse_args().rebuild_cache)


if __name__ == "__main__":
    main()