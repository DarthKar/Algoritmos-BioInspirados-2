import os
import time

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

import config

# ======================================================================
# Constantes
# ======================================================================
CSV = config.find_csv()
CFG = config.CFG

NUM = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
       "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]
CAT = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
RAW = ["term", "emp_length", "issue_d"]
AUX = ["loan_status", "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]
NUM_ALL = NUM + ["term_m", "emp_len"]
FEATURES = NUM_ALL + CAT

DEFAULT_PD = ["Charged Off", "Default", "Late (31-120 days)",
              "Does not meet the credit policy. Status:Charged Off"]
PAID = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
LGD_STATUS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]

# (unidades por capa, dropout, learning rate)
GRID = [((32,), 0.0, 1e-3),
        ((64,), 0.0, 1e-3),
        ((64, 32), 0.1, 1e-3),
        ((128, 64), 0.2, 1e-3),
        ((128, 64, 32), 0.2, 5e-4),
        ((256, 128), 0.3, 5e-4)]

COLS_SAVE = list(dict.fromkeys(
    FEATURES + ["issue_dt", "loan_status", "default", "funded_amnt",
                "total_rec_prncp", "EAD_obs", "rec_net", "lgd_ok", "LGD"]))


def titulo(texto):
    print(f"\n========={texto}=========\n")


# ======================================================================
# Datos
# ======================================================================
def cargar_datos(csv):
    """Lee solo las columnas necesarias (el archivo es grande)."""
    n_cols_total = pd.read_csv(csv, nrows=0).shape[1]
    df = pd.read_csv(csv, usecols=NUM + CAT + RAW + AUX, low_memory=False)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

    print("Registros del archivo:", len(df), "| variables del archivo:", n_cols_total,
          "| variables usadas:", df.shape[1] - 1)
    print("Periodo de originación:", df.issue_dt.min().date(), "->", df.issue_dt.max().date())
    print(df.loan_status.value_counts())
    return df


def filtrar_resueltos(df):
    """Conserva créditos resueltos y define la variable de incumplimiento."""
    df = df[df.loan_status.isin(DEFAULT_PD + PAID)].copy()
    df["default"] = df.loan_status.isin(DEFAULT_PD).astype(int)
    if CFG["DEMO"]:
        df = df.sample(frac=CFG["DEMO_FRAC"], random_state=CFG["SEED"]).copy()
    return df


def agregar_variables(df):
    """Variables de texto, EAD, recuperación neta y LGD (solo castigados con EAD > 0)."""
    df["term_m"] = df["term"].str.extract(r"(\d+)")[0].astype(float)
    df["emp_len"] = df["emp_length"].str.extract(r"(\d+)")[0].astype(float)
    df.loc[df["emp_length"].str.startswith("<", na=False), "emp_len"] = 0

    df["EAD_obs"] = df.funded_amnt - df.total_rec_prncp
    df["rec_net"] = df.recoveries - df.collection_recovery_fee
    df["lgd_ok"] = df.loan_status.isin(LGD_STATUS) & (df.EAD_obs > 0)

    lgd_raw = 1 - df.rec_net / df.EAD_obs
    df["LGD"] = np.where(df.lgd_ok, lgd_raw.clip(0, 1), np.nan)
    return df, lgd_raw


def resumen_lgd(df, lgd_raw):
    d = df[df.lgd_ok]
    print("Créditos resueltos:", len(df), "| incumplidos (PD):", df.default.sum(),
          "| con LGD definida:", len(d))
    print("Incumplidos sin LGD (Default/Late/EAD<=0, censurados):",
          int(((df.default == 1) & ~df.lgd_ok).sum()))
    print("LGD cruda < 0:", round((lgd_raw[df.lgd_ok] < 0).mean(), 3),
          "| > 1:", round((lgd_raw[df.lgd_ok] > 1).mean(), 3), "(se recortan)")
    print(d.LGD.describe())
    print("LGD == 1:", (d.LGD == 1).mean().round(3), "| LGD == 0:", (d.LGD == 0).mean().round(3))


def guardar_procesados(df):
    """La ejecución no debe depender de internet: se guarda una copia procesada."""
    ruta = os.path.join(CFG["PROC_DIR"], "lending_resueltos.csv.gz")
    df[COLS_SAVE].to_csv(ruta, index=False)


def split_temporal(df):
    t = df.issue_dt
    tr = df[t < CFG["TRAIN_END"]]
    va = df[(t >= CFG["TRAIN_END"]) & (t < CFG["VAL_END"])]
    te = df[t >= CFG["VAL_END"]]

    print(pd.DataFrame({
        "n": [len(x) for x in (tr, va, te)],
        "tasa_incumplimiento": [x.default.mean().round(4) for x in (tr, va, te)],
        "n_con_LGD": [int(x.lgd_ok.sum()) for x in (tr, va, te)],
        "LGD_media": [x.LGD.mean().round(3) for x in (tr, va, te)],
    }, index=["train", "val", "test"]))
    return tr, va, te


def construir_preprocesador():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_ALL),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


def matrices_lgd(tr, va, te):
    """Ajusta el preprocesador solo con entrenamiento y devuelve (X, y) de LGD por partición."""
    pre = construir_preprocesador()
    X = {"tr": pre.fit_transform(tr[FEATURES]),
         "va": pre.transform(va[FEATURES]),
         "te": pre.transform(te[FEATURES])}
    datos = {}
    for k, part in zip(("tr", "va", "te"), (tr, va, te)):
        m = part.lgd_ok.values
        datos[k] = (X[k][m], part.LGD.values[m])
    print("Matrices LGD:", datos["tr"][0].shape, datos["va"][0].shape, datos["te"][0].shape)
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
    p_lin = lin.predict(Xte).clip(0, 1)
    n_par = lin.coef_.size + 1

    p_media = np.full_like(yte, ytr.mean())  # referencia trivial

    titulo("Modelo de regresion lineal")
    print("Media constante  MAE=%.4f RMSE=%.4f" % metricas(yte, p_media))
    print("Regresión lineal MAE=%.4f RMSE=%.4f | parámetros=%d | tiempo=%.2fs"
          % (*metricas(yte, p_lin), n_par, t_lin))
    return p_media, p_lin, n_par


def build_mlp(units, dropout, lr, n_in):
    m = keras.Sequential([keras.Input(shape=(n_in,))])
    for u in units:
        m.add(layers.Dense(u, activation="relu"))
        if dropout > 0:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["mae"])
    return m


def entrenar(cfg, seed, datos):
    units, dropout, lr = cfg
    (Xtr, ytr), (Xva, yva), _ = datos["tr"], datos["va"], datos["te"]

    keras.utils.set_random_seed(seed)
    m = build_mlp(units, dropout, lr, Xtr.shape[1])
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=config.EPOCHS, batch_size=CFG["BATCH"], verbose=0,
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
    return mejor


def evaluar_semillas(mejor_cfg, datos):
    """Reentrena con varias semillas; la primera queda como modelo representativo."""
    yte = datos["te"][1]
    Xte = datos["te"][0]

    t0 = time.time()
    filas, mlp, p_mlp = [], None, None
    for s in range(config.N_SEEDS):
        m, _ = entrenar(mejor_cfg, CFG["SEED"] + s, datos)
        p = m.predict(Xte, verbose=0).ravel()
        filas.append(metricas(yte, p))
        if s == 0:
            mlp, p_mlp = m, p
    t_sem = time.time() - t0

    S = pd.DataFrame(filas, columns=["MAE", "RMSE"])
    n_par = mlp.count_params()

    resumen = pd.DataFrame({
        "MAE":  [S.MAE.mean(), S.MAE.std(), S.MAE.min(), S.MAE.max()],
        "RMSE": [S.RMSE.mean(), S.RMSE.std(), S.RMSE.min(), S.RMSE.max()],
    }, index=["media", "desv. est.", "mejor", "peor"]).round(4)

    print("MLP (%d semillas), config=%s, parámetros=%d, tiempo=%.1fs"
          % (config.N_SEEDS, mejor_cfg, n_par, t_sem))
    print(resumen)
    return S, mlp, p_mlp, n_par


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
    plt.savefig(os.path.join(CFG["OUT_DIR"], "lgd_distribuciones.png"), dpi=150)
    plt.show()

    print("Media observada: %.3f | lineal: %.3f | MLP: %.3f" % (yte.mean(), p_lin.mean(), p_mlp.mean()))
    print("Desv. observada: %.3f | lineal: %.3f | MLP: %.3f" % (yte.std(), p_lin.std(), p_mlp.std()))


# ======================================================================
# Pipeline principal
# ======================================================================
def main():
    df = cargar_datos(CSV)

    titulo("Analisis Estadisco")
    df = filtrar_resueltos(df)
    df, lgd_raw = agregar_variables(df)
    resumen_lgd(df, lgd_raw)
    guardar_procesados(df)

    titulo("Datos del entrenamiento")
    tr, va, te = split_temporal(df)
    datos = matrices_lgd(tr, va, te)

    p_media, p_lin, n_par_lin = modelo_lineal(datos)
    mejor_cfg = buscar_configuracion(datos)

    titulo("Evaluacion del modelo")
    S, mlp, p_mlp, n_par_mlp = evaluar_semillas(mejor_cfg, datos)

    yte = datos["te"][1]
    tabla_comparativa(yte, p_media, p_lin, S, n_par_lin, n_par_mlp)
    graficar_distribuciones(yte, p_lin, p_mlp)


if __name__ == "__main__":
    main()