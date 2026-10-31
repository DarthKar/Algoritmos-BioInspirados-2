"""Recalcula y muestra análisis PD, LGD y pérdida esperada con modelos guardados.

No entrena modelos ni ejecuta búsquedas de hiperparámetros.
Antes del cálculo permite ajustar parámetros de evaluación desde un menú.

Uso:
    python orquestador.py
    python orquestador.py --solo parte1
    python orquestador.py --solo parte2
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTE1 = ROOT / "parte1"
PARTE2 = ROOT / "parte2"
PD_DATA = PARTE1 / "datos" / "procesados" / "pd_matured.parquet"
PD_MODELS = PARTE1 / "salidas" / "modelos"
PD_RESULTS = PARTE1 / "salidas" / "resultados"
LGD_OUTPUTS = PARTE2 / "outputs"
LGD_MODELS = LGD_OUTPUTS / "models"
LGD_DATA = PARTE2 / "data" / "processed" / "lending_matured_lgd.parquet"
LGD_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "term_m", "emp_len", "grade", "sub_grade", "home_ownership",
    "verification_status", "purpose",
]


def titulo(texto):
    print(f"\n{'=' * 76}\n{texto}\n{'=' * 76}", flush=True)


def exigir_archivos(paths):
    faltantes = [path for path in paths if not path.is_file()]
    if faltantes:
        listado = "\n".join(f"  - {path}" for path in faltantes)
        raise FileNotFoundError(f"Faltan datos o modelos ya entrenados:\n{listado}")


def pedir_entero(prompt, actual, minimo, maximo):
    while True:
        valor = input(f"{prompt} [{actual}]: ").strip()
        if not valor:
            return actual
        try:
            numero = int(valor)
        except ValueError:
            print("Escribe un número entero.")
            continue
        if minimo <= numero <= maximo:
            return numero
        print(f"El valor debe estar entre {minimo} y {maximo}.")


def pedir_decimal(prompt, actual, minimo, maximo):
    while True:
        valor = input(f"{prompt} [{actual:g}]: ").strip()
        if not valor:
            return actual
        try:
            numero = float(valor.replace(",", "."))
        except ValueError:
            print("Escribe un número decimal válido.")
            continue
        if minimo <= numero <= maximo:
            return numero
        print(f"El valor debe estar entre {minimo:g} y {maximo:g}.")


def configurar_calculos(partes):
    """Permite ajustar solo controles de inferencia y evaluación."""
    modelos_lgd = sorted(
        LGD_MODELS.glob("lgd_mlp_seed_*.keras"),
        key=lambda path: int(re.search(r"seed_(\d+)", path.stem).group(1)),
    )
    valores = {
        "semillas_lgd": len(modelos_lgd),
        "batch_size": 8192,
        "n_bins": 10,
        "umbral_pd": 0.5,
    }

    titulo("PARÁMETROS DE LOS CÁLCULOS")
    if not sys.stdin.isatty():
        print("Entrada no interactiva: se usarán los valores predeterminados.")
        return valores

    print("Estos valores ajustan los cálculos, no el entrenamiento de los modelos.")
    while True:
        opciones = []
        if "parte2" in partes:
            opciones.append(("1", f"MLP LGD del ensamble: {valores['semillas_lgd']} de {len(modelos_lgd)}"))
            opciones.append(("2", f"Lote de predicción: {valores['batch_size']}"))
        if "parte1" in partes:
            opciones.append(("3", f"Intervalos de calibración PD: {valores['n_bins']}"))
            opciones.append(("4", f"Umbral de clasificación PD: {valores['umbral_pd']:.2f}"))
        for clave, texto in opciones:
            print(f"  {clave}. {texto}")
        print("  0. Continuar con estos parámetros")
        opcion = input("Elige qué cambiar [0]: ").strip() or "0"
        if opcion == "0":
            return valores
        if opcion == "1" and "parte2" in partes:
            valores["semillas_lgd"] = pedir_entero(
                "Cantidad de modelos LGD ya entrenados", valores["semillas_lgd"],
                1, len(modelos_lgd),
            )
        elif opcion == "2" and "parte2" in partes:
            valores["batch_size"] = pedir_entero(
                "Tamaño del lote para predicción", valores["batch_size"], 1, 131072,
            )
        elif opcion == "3" and "parte1" in partes:
            valores["n_bins"] = pedir_entero(
                "Cantidad de intervalos de calibración", valores["n_bins"], 2, 100,
            )
        elif opcion == "4" and "parte1" in partes:
            valores["umbral_pd"] = pedir_decimal(
                "Umbral de PD (0 a 1)", valores["umbral_pd"], 0, 1,
            )
        else:
            print("Opción no válida.")


def guardar_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def calcular_metricas_pd(y, probabilidad, n_bins, umbral):
    from parte1.evaluacion.metrics import (
        calibration_in_the_large,
        calibration_table,
        evaluate,
    )
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

    predicho = (probabilidad >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, predicho, labels=[0, 1]).ravel()
    return {
        **evaluate(y, probabilidad),
        "log_loss": float(log_loss(y, probabilidad)),
        **calibration_in_the_large(y, probabilidad),
        "n": int(len(y)),
        "calibracion": calibration_table(y, probabilidad, n_bins=n_bins).to_dict("records"),
        "clasificacion": {
            "umbral": float(umbral),
            "accuracy": float(accuracy_score(y, predicho)),
            "f1": float(f1_score(y, predicho, zero_division=0)),
            "vn": int(vn),
            "fp": int(fp),
            "fn": int(fn),
            "vp": int(vp),
        },
    }


def recalcular_parte1(parametros):
    import joblib
    import pandas as pd

    from parte1.modelos.core import predict_proba
    from parte1.modelos.mlp import load_mlp
    from parte1.preparacion.dataset import load_pd_dataset

    titulo("PARTE 1: PROBABILIDAD DE INCUMPLIMIENTO (PD)")
    exigir_archivos([
        PD_DATA,
        PD_MODELS / "pd_logistic.joblib",
        PD_MODELS / "pd_mlp.pt",
        PD_MODELS / "pd_mlp_preprocessor.joblib",
        PD_RESULTS / "pd_logistic_metricas.json",
        PD_RESULTS / "pd_mlp_metricas.json",
    ])

    df = load_pd_dataset("matured")
    pipeline_lr = joblib.load(PD_MODELS / "pd_logistic.joblib")
    pre_mlp, modelo_mlp = load_mlp("pd_mlp")
    probabilidades = {"pd_logistic": [], "pd_mlp": []}
    metricas_modelos = {"pd_logistic": {}, "pd_mlp": {}}
    predicciones = {"pd_logistic": [], "pd_mlp": []}

    for split in ("train", "val", "test"):
        part = df.loc[df["split"] == split]
        if part.empty:
            raise ValueError(f"No hay registros en la partición {split!r} de PD.")

        p_lr = pipeline_lr.predict_proba(part)[:, 1]
        X_mlp = pre_mlp.transform(part)
        p_mlp = predict_proba(modelo_mlp, X_mlp, batch_size=parametros["batch_size"])
        y = part["target"].to_numpy()

        for nombre, prob in (("pd_logistic", p_lr), ("pd_mlp", p_mlp)):
            if len(prob) != len(part):
                raise ValueError(f"Predicciones desalineadas para {nombre} en {split}.")
            probabilidades[nombre].append(prob)
            metricas_modelos[nombre][split] = calcular_metricas_pd(
                y, prob, parametros["n_bins"], parametros["umbral_pd"]
            )
            predicciones[nombre].append(pd.DataFrame({
                "id": part["id"].to_numpy(),
                "split": split,
                "target": y,
                "pd": prob,
            }))

    for nombre in predicciones:
        (PD_RESULTS / f"{nombre}_predicciones.parquet").parent.mkdir(parents=True, exist_ok=True)
        pd.concat(predicciones[nombre], ignore_index=True).to_parquet(
            PD_RESULTS / f"{nombre}_predicciones.parquet", index=False
        )
        metricas_path = PD_RESULTS / f"{nombre}_metricas.json"
        resultado_guardado = json.loads(metricas_path.read_text(encoding="utf-8"))
        if nombre == "pd_logistic":
            resultado_guardado["metricas"] = {
                split: {k: v for k, v in metricas_modelos[nombre][split].items()
                       if k not in ("calibracion", "clasificacion")}
                for split in ("train", "val", "test")
            }
            resultado_guardado["calibracion_test"] = metricas_modelos[nombre]["test"]["calibracion"]
        else:
            resultado_guardado["metricas_modelo_guardado"] = {
                split: {k: v for k, v in metricas_modelos[nombre][split].items()
                       if k not in ("calibracion", "clasificacion")}
                for split in ("train", "val", "test")
            }
            resultado_guardado["calibracion_test"] = metricas_modelos[nombre]["test"]["calibracion"]
        resultado_guardado["calculos_orquestador"] = {
            "intervalos_calibracion": parametros["n_bins"],
            "umbral_clasificacion_pd": parametros["umbral_pd"],
        }
        guardar_json(metricas_path, resultado_guardado)

    print(f"{'Modelo':<20} {'Partición':<10} {'AUC':>8} {'KS':>8} {'Brier':>9} "
          f"{'Log-loss':>10} {'Accuracy':>10} {'F1':>8}")
    print("-" * 92)
    for nombre, label in (("pd_logistic", "Logística"), ("pd_mlp", "MLP guardado")):
        for split in ("train", "val", "test"):
            m = metricas_modelos[nombre][split]
            print(f"{label:<20} {split:<10} {m['auc']:>8.4f} {m['ks']:>8.4f} "
                  f"{m['brier']:>9.4f} {m['log_loss']:>10.4f} "
                  f"{m['clasificacion']['accuracy']:>10.4f} {m['clasificacion']['f1']:>8.4f}")
    print(f"Intervalos de calibración: {parametros['n_bins']}; "
          f"umbral de clasificación: {parametros['umbral_pd']:.2f}")
    return {
        nombre: pd.concat(predicciones[nombre], ignore_index=True)
        for nombre in predicciones
    }


def recalcular_parte2(predicciones_pd, parametros):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from tensorflow import keras

    titulo("PARTE 2: PÉRDIDA DADO INCUMPLIMIENTO (LGD)")
    metricas_path = LGD_OUTPUTS / "lgd_metricas.json"
    exigir_archivos([
        LGD_DATA,
        LGD_MODELS / "lgd_preprocessor.joblib",
        LGD_MODELS / "lgd_linear.joblib",
        metricas_path,
    ])

    resultados = json.loads(metricas_path.read_text(encoding="utf-8"))
    datos = pd.read_parquet(LGD_DATA)
    test = datos.loc[datos["split"] == "test"].copy()
    train_observada = datos.loc[(datos["split"] == "train") & datos["lgd_ok"], "LGD"]
    test_observada = test.loc[test["lgd_ok"], "LGD"].to_numpy(dtype=float)
    if test.empty or test_observada.size == 0 or train_observada.empty:
        raise ValueError("El caché LGD no contiene datos suficientes de train/test observados.")

    preprocesador = joblib.load(LGD_MODELS / "lgd_preprocessor.joblib")
    X_test = preprocesador.transform(test[LGD_FEATURES])
    modelo_lineal = joblib.load(LGD_MODELS / "lgd_linear.joblib")
    pred_lineal = np.clip(modelo_lineal.predict(X_test), 0, 1)

    modelos_paths = sorted(
        LGD_MODELS.glob("lgd_mlp_seed_*.keras"),
        key=lambda path: int(re.search(r"seed_(\d+)", path.stem).group(1)),
    )
    semillas_disponibles = int(resultados["mlp"]["n_semillas"])
    if len(modelos_paths) != semillas_disponibles:
        raise FileNotFoundError(
            f"Se esperaban {semillas_disponibles} modelos MLP de LGD y se encontraron "
            f"{len(modelos_paths)} en {LGD_MODELS}."
        )
    modelos_paths = modelos_paths[:parametros["semillas_lgd"]]

    predicciones_semillas = []
    metricas_semillas = []
    for model_path in modelos_paths:
        modelo = keras.models.load_model(model_path, compile=False)
        pred = np.clip(
            modelo.predict(X_test, batch_size=parametros["batch_size"], verbose=0).reshape(-1),
            0, 1,
        )
        predicciones_semillas.append(pred)
        pred_observada = pred[test["lgd_ok"].to_numpy()]
        metricas_semillas.append({
            "mae": float(mean_absolute_error(test_observada, pred_observada)),
            "rmse": float(np.sqrt(mean_squared_error(test_observada, pred_observada))),
        })
        del modelo

    pred_mlp = np.mean(np.stack(predicciones_semillas), axis=0)
    pred_media = np.full(len(test), float(train_observada.mean()))

    def metricas_lgd(pred):
        evaluacion = pred[test["lgd_ok"].to_numpy()]
        return {
            "mae": float(mean_absolute_error(test_observada, evaluacion)),
            "rmse": float(np.sqrt(mean_squared_error(test_observada, evaluacion))),
        }

    metricas = {
        "Referencia constante": metricas_lgd(pred_media),
        "Regresión lineal": metricas_lgd(pred_lineal),
        "MLP ensamble": metricas_lgd(pred_mlp),
    }
    detalle_semillas = {}
    for nombre in ("mae", "rmse"):
        valores = [fila[nombre] for fila in metricas_semillas]
        detalle_semillas[nombre] = {
            "media": float(np.mean(valores)),
            "desv_est": float(np.std(valores, ddof=1)),
            "mejor": float(np.min(valores)),
            "peor": float(np.max(valores)),
        }

    resultados["modelo_referencia"]["test"] = metricas["Regresión lineal"]
    resultados["referencia_media_constante_test"] = metricas["Referencia constante"]
    resultados["mlp"]["test_por_semilla"] = detalle_semillas
    resultados["mlp"]["test_ensamble_semillas"] = metricas["MLP ensamble"]
    resultados["mlp"]["n_semillas_recalculadas"] = len(modelos_paths)
    guardar_json(metricas_path, resultados)

    predicciones_lgd = test[
        ["id", "split", "loan_status", "target", "funded_amnt", "EAD_obs", "LGD", "lgd_ok"]
    ].copy()
    predicciones_lgd["LGD_pred_lineal"] = pred_lineal
    predicciones_lgd["LGD_pred_mlp"] = pred_mlp
    LGD_OUTPUTS.mkdir(parents=True, exist_ok=True)
    predicciones_lgd.to_parquet(LGD_OUTPUTS / "lgd_predicciones_test.parquet", index=False)

    print(f"Casos en prueba: {len(test):,}; casos con LGD observada: {len(test_observada):,}")
    print(f"{'Modelo':<25} {'MAE':>10} {'RMSE':>10}")
    print("-" * 47)
    for nombre, valores in metricas.items():
        print(f"{nombre:<25} {valores['mae']:>10.4f} {valores['rmse']:>10.4f}")
    print(f"MLP: {len(modelos_paths)} de {semillas_disponibles} modelos guardados en el ensamble.")
    return predicciones_lgd


def calcular_perdida_esperada(predicciones_pd, predicciones_lgd):
    import numpy as np
    import pandas as pd

    titulo("PÉRDIDA ESPERADA: PD × LGD × MONTO FINANCIADO")
    base = predicciones_lgd[
        ["id", "split", "loan_status", "target", "funded_amnt", "LGD",
         "LGD_pred_lineal", "LGD_pred_mlp"]
    ].copy()
    if base["id"].duplicated().any():
        raise ValueError("El conjunto de prueba LGD tiene identificadores duplicados.")

    base = base.rename(columns={"LGD": "LGD_observada", "funded_amnt": "EAD_origin"})
    if not np.isfinite(base["EAD_origin"].to_numpy(dtype=float)).all():
        raise ValueError("El monto financiado contiene valores no numéricos o infinitos.")

    for nombre, predicciones in (
        ("logistic", predicciones_pd["pd_logistic"]),
        ("mlp", predicciones_pd["pd_mlp"]),
    ):
        pd_test = predicciones.loc[predicciones["split"] == "test", ["id", "pd"]]
        if pd_test["id"].duplicated().any():
            raise ValueError(f"Las predicciones PD de {nombre} tienen ids duplicados.")
        base = base.merge(pd_test, on="id", how="left", validate="one_to_one")
        base = base.rename(columns={"pd": f"PD_{nombre}"})
        if base[f"PD_{nombre}"].isna().any():
            raise ValueError(f"Faltan PD alineadas por id para el modelo {nombre}.")
        for modelo_lgd in ("lineal", "mlp"):
            base[f"EL_proxy_{nombre}_{modelo_lgd}"] = (
                base[f"PD_{nombre}"] * base[f"LGD_pred_{modelo_lgd}"] * base["EAD_origin"]
            )

    salida = LGD_OUTPUTS / "perdida_esperada_test.parquet"
    base.to_parquet(salida, index=False)
    resumen = base.filter(regex=r"^EL_proxy_").agg(["sum", "mean"]).T
    resumen.columns = ["Total test", "Media por crédito"]
    print(resumen.round(2).to_string())
    print(f"\nRegistros guardados: {salida}")
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Recalcula resultados con modelos ya entrenados; no ajusta modelos."
    )
    parser.add_argument("--solo", choices=["parte1", "parte2"],
                        help="limita el cálculo a PD o a LGD")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    partes = {"parte1", "parte2"} if args.solo is None else {args.solo}
    parametros = configurar_calculos(partes)
    inicio = time.perf_counter()
    predicciones_pd = None
    predicciones_lgd = None
    if args.solo != "parte2":
        predicciones_pd = recalcular_parte1(parametros)
    if args.solo != "parte1":
        if predicciones_pd is None:
            exigir_archivos([
                PD_RESULTS / "pd_logistic_predicciones.parquet",
                PD_RESULTS / "pd_mlp_predicciones.parquet",
            ])
            import pandas as pd

            predicciones_pd = {
                "pd_logistic": pd.read_parquet(PD_RESULTS / "pd_logistic_predicciones.parquet"),
                "pd_mlp": pd.read_parquet(PD_RESULTS / "pd_mlp_predicciones.parquet"),
            }
        predicciones_lgd = recalcular_parte2(predicciones_pd, parametros)
        calcular_perdida_esperada(predicciones_pd, predicciones_lgd)

    duracion = time.perf_counter() - inicio
    titulo(f"RECÁLCULO COMPLETADO EN {duracion:.1f} SEGUNDOS")
    if duracion >= 120:
        print("Aviso: la ejecución superó el objetivo de 2 minutos.")
    print("No se entrenaron modelos ni se buscaron hiperparámetros.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
