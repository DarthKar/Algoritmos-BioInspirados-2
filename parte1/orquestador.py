"""Orquestador interactivo de la Parte 1: todo lo de run.py, por menús en la consola.

Desde la raíz del repositorio:

    python parte1/orquestador.py

Permite correr el demo, el experimento completo o etapas sueltas, cambiar
parámetros para la sesión, ver los resultados en tablas (incluidos los mejores
hiperparámetros de la red), abrir la app web, verificarla y abrir las figuras.
Las etapas se ejecutan con run.py en un proceso aparte, así que su salida se ve
en vivo en la consola.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PARTE1 = Path(__file__).resolve().parent
REPO = PARTE1.parent
sys.path.insert(0, str(REPO))
from parte1.run import STAGES, load_config  # noqa: E402

RUN = PARTE1 / "run.py"
CONFIG = PARTE1 / "config.yaml"
SALIDAS = {"oficiales": PARTE1 / "salidas", "demo": PARTE1 / "salidas_demo"}
STAGE_HELP = {
    "datos": "reconstruye el dataset desde data/raw/loan.csv",
    "eda": "contrasta las 13 hipótesis y genera las figuras del EDA",
    "logistica": "entrena la regresión logística (rejilla de C)",
    "mlp": "búsqueda aleatoria del MLP y entrenamiento con varias semillas",
    "figuras": "curvas ROC, de calibración y de pérdida",
    "interpretabilidad": "SHAP sobre el MLP y la logística",
    "scorecard": "scorecard del MLP (puntos por variable con SHAP) y de la logística",
    "clasificacion": "accuracy, F1 y matriz de confusión",
    "hipotesis": "efecto de cada hipótesis en los modelos",
    "comparacion": "logística contra MLP: métricas, costos y bootstrap",
    "informe": "genera salidas/resultados/pd_resumen.md",
}
overrides: list[str] = []   # cambios de parámetros de esta sesión (equivalen a --set)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


# --------------------------------------------------------------------------- #
# Utilidades de consola
# --------------------------------------------------------------------------- #
# Título de sección con una línea debajo.
def titulo(texto: str) -> None:
    print(f"\n{texto}\n{'=' * len(texto)}")


# Pide una opción hasta que sea válida; Enter devuelve el valor por defecto.
def elegir(prompt: str, validas: list[str], defecto: str | None = None) -> str:
    while True:
        r = input(f"{prompt} ").strip().lower()
        if not r and defecto is not None:
            return defecto
        if r in validas:
            return r
        print(f"  Opción no válida. Escribe una de: {', '.join(validas)}")


# Pregunta sí/no.
def confirmar(prompt: str) -> bool:
    return elegir(f"{prompt} (s/n)", ["s", "n", "si", "sí", "no"]) in ("s", "si", "sí")


# Muestra un menú numerado y devuelve la clave elegida.
def menu(titulo_menu: str, opciones: list[tuple[str, str]]) -> str:
    titulo(titulo_menu)
    for clave, texto in opciones:
        print(f"  {clave}. {texto}")
    return elegir("\nOpción:", [c for c, _ in opciones])


# Lee un JSON de resultados; None si todavía no existe.
def leer_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- #
# Ejecutar etapas
# --------------------------------------------------------------------------- #
# Lanza run.py con las etapas y los cambios de la sesión; la salida se ve en vivo.
def ejecutar(etapas: list[str] | None = None, demo: bool = False) -> None:
    cmd = [sys.executable, str(RUN)] + (["--demo"] if demo else []) + (["--etapas", *etapas] if etapas else [])
    if overrides:
        cmd += ["--set", *overrides]
    print("\nComando equivalente:", "python parte1/run.py", " ".join(cmd[2:]), flush=True)
    inicio = time.perf_counter()
    try:
        codigo = subprocess.run(cmd, cwd=REPO).returncode
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario.")
        return
    estado = "terminó bien" if codigo == 0 else f"falló (código {codigo}); revisa el mensaje de arriba"
    print(f"\nLa ejecución {estado}. Tiempo: {time.perf_counter() - inicio:.0f} s.")


# Deja elegir una o varias etapas por número y las corre.
def elegir_etapas() -> None:
    titulo("Etapas disponibles (se corren en este orden)")
    for i, s in enumerate(STAGES, 1):
        print(f"  {i:2d}. {s:<18} {STAGE_HELP[s]}")
    r = input("\nNúmeros separados por espacio (p. ej. 7 11), o Enter para volver: ").split()
    try:
        etapas = [STAGES[int(x) - 1] for x in r]
    except (ValueError, IndexError):
        print("  Números no válidos.")
        return
    if etapas:
        demo = confirmar("¿En modo demo (salidas en parte1/salidas_demo)?")
        ejecutar(sorted(set(etapas), key=STAGES.index), demo=demo)


# --------------------------------------------------------------------------- #
# Parámetros
# --------------------------------------------------------------------------- #
EJEMPLOS = [
    ("scorecard.pdo=40", "duplicar las odds suma 40 puntos en vez de 20"),
    ("scorecard.base_score=700", "puntaje base 700"),
    ("mlp.n_seeds=5", "entrenar la red final con 5 semillas"),
    ("mlp.n_trials=50", "probar 50 configuraciones en la búsqueda"),
    ("mlp.fixed.hidden=[32]", "red fija de una capa con 32 neuronas (salta la búsqueda)"),
    ("mlp.fixed.lr=0.01", "tasa de aprendizaje 0,01 en la red fija"),
    ("datos.plazos=[36]", "volver a usar solo créditos a 36 meses (requiere la etapa datos)"),
]


# Muestra la configuración efectiva y permite cambiar valores para la sesión.
def parametros() -> None:
    while True:
        cfg = load_config(CONFIG, False, overrides)
        titulo("Configuración efectiva (config.yaml + cambios de esta sesión)")
        print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=None))
        print("Cambios de esta sesión:", overrides or "ninguno")
        op = menu("Parámetros", [("1", "Cambiar un valor"), ("2", "Usar un ejemplo"),
                                 ("3", "Borrar los cambios de la sesión"), ("0", "Volver")])
        if op == "0":
            return
        if op == "3":
            overrides.clear()
        elif op == "1":
            print("Escribe clave=valor, p. ej. mlp.n_seeds=5 o mlp.fixed.hidden=[64,32] (Enter para cancelar).")
            nuevo = input("> ").strip()
            if "=" in nuevo:
                overrides.append(nuevo)
        elif op == "2":
            for i, (cambio, texto) in enumerate(EJEMPLOS, 1):
                print(f"  {i}. {cambio:<28} {texto}")
            r = elegir("Número:", [str(i) for i in range(1, len(EJEMPLOS) + 1)] + [""], defecto="")
            if r:
                overrides.append(EJEMPLOS[int(r) - 1][0])


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
# Carpeta de resultados a mostrar: los oficiales o los del demo.
def elegir_salida() -> Path | None:
    disponibles = [k for k, p in SALIDAS.items() if (p / "resultados" / "pd_mlp_metricas.json").exists()]
    if not disponibles:
        print("\nTodavía no hay resultados. Corre el demo o el experimento completo primero.")
        return None
    if len(disponibles) == 1:
        return SALIDAS[disponibles[0]] / "resultados"
    r = elegir("\n¿Resultados oficiales (o) o del demo (d)? [o]", ["o", "d"], defecto="o")
    return SALIDAS["oficiales" if r == "o" else "demo"] / "resultados"


# Métricas de prueba de ambos modelos y la diferencia por bootstrap.
def ver_metricas(res: Path) -> None:
    comp = leer_json(res / "pd_analisis.json")["comparacion"]
    tabla = pd.DataFrame(comp["metricas_test"]).set_index("metrica")
    tabla = tabla[["referencia_tasa_base", "logistica", "mlp_media_semillas", "mlp_std_semillas", "mlp_modelo_guardado"]]
    tabla.columns = ["Tasa base", "Logística", "MLP media", "MLP d.e.", "MLP guardado"]
    titulo("Métricas en prueba")
    print(tabla.round(4).to_string())
    b = comp["bootstrap"]
    lo, hi = b["dif_auc_mlp_menos_lr"]["ic95"]
    print(f"\nBootstrap pareado (MLP guardado − logística): ΔAUC {b['dif_auc_mlp_menos_lr']['media']:+.4f}, "
          f"IC 95 % [{lo:+.4f}; {hi:+.4f}]. {'Significativa' if lo > 0 or hi < 0 else 'No significativa'}.")
    titulo("Costos")
    costos = pd.DataFrame(comp["costos"]).set_index("concepto")
    print(costos.map(lambda v: f"{v:,.1f}" if isinstance(v, float) else v).to_string())


# Configuración elegida del MLP, ranking de la búsqueda y rejilla de la logística.
def ver_hiperparametros(res: Path) -> None:
    mlp, lr = leer_json(res / "pd_mlp_metricas.json"), leer_json(res / "pd_logistic_metricas.json")
    titulo("Mejores hiperparámetros del MLP")
    cfg = mlp["mejor_config"]
    print(f"Método: {mlp['busqueda']['metodo']} ({mlp['busqueda']['n_configuraciones']} configuraciones), "
          f"criterio: {mlp['busqueda']['criterio']}.")
    filas = {"Capas ocultas": cfg["hidden"], "Activación": cfg["activation"], "Dropout": cfg["dropout"],
             "Penalización L2 (λ)": f"{cfg['l2']:.2e}", "Optimizador": cfg["optimizer"], "Tasa de aprendizaje": f"{cfg['lr']:.2e}",
             "Momento (solo SGD)": cfg["momentum"], "Tamaño de lote": cfg["batch_size"], "Épocas máximas": cfg["max_epochs"],
             "Paciencia": cfg["patience"], "Semilla guardada": cfg["seed"], "Parámetros entrenables": f"{mlp['n_parametros']:,}"}
    for k, v in filas.items():
        print(f"  {k:<24} {v}")
    confs = pd.DataFrame(mlp["busqueda"].get("configuraciones", []))
    if not confs.empty:
        titulo("Ranking de la búsqueda (menor log-loss en validación primero)")
        cols = ["hidden", "activation", "dropout", "l2", "optimizer", "lr", "batch_size", "n_parametros", "mejor_epoca", "val_log_loss", "val_auc"]
        top = confs[cols].head(10).copy()
        top["l2"], top["lr"] = top["l2"].map("{:.1e}".format), top["lr"].map("{:.1e}".format)
        print(top.round(4).to_string(index=False))
    titulo("Regresión logística: búsqueda de C")
    print(pd.DataFrame(lr["busqueda"]["resultados"]).round(5).to_string(index=False))
    print(f"\nC elegido: {lr['busqueda']['mejor_C']}")


# Hipótesis: EDA univariado frente al efecto controlado en los modelos.
def ver_hipotesis(res: Path) -> None:
    a = leer_json(res / "pd_analisis.json")
    eda = {h["id"]: h for h in a["eda_hipotesis"]}
    filas = [{"Id": h["id"], "Hipótesis": eda[h["id"]]["hipotesis"][:60], "EDA": h["veredicto_eda"],
              "Δ PD LR (pp)": None if h["delta_pd_lr_pp"] is None else round(h["delta_pd_lr_pp"], 2),
              "Δ PD MLP (pp)": None if h["delta_pd_mlp_pp"] is None else round(h["delta_pd_mlp_pp"], 2),
              "LR": h["veredicto_lr"], "MLP": h["veredicto_mlp"]} for h in a["hipotesis_vs_modelos"]]
    titulo("Hipótesis: EDA frente a los modelos")
    print(pd.DataFrame(filas).to_string(index=False))


# Scorecard del MLP: escala, puntos por variable y tramo, y bandas; la logística como comparación.
def ver_scorecard(res: Path) -> None:
    a = leer_json(res / "pd_analisis.json")
    sc, sc_lr = a["scorecard"], a.get("scorecard_logistica", {})
    p = sc["parametros"]
    titulo("Scorecard del MLP")
    print(f"puntaje = {p['offset']:.2f} + {p['factor']:.2f} · ln(odds)   (base {p['base_score']:g}, odds {p['base_odds']:g}:1, PDO {p['pdo']:g})")
    print(f"Puntos base: {sc['puntos_base']:.1f}. AUC en prueba con el puntaje: MLP {sc['auc_test_con_puntaje']:.4f}"
          + (f", logística {sc_lr['auc_test_con_puntaje']:.4f}" if sc_lr else ""))
    print("Cada variable aporta −factor · φ puntos (φ = valor SHAP sobre el logit); los puntos de un cliente suman su puntaje.")
    print("\nPuntos medios absolutos por variable:")
    print(pd.DataFrame(sc["importancia_puntos"]).round(2).to_string(index=False))
    tramos = pd.DataFrame(sc["tabla_tramos"])
    top = [r["variable"] for r in sc["importancia_puntos"][:6]]
    print("\nPuntos medios del MLP por tramo (6 variables más importantes):")
    print(tramos[tramos["variable"].isin(top)].round(2).to_string(index=False))
    print("\nBandas en prueba con el puntaje del MLP (1 = más riesgo):")
    print(pd.DataFrame(sc["bandas_test"]).round(3).to_string(index=False))
    if sc_lr:
        titulo("Logística (comparación): puntos exactos por unidad de cada variable")
        print(pd.DataFrame(sc_lr["tabla_por_unidad"]).head(12).round(4).to_string(index=False))


# Métricas por umbral en prueba.
def ver_umbrales(res: Path) -> None:
    t = pd.DataFrame(leer_json(res / "pd_analisis.json")["clasificacion"]).query("split == 'test'")
    titulo("Accuracy, F1 y matriz de confusión en prueba")
    cols = ["modelo", "criterio_umbral", "umbral", "accuracy", "precision", "sensibilidad", "especificidad", "f1", "vn", "fp", "fn", "vp"]
    print(t[cols].round(3).to_string(index=False))


# Importancia de las variables por SHAP.
def ver_shap(res: Path) -> None:
    titulo("Variables de riesgo (media de |SHAP| sobre el logit)")
    print(pd.DataFrame(leer_json(res / "pd_analisis.json")["shap_importancia"]).round(3).to_string(index=False))


# Imprime el informe Markdown completo.
def ver_informe(res: Path) -> None:
    path = res / "pd_resumen.md"
    print(path.read_text(encoding="utf-8") if path.exists() else "No existe el informe; corre la etapa informe.")


# Submenú de resultados.
def resultados() -> None:
    vistas = {"1": ver_metricas, "2": ver_hiperparametros, "3": ver_hipotesis, "4": ver_scorecard,
              "5": ver_umbrales, "6": ver_shap, "7": ver_informe}
    while True:
        op = menu("Resultados", [("1", "Métricas: logística contra MLP"), ("2", "Mejores hiperparámetros de la red"),
                                 ("3", "Hipótesis: EDA frente a los modelos"), ("4", "Scorecard"),
                                 ("5", "Accuracy, F1 y matriz de confusión"), ("6", "Variables de riesgo (SHAP)"),
                                 ("7", "Informe completo"), ("0", "Volver")])
        if op == "0":
            return
        res = elegir_salida()
        if res is None:
            return
        try:
            vistas[op](res)
        except (KeyError, TypeError):
            print("Faltan resultados de alguna etapa. Corre el experimento completo o la etapa correspondiente.")
        input("\nEnter para continuar...")


# --------------------------------------------------------------------------- #
# Estado, app, verificación y figuras
# --------------------------------------------------------------------------- #
# Qué hay en disco: datos, modelos y resultados.
def estado() -> None:
    titulo("Estado de la Parte 1")
    raw = REPO / "data" / "raw" / "loan.csv"
    print(f"  loan.csv original:   {'sí' if raw.exists() else 'no (solo hace falta para la etapa datos)'}")
    rep = leer_json(PARTE1 / "datos" / "procesados" / "pd_matured_report.json")
    if rep:
        print(f"  Dataset procesado:   {rep['limpieza']['filas_salida']:,} créditos, plazos {rep['plazos']}")
    for nombre, carpeta in SALIDAS.items():
        m = carpeta / "resultados" / "pd_mlp_metricas.json"
        if m.exists():
            fecha = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.stat().st_mtime))
            mlp = leer_json(m)
            auc = next(r for r in mlp["semillas"]["resumen_test"] if r["metrica"] == "auc")
            print(f"  Resultados {nombre:<9} {fecha}; MLP {mlp['semillas']['n']} semillas, AUC test {auc['mean']:.4f}")
        else:
            print(f"  Resultados {nombre:<9} no hay")
    print(f"  Cambios de parámetros en esta sesión: {overrides or 'ninguno'}")


# Levanta la app web de Streamlit hasta que el usuario pulse Ctrl+C.
def app_web() -> None:
    print("\nAbriendo la app en el navegador (http://localhost:8501). Pulsa Ctrl+C aquí para cerrarla y volver al menú.")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=REPO)
    except KeyboardInterrupt:
        pass
    print("App cerrada.")


# Corre los scripts que comprueban que la app reproduce los resultados guardados.
def verificar() -> None:
    for script in ("verificar_predicciones_historicas.py", "verificar_scorecard_percentiles.py", "verificar_despliegue.py"):
        titulo(script)
        subprocess.run([sys.executable, str(REPO / "scripts" / script)], cwd=REPO)


# Lista las figuras y abre la que se elija con el visor del sistema.
def figuras() -> None:
    carpeta = SALIDAS["oficiales"] / "figuras"
    archivos = sorted(carpeta.glob("*.png"))
    titulo(f"Figuras en {carpeta.relative_to(REPO)}")
    for i, f in enumerate(archivos, 1):
        print(f"  {i:2d}. {f.name}")
    r = input("\nNúmero de la figura para abrirla (Enter para volver): ").strip()
    if r.isdigit() and 1 <= int(r) <= len(archivos):
        abrir(archivos[int(r) - 1])


# Abre un archivo con la aplicación predeterminada del sistema operativo.
def abrir(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 (solo Windows)
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


# --------------------------------------------------------------------------- #
# Menú principal
# --------------------------------------------------------------------------- #
# Bucle principal del orquestador.
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    acciones = {
        "1": estado,
        "2": lambda: ejecutar(demo=True),
        "3": lambda: confirmar("El experimento completo tarda entre 10 y 35 minutos y reemplaza los resultados oficiales. ¿Continuar?") and ejecutar(),
        "4": elegir_etapas, "5": parametros, "6": resultados, "7": app_web, "8": verificar, "9": figuras,
    }
    print("Parte 1: probabilidad de incumplimiento (Lending Club)")
    while True:
        try:
            op = menu("Menú principal", [
                ("1", "Ver el estado del proyecto"), ("2", "Correr la demostración (menos de 2 minutos)"),
                ("3", "Correr el experimento completo"), ("4", "Correr etapas específicas"),
                ("5", "Ver o cambiar parámetros"), ("6", "Ver resultados (métricas, hiperparámetros, scorecard...)"),
                ("7", "Abrir la aplicación web"), ("8", "Verificar la aplicación"),
                ("9", "Abrir una figura"), ("0", "Salir")])
            if op == "0":
                print("Hasta luego.")
                return
            acciones[op]()
        except KeyboardInterrupt:
            print("\n(Cancelado. Vuelves al menú principal.)")
        except EOFError:
            print()
            return


if __name__ == "__main__":
    main()
