"""Resumen legible de la Parte 1: parte1/outputs/results/pd_resumen.md.

Reúne en un solo Markdown lo que está repartido en los JSON de resultados
(datos, hipótesis, comparación, umbrales, SHAP, scorecard y búsqueda), para
revisarlo sin abrir archivos de máquina. Se regenera en cada corrida.
"""
import json

from parte1.common.utils import RESULTS_DIR, read_analysis


# Tabla Markdown a partir de una lista de dicts, con columnas y formato elegidos.
def md_table(rows: list[dict], columns: dict[str, str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(columns.values()) + " |\n|" + "---|" * len(columns) + "\n"
    body = ""
    for r in rows:
        cells = []
        for key in columns:
            v = r.get(key, "")
            cells.append(format(v, fmt[key]) if key in fmt and isinstance(v, (int, float)) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


# Lee el JSON de métricas de un modelo.
def _model(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}_metricas.json").read_text(encoding="utf-8"))


# Arma el Markdown completo y lo guarda en outputs/results/pd_resumen.md.
def run_report() -> str:
    datos, lr, mlp = read_analysis("datos"), _model("pd_logistic"), _model("pd_mlp")
    eda = {h["id"]: h for h in read_analysis("eda_hipotesis")}
    contrast = read_analysis("hipotesis_vs_modelos")
    comp, sc = read_analysis("comparacion"), read_analysis("scorecard")
    n_seeds = mlp["semillas"]["n"]

    hyp_rows = [{**c, "auc": eda[c["id"]]["auc_univariado"], "p": eda[c["id"]]["p_valor"],
                 "tasas": f"{100 * c['eda_tasa_grupo_bajo']:.1f} % → {100 * c['eda_tasa_grupo_alto']:.1f} %"}
                for c in contrast]
    seeds = {r["metrica"]: r for r in mlp["semillas"]["resumen_test"]}
    metric_rows = [{**r, "mlp_peor": seeds[r["metrica"]]["peor"], "mlp_mejor": seeds[r["metrica"]]["mejor"]}
                   for r in comp["metricas_test"]]
    boot = comp["bootstrap"]
    clas = [r for r in read_analysis("clasificacion") if r["split"] == "test"]
    p = sc["parametros"]

    md = f"""# Resumen de resultados, Parte 1 (generado automáticamente)

No editar a mano: lo reescribe `python run.py`. Las conclusiones redactadas están en `parte1/docs/eda_y_conclusiones.md`. Las rutas de figuras son relativas a `parte1/outputs/`.

## 1. Datos

- Registros: {datos['n_registros']:,}; variables guardadas: {datos['n_variables_guardadas']}; originación {datos['periodo_originacion'][0]} a {datos['periodo_originacion'][1]}; descarga de Kaggle: {datos['fecha_descarga']}.
- Partición temporal: entrenamiento {datos['particiones']['train']:,} ({100 * datos['tasa_malos']['train']:.1f} % malos), validación {datos['particiones']['val']:,} ({100 * datos['tasa_malos']['val']:.1f} %), prueba {datos['particiones']['test']:,} ({100 * datos['tasa_malos']['test']:.1f} %).

Calidad de las variables numéricas en entrenamiento (antes del preprocesamiento):

{md_table(read_analysis("calidad_datos"), {"variable": "Variable", "faltantes": "Faltantes", "ceros": "Ceros",
                                           "negativos": "Negativos", "minimo": "Mínimo", "p1": "P1", "mediana": "Mediana",
                                           "p99": "P99", "maximo": "Máximo"},
          {"faltantes": ".1%", "ceros": ".1%", "minimo": ",.4g", "p1": ",.4g", "mediana": ",.4g", "p99": ",.4g", "maximo": ",.4g"})}
## 2. Hipótesis: EDA univariado y efecto controlado en los modelos

EDA en entrenamiento (Mann-Whitney, α = 0,01, efecto mínimo). Δ PD: cambio medio de la PD al mover la variable de un valor bajo a uno alto dejando las demás fijas (pp = puntos porcentuales). Figura: `figures/pd_eda_hipotesis.png`.

{md_table(hyp_rows, {"id": "Id", "hipotesis": "Hipótesis", "tasas": "Tasa de malos bajo → alto", "auc": "AUC univ.",
                     "veredicto_eda": "EDA", "escenario": "Escenario", "delta_pd_lr_pp": "Δ PD LR (pp)",
                     "delta_pd_mlp_pp": "Δ PD MLP (pp)", "veredicto_lr": "LR", "veredicto_mlp": "MLP",
                     "mejor_rango_shap_mlp": "Rango SHAP"},
          {"auc": ".3f", "delta_pd_lr_pp": "+.2f", "delta_pd_mlp_pp": "+.2f"})}
## 3. Logística contra MLP en prueba

MLP: media y desviación de {n_seeds} semillas; peor y mejor semilla. Figuras: `figures/pd_*_roc.png`, `figures/pd_*_calibracion.png`, `figures/pd_mlp_perdidas.png`.

{md_table(metric_rows, {"metrica": "Métrica", "referencia_tasa_base": "Tasa base", "logistica": "Logística",
                        "mlp_media_semillas": "MLP media", "mlp_std_semillas": "MLP d.e.", "mlp_peor": "MLP peor",
                        "mlp_mejor": "MLP mejor", "mlp_modelo_guardado": "MLP guardado"},
          {k: ".4f" for k in ("referencia_tasa_base", "logistica", "mlp_media_semillas", "mlp_std_semillas",
                              "mlp_peor", "mlp_mejor", "mlp_modelo_guardado")})}
Bootstrap pareado en prueba ({boot['n_boot']} réplicas), MLP guardado menos logística: ΔAUC {boot['dif_auc_mlp_menos_lr']['media']:+.4f} (IC 95 % {boot['dif_auc_mlp_menos_lr']['ic95'][0]:+.4f} a {boot['dif_auc_mlp_menos_lr']['ic95'][1]:+.4f}); ΔBrier {boot['dif_brier_mlp_menos_lr']['media']:+.5f} (IC 95 % {boot['dif_brier_mlp_menos_lr']['ic95'][0]:+.5f} a {boot['dif_brier_mlp_menos_lr']['ic95'][1]:+.5f}).

{md_table(comp["costos"], {"concepto": "Costo", "logistica": "Logística", "mlp": "MLP"},
          {"logistica": ",.5g", "mlp": ",.5g"})}
## 4. Accuracy, F1 y matriz de confusión (prueba)

Umbral 0,5 y umbral que maximiza F1 en validación. Figuras: `figures/pd_matriz_confusion.png` y `figures/pd_distribucion_por_clase.png`.

{md_table(clas, {"modelo": "Modelo", "criterio_umbral": "Umbral", "umbral": "Valor", "accuracy": "Accuracy",
                 "precision": "Precisión", "sensibilidad": "Sensibilidad", "especificidad": "Especificidad",
                 "f1": "F1", "tasa_rechazo": "Rechazados", "vn": "VN", "fp": "FP", "fn": "FN", "vp": "VP"},
          {k: ".3f" for k in ("umbral", "accuracy", "precision", "sensibilidad", "especificidad", "f1", "tasa_rechazo")})}
## 5. Variables de riesgo (SHAP, |valor| medio sobre el logit)

Figura: `figures/pd_mlp_shap.png`.

{md_table(read_analysis("shap_importancia"), {"variable": "Variable", "mlp_shap_abs": "MLP", "rango_mlp": "Rango MLP",
                                              "lr_shap_abs": "Logística", "rango_lr": "Rango LR"},
          {"mlp_shap_abs": ".3f", "lr_shap_abs": ".3f"})}
## 6. Scorecard (desde la logística)

Puntaje base {p['base_score']:g} con odds {p['base_odds']:g}:1 y PDO {p['pdo']:g}: score = {p['offset']:.2f} + {p['factor']:.2f} · ln(odds buenos/malos). AUC en prueba con el puntaje: {sc['auc_test_con_puntaje']:.4f}.

{md_table(sc["bandas_test"], {"banda": "Banda (1 = más riesgo)", "n": "Créditos", "score_min": "Puntaje mín.",
                               "score_max": "Puntaje máx.", "pd_media": "PD media", "tasa_malos": "Tasa observada"},
          {"score_min": ".1f", "score_max": ".1f", "pd_media": ".3f", "tasa_malos": ".3f"})}
## 7. MLP: búsqueda y configuración elegida

Método: {mlp['busqueda']['metodo']} ({mlp['busqueda']['n_configuraciones']} configuraciones, {mlp['busqueda']['segundos']:.0f} s); {n_seeds} semillas ({mlp['semillas']['segundos']:.0f} s). Parámetros entrenables: {mlp['n_parametros']:,}. Configuración elegida: `{json.dumps(mlp['mejor_config'])}`. Logística: C = {lr['busqueda']['mejor_C']:g} entre {lr['busqueda']['n_configuraciones']} valores.

{md_table(mlp['busqueda'].get('configuraciones', [])[:5],
          {"hidden": "Capas", "activation": "Activación", "dropout": "Dropout", "l2": "L2", "optimizer": "Optimizador",
           "lr": "Tasa", "batch_size": "Lote", "mejor_epoca": "Mejor época", "val_log_loss": "Log-loss val.", "val_auc": "AUC val."},
          {"l2": ".1e", "lr": ".1e", "val_log_loss": ".4f", "val_auc": ".4f"})}"""
    path = RESULTS_DIR / "pd_resumen.md"
    path.write_text(md, encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    print(run_report())
