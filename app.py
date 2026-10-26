"""Interfaz web de riesgo crediticio. Iniciar: streamlit run app.py."""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # Necesario para deserializar src.features._safe_log1p
from src.pd_model.scorecard import ScorecardParams, prob_to_score
from src.pd_model.adapter import load_registered_model

st.set_page_config(page_title="Simulador de riesgo crediticio", page_icon="📊", layout="wide")

@st.cache_resource(show_spinner="Cargando el modelo guardado…")
def load_model(path):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_registry():
    p = ROOT / "app_config.json"
    return json.loads(p.read_text(encoding="utf-8"))


def predictors(model):
    """Obtiene las entradas reales del preprocesador, no las columnas accesorias del dataset."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        pre = model.named_steps["preprocessor"]
        return list(dict.fromkeys(col for name, _, cols in pre.transformers_ if name != "remainder" and name != "drop" for col in cols))
    if hasattr(model, "input_columns"):
        return list(model.input_columns)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError("El modelo debe incluir un preprocesador sklearn con columnas identificables.")


def field_metadata(data, col, options):
    """Estadísticos del conjunto de entrenamiento o valores alternativos del archivo JSON."""
    settings = options.get(col, {})
    if data is not None and col in data:
        series = data[col].dropna()
        if len(series):
            if pd.api.types.is_numeric_dtype(series):
                return {"type": "numeric", "value": float(series.median()), "min": float(series.min()), "max": float(series.max())}
            values = series.astype(str).value_counts()
            return {"type": "category", "value": str(values.index[0]), "options": sorted(values.index.tolist())}
    if settings.get("type") == "category":
        return {"type": "category", "value": settings.get("default", ""), "options": settings.get("options", [""])}
    return {"type": "numeric", "value": float(settings.get("default", 0)), "min": float(settings.get("min", 0)), "max": float(settings.get("max", 1e9))}


def widget(col, meta, key, labels):
    label = labels.get(col, col.replace("_", " ").capitalize())
    if meta["type"] == "category":
        choices = list(meta["options"])
        default = choices.index(meta["value"]) if meta["value"] in choices else 0
        return st.selectbox(label, choices, index=default, key=key, help=f"Campo: {col}")
    minimum, maximum, value = meta["min"], meta["max"], meta["value"]
    value = float(np.clip(value, minimum, maximum))
    # No limitar artificialmente las modificaciones al máximo histórico.
    return st.number_input(label, value=value, key=key, help=f"Campo: {col}. Valor típico: {value:g}")


def inference(model, entry, params):
    p = float(model.predict_proba(pd.DataFrame([entry]))[0, 1])
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("El modelo devolvió una probabilidad inválida.")
    return p, float(prob_to_score([p], params)[0])


def reference_scores(pred_path, params, split):
    if not pred_path.exists():
        return None
    d = load_data(str(pred_path))
    if "pd" not in d or "split" not in d:
        return None
    subset = d[d["split"] == split]
    if subset.empty:
        return None
    ps = pd.to_numeric(subset["pd"], errors="coerce").dropna().to_numpy()
    if not len(ps):
        return None
    return prob_to_score(ps, params)



# Información orientativa de interfaz; las columnas del modelo siguen siendo la fuente
# de verdad. Las descripciones deben cotejarse con el diccionario del conjunto de datos.
FIELD_GUIDANCE = {
    "loan_amnt": ("Préstamo", "Cantidad solicitada en dólares estadounidenses."),
    "annual_inc": ("Ingresos y empleo", "Ingresos anuales declarados, en USD."),
    "emp_length": ("Ingresos y empleo", "Años de antigüedad laboral registrados."),
    "dti": ("Ingresos y empleo", "Relación deuda/ingresos expresada como porcentaje; utiliza la definición del conjunto Lending Club."),
    "home_ownership": ("Ingresos y empleo", "Situación de vivienda registrada en el crédito."),
    "verification_status": ("Ingresos y empleo", "Estado de verificación de ingresos registrado por la plataforma."),
    "purpose": ("Préstamo", "Finalidad declarada para solicitar el préstamo."),
    "sub_grade_num": ("Préstamo", "Codificación numérica de la subcalificación de Lending Club. No es una nota escolar ni una clasificación universal."),
    "delinq_2yrs": ("Historial crediticio", "Cantidad de eventos de morosidad registrados en los últimos dos años."),
    "inq_last_6mths": ("Historial crediticio", "Número de consultas de crédito recientes, dentro de seis meses."),
    "mths_since_last_delinq": ("Historial crediticio", "Meses desde la última morosidad registrada."),
    "never_delinquent": ("Historial crediticio", "Indicador del conjunto de datos: 1 significa sin moras anteriores y 0 lo contrario."),
    "open_acc": ("Historial crediticio", "Número de cuentas crediticias abiertas."),
    "total_acc": ("Historial crediticio", "Cantidad total de cuentas crediticias registradas."),
    "pub_rec": ("Historial crediticio", "Número de registros públicos desfavorables."),
    "credit_history_months": ("Historial crediticio", "Duración del historial crediticio expresada en meses."),
    "revol_util": ("Crédito rotativo", "Porcentaje utilizado del crédito rotativo disponible; puede superar 100 % en ciertos casos."),
    "revol_bal": ("Crédito rotativo", "Saldo vigente en cuentas de crédito rotativo, en USD."),
    "revol_limit": ("Crédito rotativo", "Variable derivada de límite rotativo. Mantén la transformación utilizada por el equipo."),
}
CATEGORY_NAMES = {
    "RENT": "Alquiler (RENT)", "OWN": "Propia (OWN)", "MORTGAGE": "Hipoteca (MORTGAGE)",
    "OTHER": "Otro (OTHER)", "NONE": "Ninguno (NONE)", "ANY": "Cualquiera (ANY)",
    "Not Verified": "No verificados (Not Verified)", "Verified": "Verificados (Verified)",
    "Source Verified": "Verificados en origen (Source Verified)",
    "debt_consolidation": "Consolidación de deudas", "credit_card": "Tarjeta de crédito",
    "home_improvement": "Mejoras del hogar", "major_purchase": "Compra importante",
    "small_business": "Pequeña empresa", "car": "Vehículo", "medical": "Gastos médicos",
    "moving": "Mudanza", "vacation": "Vacaciones", "wedding": "Boda",
    "house": "Vivienda", "renewable_energy": "Energía renovable", "other": "Otro"
}

def friendly_widget(col, meta, key, labels, initial=None):
    title = labels.get(col, col.replace("_", " ").capitalize())
    explanation = FIELD_GUIDANCE.get(col, ("Otras características", "Campo requerido por el modelo."))[1]
    if meta["type"] == "category":
        choices = list(meta["options"])
        val = initial if initial in choices else meta["value"]
        return st.selectbox(title, choices, index=choices.index(val) if val in choices else 0,
                            format_func=lambda x: CATEGORY_NAMES.get(str(x), str(x).replace("_", " ").capitalize()),
                            key=key, help=f"{explanation} Variable original: {col}")
    val = float(meta["value"] if initial is None else initial)
    # Preservar la posibilidad de valores fuera del intervalo de entrenamiento.
    # No aplicar redondeos que cambien la inferencia.
    if col == "never_delinquent":
        return st.selectbox(title, [0, 1], index=1 if val >= 0.5 else 0,
                            format_func=lambda x: "Sí" if x == 1 else "No", key=key,
                            help=f"{explanation} Variable original: {col}")
    return st.number_input(title, value=val, key=key,
                           help=f"{explanation} Variable original: {col}")


def percentile_for(score, ref):
    return float(100.0 * np.mean(np.asarray(ref) <= score))


def plot_distribution(ref, score, label):
    import altair as alt
    values = np.asarray(ref, dtype=float)
    chart_df = pd.DataFrame({"Puntaje": values})
    hist = alt.Chart(chart_df).mark_bar(opacity=0.72).encode(
        x=alt.X("Puntaje:Q", bin=alt.Bin(maxbins=32), title="Puntaje"),
        y=alt.Y("count():Q", title="Cantidad de préstamos"),
        tooltip=[alt.Tooltip("count():Q", title="Préstamos")]
    )
    position = alt.Chart(pd.DataFrame({"puntaje_usuario": [score]})).mark_rule(color="#E45756", size=3).encode(
        x="puntaje_usuario:Q", tooltip=[alt.Tooltip("puntaje_usuario:Q", title="Tu puntaje", format=".2f")]
    )
    st.altair_chart((hist + position).properties(height=260, title=label), use_container_width=True)


def main():
    cfg = load_registry()
    st.title("📊 ¿Me prestas? | Simulador de riesgo crediticio")
    st.write("Explora tu probabilidad estimada de incumplimiento, compárala con perfiles históricos y prueba cambios hipotéticos.")
    st.info("Proyecto educativo con préstamos históricos de Lending Club. No determina la aprobación de créditos ni constituye asesoría financiera.")
    available = {m["name"]: m for m in cfg["models"] if (ROOT / m["path"]).is_file()}
    if not available:
        st.error("No se encontraron modelos guardados. Verifica app_config.json y models/.")
        st.stop()

    loaded, model_columns = {}, {}
    for name, mc in available.items():
        try:
            m = load_registered_model(str(ROOT / mc["path"]), mc.get("type", "sklearn"),
                                      str(ROOT / mc["preprocessor_path"]) if mc.get("preprocessor_path") else None)
            loaded[name] = m
            model_columns[name] = predictors(m)
        except Exception as e:
            st.warning(f"{name} no está disponible: {e}")
    if not loaded:
        st.stop()

    with st.sidebar:
        st.header("Opciones")
        selected = st.selectbox("Modelo principal", list(loaded))
        scorecfg = cfg.get("scorecard", {})
        with st.expander("Ajustar escala de puntaje"):
            base_score = st.number_input("Puntaje base", value=float(scorecfg.get("base_score", 600)))
            base_odds = st.number_input("Odds buenos/malos de referencia", min_value=0.01, value=float(scorecfg.get("base_odds", 7)))
            pdo = st.number_input("PDO", min_value=0.01, value=float(scorecfg.get("pdo", 20)))
            st.caption("La escala cambia el puntaje, pero no la probabilidad estimada ni el orden de los percentiles.")
        st.divider()
        if cfg.get("report_url"):
            st.link_button("📄 Reporte técnico", cfg["report_url"])
        else:
            st.caption("📄 Reporte técnico: enlace pendiente")
        if cfg.get("video_url"):
            st.link_button("🎬 Video promocional", cfg["video_url"])
        else:
            st.caption("🎬 Video promocional: enlace pendiente")
        st.caption("Los perfiles ingresados no se guardan en una base de datos por esta aplicación.")

    params = ScorecardParams(base_score, base_odds, pdo)
    # Unión de columnas para evaluar exactamente el mismo perfil en ambos modelos.
    columns = list(dict.fromkeys(c for cols in model_columns.values() for c in cols))
    mc = available[selected]
    data_file = ROOT / mc["data_path"] if mc.get("data_path") else None
    data = None
    if data_file is not None and data_file.exists():
        try:
            data = load_data(str(data_file))
            if "split" in data:
                data = data.loc[data["split"] == "train"]
        except Exception as e:
            st.warning(f"No fue posible obtener los valores sugeridos: {e}")
    fields, labels = cfg.get("fallback_fields", {}), cfg.get("labels", {})
    metadata = {c: field_metadata(data, c, fields) for c in columns}
    groups = ["Préstamo", "Ingresos y empleo", "Historial crediticio", "Crédito rotativo", "Otras características"]
    tabs = st.tabs(["📝 Perfil", "📈 Resultados y comparación", "🔄 Simulador de cambios"])
    with tabs[0]:
        st.subheader("1. Características del préstamo")
        st.caption("Los valores iniciales reflejan la mediana o categoría más frecuente en entrenamiento. Los campos dependen de los modelos disponibles; no introduzcas información identificable.")
        with st.form("profile_form"):
            profile = {}
            for group in groups:
                members = [c for c in columns if FIELD_GUIDANCE.get(c, ("Otras características",))[0] == group]
                if not members:
                    continue
                st.markdown(f"**{group}**")
                left, right = st.columns(2)
                for i, c in enumerate(members):
                    with left if i % 2 == 0 else right:
                        profile[c] = friendly_widget(c, metadata[c], f"profile_{c}", labels)
            submitted = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)
        if submitted:
            st.session_state["risk_profile"] = dict(profile)
            st.session_state.pop("risk_scenario", None)
            st.success("Perfil actualizado. Abre la pestaña «Resultados y comparación».")

    current = st.session_state.get("risk_profile")
    if current is None:
        with tabs[1]:
            st.info("Completa el perfil y pulsa «Calcular riesgo» para ver los resultados.")
        with tabs[2]:
            st.info("Calcula primero el riesgo del perfil inicial para habilitar el simulador.")
        return
    # Rellenar nuevas columnas en una actualización sin perder el perfil anterior.
    current = {c: current.get(c, metadata[c]["value"]) for c in columns}

    def get_results(entry):
        output = {}
        for name, model in loaded.items():
            try:
                p, s = inference(model, {c: entry[c] for c in model_columns[name]}, params)
                refmc = available[name]
                ref_file = ROOT / refmc.get("predictions_path", "results/none.parquet")
                try:
                    ref = reference_scores(ref_file, params, refmc.get("reference_split", "train"))
                except Exception:
                    ref = None
                output[name] = {"pd": p, "score": s, "ref": ref,
                                "percentile": percentile_for(s, ref) if ref is not None else None,
                                "split": refmc.get("reference_split", "train")}
            except Exception as e:
                st.warning(f"No fue posible calcular {name}: {e}")
        return output

    results = get_results(current)
    with tabs[1]:
        if selected not in results:
            st.error("El modelo principal no pudo calcular resultados.")
        else:
            result = results[selected]
            st.subheader("2. Resultado principal")
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilidad de incumplimiento", f"{result['pd']:.2%}")
            c2.metric("Puntaje", f"{result['score']:.1f}")
            c3.metric("Percentil histórico", f"{result['percentile']:.1f}" if result['percentile'] is not None else "No disponible")
            st.caption("Un puntaje más alto corresponde a menor probabilidad de incumplimiento. El percentil mide la posición del puntaje frente a los préstamos de entrenamiento del mismo modelo, no frente a toda la población.")
            if result["ref"] is not None:
                plot_distribution(result["ref"], result["score"], f"Puntajes históricos: {selected}")
                st.caption(f"Línea roja: puntaje de este perfil. Referencia: {result['split']} (n={len(result['ref']):,}); percentil con empates incluidos (≤).")
            else:
                st.warning("No hay predicciones históricas para graficar percentiles.")
            st.subheader("Comparación para el mismo perfil")
            if len(results) > 1:
                rows = [{"Modelo": name, "Probabilidad de incumplimiento (%)": round(res["pd"]*100, 4),
                         "Puntaje": round(res["score"], 2),
                         "Percentil de su referencia": round(res["percentile"], 2) if res["percentile"] is not None else None}
                        for name, res in results.items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("Las probabilidades sí pueden compararse para el mismo perfil; cada percentil emplea una distribución histórica propia. Diferencias entre modelos no indican por sí mismas cuál es más preciso.")
            else:
                st.caption("Cuando se registre otro modelo compatible, aquí aparecerá su comparación.")

    with tabs[2]:
        st.subheader("3. Simular cambios")
        st.caption("Compara valores alternativos de las mismas características. Estas asociaciones no demuestran causalidad.")
        with st.form("scenario_form"):
            choices = st.multiselect("Características que deseas modificar", columns, default=[],
                                    format_func=lambda c: labels.get(c, c.replace("_", " ")))
            new_values = {}
            for c in choices:
                new_values[c] = friendly_widget(c, metadata[c], f"scenario_{c}", labels, initial=current[c])
            run_scenario = st.form_submit_button("Comparar escenarios", use_container_width=True)
        if run_scenario:
            st.session_state["risk_scenario"] = {**current, **new_values}
        scenario = st.session_state.get("risk_scenario")
        if scenario is not None:
            if set(scenario) != set(columns):
                scenario = {**current, **{c: v for c, v in scenario.items() if c in columns}}
            revised = get_results(scenario)
            if selected in revised and selected in results:
                a, b = results[selected], revised[selected]
                c1, c2, c3 = st.columns(3)
                c1.metric("Nueva probabilidad", f"{b['pd']:.2%}", delta=f"{(b['pd']-a['pd'])*100:+.2f} pp", delta_color="inverse")
                c2.metric("Nuevo puntaje", f"{b['score']:.1f}", delta=f"{b['score']-a['score']:+.1f}")
                c3.metric("Nuevo percentil", f"{b['percentile']:.1f}" if b['percentile'] is not None else "N/D",
                          delta=f"{b['percentile']-a['percentile']:+.1f} pp" if b['percentile'] is not None and a['percentile'] is not None else None)
                changed = [{"Variable": labels.get(c, c), "Original": current[c], "Escenario": scenario[c]}
                           for c in columns if current[c] != scenario[c]]
                if changed:
                    st.dataframe(pd.DataFrame(changed), hide_index=True, use_container_width=True)
                else:
                    st.info("El escenario no tiene cambios frente al perfil inicial.")
            if len(revised) > 1:
                comparison = [{"Modelo": n, "PD inicial (%)": round(results[n]["pd"]*100, 3),
                               "PD simulada (%)": round(r["pd"]*100, 3),
                               "Cambio (pp)": round((r["pd"]-results[n]["pd"])*100, 3),
                               "Cambio en puntaje": round(r["score"]-results[n]["score"], 2)}
                              for n, r in revised.items() if n in results]
                st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
    st.divider()
    st.caption("Los resultados se basan en modelos de investigación entrenados con datos históricos. Las variables de subcalificación de la plataforma pueden no estar disponibles antes de la aprobación de un préstamo; revisa el caso de uso del proyecto.")

if __name__ == "__main__":
    main()