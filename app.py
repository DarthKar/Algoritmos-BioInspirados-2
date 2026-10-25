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


def main():
    cfg = load_registry()
    st.title("📊 Simulador de riesgo crediticio")
    st.write("Explora cómo distintas características de un préstamo se relacionan con la probabilidad estimada de incumplimiento.")
    st.info("Herramienta educativa basada en préstamos históricos de Lending Club. No decide aprobaciones ni sustituye una evaluación crediticia profesional.")
    models = {m["name"]: m for m in cfg["models"] if (ROOT / m["path"]).is_file()}
    if not models:
        st.error("No se encontraron modelos guardados. Revisa app_config.json y la carpeta models/.")
        st.stop()
    with st.sidebar:
        st.header("Configuración")
        choice = st.selectbox("Modelo", list(models))
        mcfg = models[choice]
        scorecfg = cfg.get("scorecard", {})
        with st.expander("Parámetros de scorecard"):
            base_score = st.number_input("Puntaje base", value=float(scorecfg.get("base_score", 600)))
            base_odds = st.number_input("Odds buenos/malos de referencia", min_value=0.01, value=float(scorecfg.get("base_odds", 50)))
            pdo = st.number_input("PDO: puntos para duplicar odds", min_value=0.01, value=float(scorecfg.get("pdo", 20)))
        params = ScorecardParams(base_score, base_odds, pdo)
        st.caption("Un puntaje más alto representa un riesgo estimado menor.")
        if cfg.get("report_url"):
            st.link_button("📄 Reporte técnico", cfg["report_url"])
        else:
            st.caption("Reporte técnico: enlace pendiente")
        if cfg.get("video_url"):
            st.link_button("🎬 Video promocional", cfg["video_url"])
        else:
            st.caption("Video promocional: enlace pendiente")
    try:
        model = load_model(str(ROOT / mcfg["path"]))
        cols = predictors(model)
    except Exception as exc:
        st.error(f"No fue posible cargar el modelo {choice}: {exc}")
        st.stop()
    data_path = ROOT / mcfg.get("data_path", "") if mcfg.get("data_path") else None
    try:
        dataset = load_data(str(data_path)) if data_path and data_path.is_file() else None
        if dataset is not None and "split" in dataset:
            dataset = dataset[dataset["split"] == "train"]
    except Exception as exc:
        dataset = None
        st.warning(f"No se pudieron leer los datos para valores sugeridos: {exc}")
    fields = cfg.get("fallback_fields", {})
    labels = cfg.get("labels", {})
    metadata = {c: field_metadata(dataset, c, fields) for c in cols}
    st.subheader("1. Describe el perfil del préstamo")
    st.caption("Los valores iniciales son medianas o categorías frecuentes del entrenamiento, no datos personales reales. Los campos disponibles se detectan automáticamente en el modelo.")
    original = {}
    with st.form("client_data"):
        left, right = st.columns(2)
        for i, col in enumerate(cols):
            with (left if i % 2 == 0 else right):
                original[col] = widget(col, metadata[col], f"original_{col}_{choice}", labels)
        submitted = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)
    if not submitted and st.session_state.get("last_model") != choice:
        return
    if submitted:
        st.session_state["last_model"] = choice
        st.session_state["last_original"] = original.copy()
    original = st.session_state.get("last_original", original)
    try:
        p, score = inference(model, original, params)
    except Exception as exc:
        st.error(f"Error de predicción: {exc}")
        return
    st.subheader("2. Resultado del perfil")
    a, b, c = st.columns(3)
    a.metric("Probabilidad estimada de incumplimiento", f"{p:.2%}")
    b.metric("Puntaje scorecard", f"{score:,.0f}")
    ref_path = ROOT / mcfg.get("predictions_path", "results/none.parquet")
    split = mcfg.get("reference_split", "train")
    try:
        reference = reference_scores(ref_path, params, split)
    except Exception as exc:
        reference = None
        st.warning(f"No se pudo calcular la comparación poblacional: {exc}")
    if reference is not None:
        percentile = float(100 * np.mean(reference <= score))
        c.metric("Percentil de puntaje", f"{percentile:.1f}")
        st.caption(f"El puntaje es igual o superior al de aproximadamente {percentile:.1f}% de los casos de referencia ({split}, n={len(reference):,}). No equivale al percentil de toda la población colombiana.")
        st.bar_chart(pd.DataFrame({"Puntaje": reference}).assign(Tramo=lambda df: pd.cut(df['Puntaje'], bins=25)).groupby('Tramo', observed=True).size().reset_index(name='Frecuencia').assign(Tramo=lambda df: df['Tramo'].astype(str)).set_index('Tramo'))
    else:
        c.metric("Percentil de puntaje", "No disponible")
        st.caption("Para habilitar esta comparación, guarda predicciones con columnas pd y split en un archivo parquet.")
    st.subheader("3. ¿Qué pasaría si cambian las características?")
    st.caption("Compara escenarios hipotéticos; una variación del puntaje no demuestra causalidad ni garantiza mejores condiciones crediticias.")
    scenario = original.copy()
    with st.form("scenario_data"):
        candidates = st.multiselect("Variables que deseas modificar", cols, default=cols[:1], format_func=lambda x: labels.get(x, x))
        for col in candidates:
            scenario[col] = widget(col, {**metadata[col], "value": original[col]}, f"scenario_{col}_{choice}", labels)
        simulate = st.form_submit_button("Comparar escenarios", type="secondary", use_container_width=True)
    if simulate:
        try:
            p2, score2 = inference(model, scenario, params)
            l, r = st.columns(2)
            l.metric("Nueva probabilidad", f"{p2:.2%}", delta=f"{(p2-p)*100:+.2f} puntos porcentuales", delta_color="inverse")
            r.metric("Nuevo puntaje", f"{score2:,.0f}", delta=f"{score2-score:+.1f} puntos")
            changes = [{"Variable": labels.get(k, k), "Antes": original[k], "Después": v} for k,v in scenario.items() if v != original[k]]
            if changes: st.dataframe(pd.DataFrame(changes), hide_index=True, use_container_width=True)
            else: st.info("No modificaste ninguna característica.")
        except Exception as exc:
            st.error(f"Error al simular: {exc}")
    st.divider()
    st.caption(f"Modelo: {choice}. La aplicación carga artefactos preentrenados; no entrena al recibir solicitudes. No introduzcas datos personales identificables.")

if __name__ == "__main__":
    main()