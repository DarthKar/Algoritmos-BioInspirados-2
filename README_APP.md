# Simulador web de riesgo crediticio

## Arranque local

Se recomienda Python 3.12 o 3.13. Desde la raíz del repositorio:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements_app.txt
streamlit run app.py
```

El modelo se carga desde `models/pd_logistic.joblib`. **No se reentrena**. El objeto `joblib` y el preprocesamiento deben provenir de código confiable; no cargues archivos desconocidos. Versiones del entorno de despliegue deben coincidir con las usadas para entrenar/serializar los modelos.

## Publicar

1. Subir el repositorio con `app.py`, `app_config.json`, `requirements-app.txt`, `models/`, `src/`, `results/` y `data/processed/`.
2. En Streamlit Community Cloud, conectar GitHub y seleccionar `web/app.py` como archivo principal.
3. La nube usará `web/requirements.txt` (al estar junto al punto de entrada) en vez de `requirements.txt` de entrenamiento. Configurar Python 3.13 si el entorno lo permite.
4. Actualizar `report_url` y `video_url` dentro de `app_config.json` con enlaces HTTPS públicos reales.
5. Comprobar que un usuario externo pueda abrir el enlace y predecir; después probar en clon limpio.

## Agregar o cambiar modelos

Registrar otra entrada en `app_config.json` bajo `models`, apuntando a un **pipeline sklearn serializado** que admita `predict_proba(DataFrame)` y use clase 1 para incumplimiento. El preprocesador debe llamarse `preprocessor` y formar parte del pipeline, o el estimador debe exponer `feature_names_in_`. Por ejemplo:

```json
{
  "name": "Red neuronal MLP",
  "path": "models/pd_mlp_pipeline.joblib",
  "data_path": "data/processed/pd_matured.parquet",
  "predictions_path": "results/pd_mlp_predicciones.parquet",
  "reference_split": "train"
}
```

**No basta con guardar pesos `.pt` de PyTorch:** deberán preparar y guardar un adaptador de inferencia confiable que aplique el preprocesamiento ajustado y exponga `predict_proba`, o ampliar el cargador para manejar el formato elegido. El conjunto de entrenamiento utilizado para la referencia debe corresponder al modelo, sin usar resultados de prueba para ajustar decisiones.

Los campos del formulario se descubren del preprocesador del modelo. Sus valores iniciales provienen de `data_path`; al cambiar columnas deben actualizar también el dataset o definir valores en `fallback_fields` para columnas que no estén allí. Validar el significado, unidad y valores permitidos de cada campo contra el diccionario real de Lending Club antes de entregar. El sistema NO recalcula variables derivadas a partir de otras: si cambian la ingeniería de características, deben adaptar la capa de entrada para preservar las mismas transformaciones del entrenamiento.

La scorecard se calcula con las odds `P(bueno)/P(malo)` y `PDO` configurable. El percentil corresponde a la partición de referencia configurada, no a toda la población. El simulador es descriptivo, no causal.

## Verificaciones para la sustentación

- Ejecutar `streamlit run app.py` con un solo comando, después de instalar dependencias.
- Verificar que el percentil y la probabilidad sean coherentes con el artefacto guardado.
- Cambiar parámetros de scorecard sin modificar código y comprobar el nuevo puntaje.
- Comparar un escenario modificado.
- Verificar enlaces al blog y al video y el despliegue público.