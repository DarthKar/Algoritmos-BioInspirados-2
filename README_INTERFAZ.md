# Interfaz mejorada — ¿Me prestas?

## Ejecutar

Desde la carpeta principal del proyecto, con el entorno virtual activado:

```powershell
python -m pip install -r requirements-app.txt
streamlit run app.py
```

Para Streamlit Community Cloud, archivo principal: `web/app.py` y dependencias `web/requirements.txt`. Configura en `app_config.json` los enlaces `report_url` y `video_url` cuando estén disponibles.

## Cambios de interfaz

- Formulario agrupado por tipo de información con descripciones, valores de referencia y equivalencias de categorías.
- Resultados con probabilidad de incumplimiento, puntaje y percentil, más histograma y marcador del perfil.
- Comparación de la regresión logística y de la MLP para los **mismos valores** de entrada.
- Simulación de cambios con deltas de probabilidad, puntaje y percentil de cada modelo.
- Unión dinámica de columnas requeridas por los modelos presentes en `app_config.json`. Si se agrega una nueva columna, se muestra automáticamente; compleméntala en `labels` y `FIELD_GUIDANCE` para que sea comprensible.
- Escala scorecard configurable. Si falta el histórico de algún modelo, se deja el percentil como no disponible.

## Notas de interpretación

- Los percentiles se calculan frente a las predicciones del conjunto `train` de **cada modelo**, usando la proporción con puntaje menor o igual. No son percentiles nacionales.
- El puntaje es una transformación monotónica de PD, no una scorecard aditiva por variable de la MLP.
- Las comparaciones de escenarios no establecen causalidad.
- Las descripciones son orientativas; antes de publicar, contrasta cada definición con el diccionario de variables de Kaggle.
- `sub_grade_num` es una codificación de una calificación propia de Lending Club que puede no estar disponible a un solicitante en fase temprana. Acuerda el caso de uso con el equipo.
- El formulario no envía ni almacena perfiles en base de datos; Streamlit mantiene datos en estado de sesión mientras se usa la app.

## Verificaciones previas

```powershell
python scripts/verificar_integracion_web.py
python scripts/verificar_predicciones_historicas.py
python scripts/verificar_scorecard_percentiles.py
```

Los scripts conservan el cálculo original para contraste; la actualización de interfaz no cambia `scorecard.py`, `adapter.py` ni los artefactos de los modelos.