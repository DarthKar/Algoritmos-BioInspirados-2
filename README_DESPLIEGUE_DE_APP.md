# Publicación de ¿Me prestas? en GitHub y Streamlit Community Cloud

## Archivos de la aplicación

- Entrada local: `app.py`
- Entrada cloud: `web/app.py` (carga la aplicación raíz)
- Dependencias cloud: `web/requirements.txt` (NO usar `requirements.txt` raíz para este despliegue)
- Dependencias locales web: `requirements-app.txt`
- Registro de modelos y enlaces: `app_config.json`
- Artefactos de inferencia: `parte1/salidas/modelos/`, `parte1/salidas/resultados/pd_*_predicciones.parquet`, `parte1/datos/procesados/pd_matured.parquet`

## Antes de subir archivos

1. Ejecuta `python scripts/verificar_despliegue.py` desde la raíz.
2. Comprueba que las versiones de Python y bibliotecas usadas para guardar los modelos son compatibles; usa Python 3.12 en Community Cloud inicialmente y prueba el despliegue. Si tus modelos se guardaron en otra versión, prioriza la misma versión utilizada en el entrenamiento.
3. **Revisa privacidad y licencia:** `parte1/datos/procesados/` incluye registros históricos y el repositorio puede hacerse público. No publiques datos personales, credenciales ni archivos originales sin comprobar que su distribución está autorizada. Para una publicación pública, preferir un conjunto de referencia anonimizado y mínimo; la versión incluida conserva los datos procesados porque el trabajo académico exige reproducibilidad y las pruebas los utilizan.
4. Revisa el histórico Git antes de publicar: un `.gitignore` no elimina datos que ya se hayan subido al repositorio.
5. Los enlaces al reporte técnico y al video están pendientes: configúralos como `report_url` y `video_url` en `app_config.json` cuando existan.

## Probar en Windows 11

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python -m pip check
python scripts/verificar_despliegue.py
streamlit run web/app.py
```

Para continuar usando el comando anterior del equipo, `streamlit run app.py` también es válido.

## Subir a GitHub

Este ZIP contiene los archivos del proyecto, pero no el historial `.git`. **Integra los cambios en el clon existente de tu equipo** y crea commits incrementales en vez de sustituir el repositorio por un único commit. Con GitHub Desktop: copia solo los archivos nuevos/modificados al clon, revisa Changes, escribe un mensaje descriptivo y usa Commit + Push. Comprueba en GitHub que se vean los modelos, los archivos de predicciones, `web/requirements.txt` y `web/app.py`.

## Desplegar en Community Cloud

1. Abre https://share.streamlit.io/ e inicia sesión con GitHub.
2. `Create app` > `Yup, I have an app`.
3. Selecciona el repositorio de GitHub, rama y **Entry point**: `web/app.py`.
4. En `Advanced settings`, selecciona la versión compatible de Python (se propone **3.12** para la primera prueba). No copies secretos al repositorio.
5. Pulsa `Deploy`, mira el registro de instalación y prueba ambos modelos, gráfica y simulador desde una ventana privada.
6. Si falla la instalación, consulta los logs del deployment. Si falla la carga del modelo, comprueba existencia de artefactos y versiones de scikit-learn/PyTorch.

**Nota:** Community Cloud elige `web/requirements.txt` porque está junto a `web/app.py`; esto evita instalar las dependencias de investigación de la raíz.

## Verificaciones posteriores

- Asegúrate de que los resultados del perfil de prueba conocido sigan siendo iguales (MLP: `22.463845 %` para el préstamo de validación manual previamente usado).
- Comprueba la comparación de ambos modelos, las gráficas y el simulador.
- Verifica el enlace público desde una ventana sin sesión de GitHub.
- Comprueba que el profesor puede lanzar la app localmente con `streamlit run app.py`.
- Documenta la URL definitiva en el README principal o en el reporte.

## Actualizaciones de los modelos

Sustituye de manera coordinada el checkpoint, el preprocesador y las predicciones históricas, actualiza `app_config.json` si cambia el nombre y ejecuta las pruebas de integración antes de subir el cambio. Si cambia el formato del checkpoint, modifica el adaptador.