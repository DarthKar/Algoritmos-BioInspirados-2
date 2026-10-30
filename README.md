# ¿Me prestas? Redes neuronales para riesgo de crédito

Trabajo 2 del curso *Redes neuronales y algoritmos bioinspirados* (Universidad Nacional de Colombia, Facultad de Minas, 2026-02). Usa créditos de consumo de Lending Club para responder dos preguntas:

- **Parte 1, probabilidad de incumplimiento (PD):** qué tan probable es que un crédito no se pague. Es un problema de clasificación binaria; se compara una regresión logística con un perceptrón multicapa (MLP) y se construye una scorecard.
- **Parte 2, pérdida dado incumplimiento (LGD):** qué fracción se pierde cuando un crédito no se paga. Es un problema de regresión. *En construcción*, en `parte2/`.

Hay además una aplicación web (Streamlit) donde una persona ingresa sus datos y obtiene su PD, su puntaje y su percentil.

**Enlaces:** reporte técnico (pendiente) · aplicación publicada (pendiente) · video promocional (pendiente)

## Contenido

1. [Instalación](#1-instalación)
2. [Datos](#2-datos)
3. [Ejecutar el experimento](#3-ejecutar-el-experimento)
4. [Cambiar parámetros](#4-cambiar-parámetros)
5. [Aplicación web](#5-aplicación-web)
6. [Resultados](#6-resultados)
7. [Reproducibilidad](#7-reproducibilidad)
8. [Estructura del repositorio](#8-estructura-del-repositorio)

## 1. Instalación

Se probó con Python 3.14.7 en Windows 11, solo con CPU (no se necesita GPU).

```bash
git clone https://github.com/DarthKar/Algoritmos-BioInspirados-2.git
cd Algoritmos-BioInspirados-2
python -m venv .venv
```

Activa el entorno: `.venv\Scripts\activate` en Windows o `source .venv/bin/activate` en Linux y macOS. Después instala las dependencias, que tienen versiones fijas en `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Eso basta para la Parte 1 y la app. La Parte 2 usa además TensorFlow, que se instala aparte:

```bash
python -m pip install -r parte2/requirements.txt
```

En Python 3.14 la única versión publicada de TensorFlow es una versión candidata (2.22.0rc0); por eso queda fijada en `parte2/requirements.txt`.

En Linux, `pip` instala por defecto la versión de PyTorch con CUDA, que pesa varios GB. Si no tienes GPU, instala antes la versión para CPU:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

## 2. Datos

**No necesitas descargar nada para correr el experimento.** El repositorio ya trae los datos procesados que usan los modelos (`parte1/data/processed/pd_matured.parquet`, 3 MB) y los modelos entrenados (`parte1/outputs/models/`).

El archivo original no está en el repositorio: `loan.csv` pesa 442 MB y GitHub no admite archivos de más de 100 MB. Para regenerar los datos procesados desde cero:

1. Descarga el conjunto *Credit Risk Analysis* de Kaggle: https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset. El nuestro se descargó el **24 de octubre de 2026**.
2. Copia `loan.csv` y `LCDataDictionary.xlsx` en `data/raw/` (carpeta compartida con la Parte 2).
3. Ejecuta `python parte1/run.py --etapas datos`.

Esa etapa aplica, en orden:
1. Construye la variable objetivo desde `loan_status`, según la tabla del enunciado.
2. Selecciona la muestra: créditos que ya habían terminado su plazo en enero de 2016, la fecha de corte del archivo.
3. Aplica una limpieza fija.
4. Hace la partición temporal por fecha de originación.

El resultado son 78.302 créditos:
- 74.930 a 36 meses, originados entre 2007-06 y 2012-12.
- 3.372 a 60 meses, todos de 2010. Lending Club ofrece este plazo desde mayo de 2010, y los originados después aún no habían vencido en el corte.

Entrenamiento va hasta 2012-06, validación es 2012 T3 y prueba es 2012 T4. Por eso los créditos a 60 meses quedan todos en entrenamiento: el efecto del plazo no se puede validar fuera de tiempo. El detalle y la justificación están en `parte1/docs/caso_de_uso.md`.

## 3. Ejecutar el experimento

Todo se lanza con un solo comando:

| Comando | Qué hace | Tiempo aprox. |
|---|---|---|
| `python parte1/run.py --demo` | Demostración: todas las etapas con un presupuesto pequeño (MLP con la mejor arquitectura ya encontrada, 3 semillas, 20 épocas). Escribe en `parte1/outputs_demo/` y no toca los resultados oficiales. | 1 a 1,5 min |
| `python parte1/run.py` | Experimento completo: datos, EDA, logística, búsqueda del MLP con 25 configuraciones, 30 semillas, figuras, SHAP, scorecard e informe. | 15–35 min según la máquina |
| `python parte1/run.py --etapas scorecard informe` | Solo las etapas indicadas, a partir de lo que ya está guardado. | segundos |
| `python parte1/run.py --listar` | Lista las etapas disponibles. | — |

Etapas, en orden:

| Etapa | Qué produce |
|---|---|
| `datos` | `parte1/data/processed/pd_matured.parquet`. Se salta si no está `data/raw/loan.csv`. |
| `eda` | Contraste de las 13 hipótesis en entrenamiento y figuras del EDA. |
| `logistica` | Regresión logística con L2; elige C por log-loss en validación. |
| `mlp` | Búsqueda aleatoria de arquitectura e hiperparámetros, y la mejor configuración entrenada con 30 semillas. |
| `figuras` | Curvas ROC, de calibración y de pérdida por época. |
| `interpretabilidad` | SHAP sobre el MLP y la logística. |
| `scorecard` | Puntajes, tabla de puntos y bandas de riesgo. |
| `clasificacion` | Accuracy, F1 y matriz de confusión con dos umbrales. |
| `hipotesis` | Efecto de cada hipótesis en los modelos, controlando el resto de variables. |
| `comparacion` | Logística contra MLP: métricas, costos y bootstrap de la diferencia. |
| `informe` | `parte1/outputs/results/pd_resumen.md`, con todas las tablas. |

## 4. Cambiar parámetros

Los parámetros están en [`parte1/config.yaml`](parte1/config.yaml): plazos incluidos, semilla, rejilla de C, número de configuraciones y semillas del MLP, épocas, paciencia, parámetros de la scorecard y de SHAP, y réplicas del bootstrap. Se pueden editar ahí o cambiar desde la línea de comandos con `--set`, sin tocar el código:

```bash
python parte1/run.py --demo --set mlp.fixed.lr=0.05 mlp.fixed.hidden=[32] mlp.n_seeds=5
```

Más ejemplos:
- `python parte1/run.py --set mlp.n_trials=50 mlp.max_epochs=200` amplía la búsqueda.
- `python parte1/run.py --etapas scorecard informe --set scorecard.pdo=40 scorecard.base_score=700` cambia la escala de la scorecard sin reentrenar. Sin `--demo` se sobrescriben las salidas; para volver a los resultados oficiales usa `git checkout parte1/outputs/`.
- `python parte1/run.py --etapas mlp figuras --set mlp.fixed.dropout=0.5 mlp.fixed.optimizer=adam mlp.fixed.lr=0.001` entrena una arquitectura fija sin búsqueda.

Con `mlp.fixed` el MLP se salta la búsqueda y entrena esa configuración. Los campos posibles son `hidden`, `activation`, `dropout`, `l2`, `optimizer`, `lr`, `momentum` y `batch_size`; lo que no se indique toma el valor por defecto de `MLPConfig` en `parte1/modeling/core.py`.

## 5. Aplicación web

```bash
streamlit run app.py
```

La app carga los modelos guardados en `parte1/outputs/models/` y nunca reentrena. Para cada perfil muestra:
- la PD;
- el puntaje de la scorecard;
- el percentil frente a los créditos de entrenamiento;
- cómo cambiarían esos valores si se modifica alguna característica.

Los modelos disponibles y la escala de la scorecard se configuran en `app_config.json`. Más detalles en `README_APP.md`, `README_INTERFAZ.md` y `README_DESPLIEGUE_DE_APP.md`.

Para comprobar que la app reproduce las predicciones guardadas y que la scorecard es correcta:

```bash
python scripts/verificar_predicciones_historicas.py
python scripts/verificar_scorecard_percentiles.py
```

## 6. Resultados

Partición de prueba (2012 T4, 15.136 créditos). Tablas completas en [`parte1/outputs/results/pd_resumen.md`](parte1/outputs/results/pd_resumen.md); conclusiones en [`parte1/docs/eda_y_conclusiones.md`](parte1/docs/eda_y_conclusiones.md).

| Métrica | Tasa base | Regresión logística | MLP (media ± d.e., 30 semillas) | MLP guardado |
|---|---|---|---|---|
| AUC | 0,500 | 0,638 | 0,641 ± 0,002 | 0,645 |
| Gini | 0 | 0,276 | 0,283 ± 0,004 | 0,290 |
| KS | 0 | 0,206 | 0,214 ± 0,005 | 0,220 |
| Brier | 0,1137 | 0,1110 | 0,1111 ± 0,0002 | 0,1106 |
| Parámetros entrenables | — | 38 | 13.185 | 13.185 |

- **La red aporta poco:** el MLP guardado mejora el AUC en 0,007 frente a la logística (IC 95 % [+0,002; +0,012]); en promedio sobre las 30 semillas la mejora es 0,003, y la calibración no mejora.
- **El límite lo pone la información disponible**, no el tipo de modelo.
- **Variables que más pesan en ambos:** ingreso, subcalificación de Lending Club, consultas recientes, propósito del crédito y utilización del crédito rotativo.
- **Plazo de 60 meses:** a igual perfil sube la PD entre 4,5 y 7,1 pp según el modelo, pero ese efecto se estima con una sola cohorte (2010).

Salidas del experimento, en `parte1/outputs/`:

| Carpeta | Contenido |
|---|---|
| `models/` | `pd_logistic.joblib` (preprocesador y logística), `pd_mlp.pt` (red), `pd_mlp_preprocessor.joblib`. |
| `results/` | `pd_resumen.md` (legible), `pd_analisis.json` (EDA, SHAP, scorecard, comparación), `pd_*_metricas.json` (métricas, búsqueda y semillas de cada modelo), `pd_*_predicciones.parquet` (los usa la app), `pd_mlp_historial.csv` (pérdida por época). |
| `figures/` | Figuras del reporte; todas se regeneran con `python parte1/run.py --etapas eda figuras interpretabilidad clasificacion`. |

## 7. Reproducibilidad

- **Semillas fijas:**
  - `semilla: 42` en `parte1/config.yaml` controla la logística, la búsqueda aleatoria del MLP, la muestra de clientes de SHAP y el bootstrap.
  - La partición no es aleatoria: es temporal, por fecha de originación.
  - Las 30 corridas finales del MLP usan las semillas 0 a 29.
  - El modelo guardado es la semilla con menor log-loss en validación.
- **Versiones fijas** en `requirements.txt`.
- **Sin fuga de información:** el preprocesamiento (imputación, transformaciones, escalado, one-hot) se ajusta solo con entrenamiento.
- **Determinismo entre máquinas:** en una misma máquina el entrenamiento en CPU es determinista. Entre máquinas distintas, el AUC puede variar en el orden de 10⁻⁴ por diferencias de punto flotante; la media de las 30 semillas se reproduce igual.

## 8. Estructura del repositorio

```
app.py, app_config.json     aplicación web, compartida por ambas partes (web/app.py: entrada para Streamlit Cloud)
scripts/                    verificaciones de la app
data/raw/                   loan.csv y diccionario (no versionados), compartidos por ambas partes
parte1/                     Parte 1: probabilidad de incumplimiento
  run.py                    comando único del experimento
  config.yaml               parámetros (editables o con --set)
  common/                   rutas, semillas y estilo de figuras
  data_prep/                lectura, variable objetivo, muestra, limpieza, partición y preprocesamiento
  modeling/                 regresión logística y MLP (definición, entrenamiento y experimentos)
  evaluation/               métricas, umbrales, comparación y figuras de evaluación
  analysis/                 EDA e hipótesis, SHAP y contraste de hipótesis con los modelos
  scorecard/                transformación PD → puntaje y scorecard de la logística
  report.py                 informe automático (outputs/results/pd_resumen.md)
  inference.py              carga de modelos para la app
  data/processed/           datos procesados que usan los modelos
  outputs/                  modelos, resultados y figuras regenerables
  docs/                     caso de uso, conclusiones y reporte técnico
parte2/                     Parte 2: pérdida dado incumplimiento (LGD), en construcción
  analisis.py, config.py    experimento (python parte2/analisis.py) y sus parámetros
  requirements.txt          dependencias propias (TensorFlow)
  data/processed/, outputs/ datos procesados y figuras de la Parte 2
```

Cada función del código tiene un comentario corto que dice qué hace.
