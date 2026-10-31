# ¿Me prestas? Riesgo de crédito con redes neuronales y algoritmos bioinspirados

Proyecto académico sobre créditos de Lending Club. Estudia dos componentes del
riesgo:

- **Probabilidad de incumplimiento (PD):** clasificación binaria. Compara una
  regresión logística con un perceptrón multicapa (MLP) y construye una
  scorecard.
- **Pérdida dado incumplimiento (LGD):** regresión de la severidad observada,
  con un modelo lineal y un ensamble MLP. A partir de PD, LGD y monto financiado
  se calcula además un proxy indicativo de pérdida por crédito.

La aplicación Streamlit permite explorar la PD de un perfil con los modelos
guardados. Los resultados son un ejercicio retrospectivo y **no deben usarse
para aprobar o denegar créditos**.

## Contenido

1. [Requisitos e instalación](#1-requisitos-e-instalación)
2. [Datos y particiones](#2-datos-y-particiones)
3. [Recalcular resultados con modelos existentes](#3-recalcular-resultados-con-modelos-existentes)
4. [Entrenar o cambiar modelos](#4-entrenar-o-cambiar-modelos)
5. [Resultados y metodología](#5-resultados-y-metodología)
6. [Aplicación web](#6-aplicación-web)
7. [Publicación y verificaciones](#7-publicación-y-verificaciones)
8. [Estructura de archivos](#8-estructura-de-archivos)

## 1. Requisitos e instalación

El proyecto se probó en Windows con Python 3.14 y CPU. La instalación principal
incluye las dependencias de la Parte 1 y de la aplicación:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En Linux o macOS:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Para ejecutar la Parte 2 instala además sus dependencias, incluido TensorFlow:

```powershell
python -m pip install -r parte2\requirements.txt
```

`parte2/requirements.txt` incorpora las dependencias de la raíz y fija las
versiones de TensorFlow/Keras usadas por los modelos serializados. En Windows
con CPU no se necesita GPU. En Linux, si no se usa GPU, se puede instalar la
versión CPU de PyTorch antes de los requisitos del proyecto:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

## 2. Datos y particiones

El repositorio incluye los datos procesados y los modelos requeridos para
reproducir inferencias y análisis sin descargar el CSV original:

- PD: `parte1/datos/procesados/pd_matured.parquet`.
- LGD: `parte2/data/processed/lending_matured_lgd.parquet`.
- Modelos PD y LGD: `parte1/salidas/modelos/` y `parte2/outputs/models/`.

El archivo original `loan.csv` (aprox. 442 MB) no está incluido. Para volver a
construir el dataset PD se puede descargar el conjunto *Credit Risk Analysis*
de [Kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset),
colocar `loan.csv` y `LCDataDictionary.xlsx` en `data/raw/` y ejecutar:

```powershell
python parte1\run.py --etapas datos
```

La muestra `matured` contiene 78.302 créditos originados entre junio de 2007 y
diciembre de 2012. La partición es temporal: entrenamiento hasta junio de 2012,
validación en 2012 T3 y prueba en 2012 T4. La Parte 2 usa la misma muestra,
limpieza e identificadores para alinear los datos y las predicciones por
crédito.

Para la Parte 2, el caché alineado y sus metadatos ya están en
`parte2/data/processed/`. Reconstruirlo desde `loan.csv` es una operación
distinta del recálculo de resultados; el orquestador de la raíz no reconstruye
el caché.

> **Privacidad y distribución:** antes de publicar el repositorio, revisa que
> los datos procesados y cualquier dato histórico puedan compartirse. Un
> `.gitignore` no elimina archivos que ya estén en el historial Git.

## 3. Recalcular resultados con modelos existentes

Desde la raíz, ejecuta:

```powershell
python orquestador.py
```

El orquestador abre un menú para ajustar parámetros de cálculo y luego:

1. Carga la regresión logística y la MLP PD guardadas y calcula de nuevo las
   predicciones y métricas en train, validación y prueba.
2. Carga la regresión lineal y los modelos MLP LGD ya entrenados, recalcula las
   predicciones y las métricas de prueba.
3. Une las predicciones por `id` y calcula los proxies de pérdida esperada.
4. Guarda las predicciones, métricas y resultados actualizados en las carpetas
   de salida.

Esta ruta **no entrena modelos ni ejecuta búsquedas de hiperparámetros**. El
tiempo depende del equipo y de los valores elegidos; al final informa cuánto
duró y avisa si supera dos minutos.

El menú permite cambiar los siguientes parámetros:

| Parámetro | Valor inicial | Efecto |
|---|---:|---|
| Modelos MLP LGD del ensamble | Todos los modelos guardados (30 en la configuración actual) | Limitar el promedio a los primeros N modelos guardados. Menos modelos suelen reducir el tiempo de predicción. |
| Tamaño del lote | 8192 | Número de filas por lote de inferencia; modifica el uso de memoria y puede afectar el tiempo. |
| Intervalos de calibración PD | 10 | Número de grupos usados al recalcular las tablas de calibración. |
| Umbral de clasificación PD | 0,5 | Umbral que determina la clase para las métricas de accuracy, F1 y matriz de confusión. |

Pulsa Enter para conservar el valor actual y `0` para iniciar los cálculos.
Los valores de entrenamiento —por ejemplo, learning rate, arquitectura, dropout,
número de épocas o paciencia— no se pueden cambiar desde este menú: determinan
los pesos de una red. El orquestador usa los pesos de los artefactos disponibles.
Para usar otros hiperparámetros debe existir un modelo previamente entrenado y
guardado con ellos.

También se puede recalcular cada parte por separado:

```powershell
python orquestador.py --solo parte1
python orquestador.py --solo parte2
```

`--solo parte2` usa las predicciones PD que ya estén guardadas. Si se selecciona
un ensamble LGD más pequeño, el cálculo usa las primeras N semillas ordenadas y
actualiza la métrica del ensamble con esa selección. El menú es interactivo;
si el proceso se ejecuta sin una consola interactiva, se usan los valores
predeterminados.

Salidas principales del recálculo:

- `parte1/salidas/resultados/pd_logistic_predicciones.parquet`
- `parte1/salidas/resultados/pd_mlp_predicciones.parquet`
- `parte1/salidas/resultados/pd_logistic_metricas.json`
- `parte1/salidas/resultados/pd_mlp_metricas.json`
- `parte2/outputs/lgd_predicciones_test.parquet`
- `parte2/outputs/lgd_metricas.json`
- `parte2/outputs/perdida_esperada_test.parquet`

## 4. Entrenar o cambiar modelos

El recálculo de la sección anterior no cambia los modelos. Para entrenar de
nuevo, usa explícitamente los comandos de cada parte; pueden tardar más de dos
minutos y sobrescribir artefactos de salida.

### Parte 1: PD

```powershell
python parte1\run.py --demo
python parte1\run.py
python parte1\run.py --etapas logistica mlp
python parte1\run.py --listar
```

`python parte1\orquestador.py` abre un menú para hacer lo mismo sin escribir comandos y para consultar
los resultados, incluidos los mejores hiperparámetros de la red.

`--demo` usa un presupuesto pequeño y guarda sus salidas en
`parte1/salidas_demo/`. El experimento completo incluye EDA, regresión
logística, búsqueda de MLP, semillas, figuras, interpretabilidad, scorecard y
reporte. Para ajustar parámetros de entrenamiento, usa `parte1/config.yaml` o
`--set`, por ejemplo:

```powershell
python parte1\run.py --etapas mlp --set mlp.n_seeds=5 mlp.max_epochs=50
python parte1\run.py --etapas mlp --set mlp.fixed.hidden=[32] mlp.fixed.lr=0.01
```

Etapas disponibles:

| Etapa | Trabajo |
|---|---|
| `datos` | Construye el dataset procesado desde el CSV original, si está disponible. |
| `eda` | Contrasta las hipótesis y genera análisis descriptivos. |
| `logistica` | Entrena la regresión logística y elige C con validación. |
| `mlp` | Busca hiperparámetros o entrena una configuración fija y guarda el modelo. |
| `figuras` | Genera curvas ROC, calibración y pérdidas desde resultados guardados. |
| `interpretabilidad` | Calcula SHAP para la logística y el MLP. |
| `scorecard` | Genera la scorecard del MLP (puntos por variable con SHAP), la de la logística como comparación y las bandas de riesgo. |
| `clasificacion` | Calcula métricas de clasificación por umbral. |
| `hipotesis` | Mide el efecto de escenarios sobre las PD de los modelos guardados. |
| `comparacion` | Compara los modelos con métricas y bootstrap. |
| `informe` | Genera `parte1/salidas/resultados/pd_resumen.md`. |

### Parte 2: LGD

El comando `python parte2/analisis.py` ejecuta el análisis de LGD y **entrena**
los modelos definidos en `parte2/config.py`; no es el comando de recálculo
rápido. Úsalo solo cuando se busque entrenar de nuevo o reconstruir resultados
del experimento:

```powershell
python parte2\analisis.py
```

La configuración del entrenamiento de LGD incluye arquitectura, dropout,
learning rate, épocas, tamaño de lote, paciencia, semillas y la cuadrícula de
arquitecturas. Cambiar esos valores exige entrenar un nuevo modelo. El
orquestador de la raíz no los modifica ni selecciona variantes por hiperparámetro;
solo puede reducir el número de modelos existentes usados en el ensamble.

## 5. Resultados y metodología

### Parte 1: PD

El informe legible de resultados está en
`parte1/salidas/resultados/pd_resumen.md`. El archivo
`parte1/salidas/resultados/pd_analisis.json` reúne resultados de EDA,
hipótesis, comparación, scorecard y SHAP.

La variable objetivo representa incumplimiento según los estados de préstamo
definidos por la Parte 1. El preprocesamiento se ajusta con entrenamiento; la
partición temporal evita mezclar datos futuros al entrenar.

Los modelos de PD no usan las evaluaciones de riesgo de Lending Club
(subcalificación, calificación, tasa de interés ni cuota): son salidas del
modelo de riesgo de la plataforma, y un solicitante no las conoce al pedir el
crédito.

### Parte 2: LGD y pérdida

La LGD observada se limita a préstamos `Charged Off` con monto financiado
positivo, exposición observada positiva y datos de recuperación disponibles.
Se calcula como:

```text
EAD_obs = funded_amnt - total_rec_prncp
rec_net = recoveries - collection_recovery_fee
LGD = clip((EAD_obs - rec_net) / funded_amnt, 0, 1)
```

Es una pérdida acumulada observada como fracción del principal financiado al
origen; no es una LGD estándar sobre el saldo exacto en la fecha de default.
Las recuperaciones pueden estar censuradas por la fecha de corte y el recorte a
`[0, 1]` limita los valores extremos.

Para cada crédito de prueba, el proxy se calcula como:

```text
PD predicha × LGD predicha × funded_amnt
```

La PD y la LGD se unen por `id`. La predicción LGD usa la regresión lineal o el
promedio de predicciones de los modelos MLP seleccionados; no usa la LGD
observada del crédito de prueba. `funded_amnt` corresponde al monto de origen,
no al saldo expuesto al incumplimiento. Además, el objetivo PD incluye estados
`Late (31-120 days)` mientras que la LGD observada solo incluye préstamos
castigados. Por estas diferencias, el resultado es un **proxy indicativo**, no
una medida estándar ni una estimación validada contra pérdidas realizadas.

## 6. Aplicación web

Desde la raíz del proyecto:

```powershell
streamlit run app.py
```

La aplicación carga artefactos guardados y no entrena. En `app_config.json` se
definen los modelos disponibles, los datos de referencia y la configuración de
la scorecard. La interfaz incluye:

- formulario con campos, unidades y valores de referencia;
- PD, puntaje y percentil para cada modelo disponible;
- de dónde sale el puntaje: los puntos que suma o resta cada característica;
- comparación de logística y MLP para el mismo perfil;
- simulación descriptiva de cambios en las características.

Los percentiles se calculan frente a las predicciones históricas de la partición
de referencia del modelo; no representan percentiles nacionales. La
comparación de escenarios no establece causalidad. Los perfiles viven en el
estado de sesión de Streamlit y no se guardan en una base de datos.

El simulador no recalcula automáticamente variables derivadas a partir de otras.
Si se modifican las características o el preprocesamiento del modelo, hay que
actualizar el adaptador de inferencia y validar los valores del formulario
contra el diccionario de Lending Club. Los artefactos serializados deben
provenir del proyecto y de fuentes confiables.

## 7. Publicación y verificaciones

La entrada local es `app.py`; la entrada alternativa para despliegue es
`web/app.py`. Los requisitos de la raíz están en `requirements.txt`; no hay un
archivo `web/requirements.txt` separado en esta versión del proyecto.

Antes de publicar:

1. Revisa los archivos, dependencias y artefactos incluidos.
2. Comprueba la compatibilidad de Python y de las bibliotecas con las que se
   serializaron los modelos.
3. Revisa la privacidad y autorización para distribuir los datos procesados.
4. Configura `report_url` y `video_url` en `app_config.json` cuando existan
   enlaces públicos.
5. Despliega `web/app.py` y prueba ambos modelos, las gráficas y el simulador
   desde una sesión externa.

Verificaciones disponibles desde la raíz:

```powershell
python scripts\verificar_despliegue.py
python scripts\verificar_integracion_web.py
python scripts\verificar_predicciones_historicas.py
python scripts\verificar_scorecard_percentiles.py
```

El chequeo de despliegue es estructural; no sustituye una prueba remota ni la
revisión de privacidad. Antes de actualizar un modelo, cambia de forma
coordinada el checkpoint, el preprocesador y las predicciones históricas, y
ejecuta las verificaciones de integración.

## 8. Estructura de archivos

```text
app.py, app_config.json      Aplicación local y configuración de modelos
web/app.py                   Entrada alternativa para despliegue
scripts/                     Verificaciones de la aplicación
data/raw/                    loan.csv y diccionario opcionales, no versionados
parte1/
  run.py                     Etapas de preparación, entrenamiento y análisis PD
  orquestador.py             Menú interactivo de la Parte 1 (correr etapas y ver resultados)
  config.yaml                Configuración del experimento PD
  modelos/                   Definición, entrenamiento e inferencia de PD
  preparacion/               Limpieza, dataset, particiones y variables
  evaluacion/                Métricas, comparación y figuras
  analisis/                  EDA, SHAP e hipótesis
  puntaje/                   Conversión PD a puntaje y scorecard
  datos/procesados/          Datasets utilizados por los modelos
  salidas/                   Modelos, resultados y figuras
parte2/
  analisis.py                Entrenamiento y análisis LGD
  config.py                  Configuración del experimento LGD
  requirements.txt           TensorFlow/Keras además de los requisitos raíz
  data/processed/            Caché LGD alineado con Parte 1
  outputs/                   Modelos, métricas, predicciones y proxy
orquestador.py               Recálculo PD/LGD/proxy con modelos guardados
README.md                    Guía única del proyecto
```
