# Parte 2: LGD y proxy de pérdida

La Parte 2 obtiene su muestra con `build_pd_dataset()` de la Parte 1 en modo
`matured`. Reutiliza su limpieza, variable objetivo y particiones temporales.
Conserva las columnas posteriores al otorgamiento que necesita la LGD, pero no
las usa como predictoras.

## Ejecución y caché

Desde la raíz del repositorio, crea el entorno e instala el conjunto de
dependencias de Parte 2 (incluye las de la raíz y TensorFlow):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r parte2\requirements.txt
```

Después, sin subcomando se abre el menú interactivo de sustentación:

```powershell
.\.venv\Scripts\python.exe parte2\orquestador.py
```

Cada modelo se puede ejecutar por separado o en conjunto. La salida de cada
ejecución queda aislada por modelo y modo (`outputs\<modelo>\` o
`outputs_demo\<modelo>\`):

```powershell
.\.venv\Scripts\python.exe parte2\orquestador.py lineal
.\.venv\Scripts\python.exe parte2\orquestador.py perceptron
.\.venv\Scripts\python.exe parte2\orquestador.py conjunto
.\.venv\Scripts\python.exe parte2\orquestador.py demo --modelo perceptron
.\.venv\Scripts\python.exe parte2\orquestador.py resumen --modelo lineal
.\.venv\Scripts\python.exe parte2\orquestador.py figuras --modelo conjunto
.\.venv\Scripts\python.exe parte2\orquestador.py proxy
```

`lineal` entrena solo la regresión; `perceptron` entrena solo la red neuronal;
`conjunto` ejecuta ambos, compara sus métricas y calcula el proxy de pérdida.
La salida de consola muestra únicamente métricas de prueba, parámetros
principales, figura y tiempo total.

La regresión tiene parámetros propios `--lineal-fit-intercept` /
`--no-lineal-fit-intercept` y `--lineal-positive` / `--no-lineal-positive`.
El perceptrón tiene arquitectura, dropout, tasa de aprendizaje, épocas, tamaño
del ensamble, batch y paciencia. Ejemplos:

```powershell
.\.venv\Scripts\python.exe parte2\orquestador.py lineal `
  --no-lineal-fit-intercept --lineal-positive --semilla 42
.\.venv\Scripts\python.exe parte2\orquestador.py perceptron `
  --arquitectura 128,64 --dropout 0.2 --learning-rate 0.001 `
  --epocas 100 --semillas 30 --batch-size 256 --paciencia 8 --semilla 42
.\.venv\Scripts\python.exe parte2\orquestador.py conjunto `
  --arquitectura 128,64 --dropout 0.2 --learning-rate 0.001 `
  --epocas 100 --semillas 30 --batch-size 256 --paciencia 8
```

Si se omiten opciones del perceptrón en una ejecución completa, `perceptron`
usa la arquitectura predeterminada y `conjunto` usa la cuadrícula de `config.py`.
En ambos casos, cualquier parámetro explícito del MLP restringe la búsqueda a
esa configuración. Los valores predeterminados y la cuadrícula se pueden editar
en `parte2\config.py`; el menú también solicita los parámetros antes de entrenar.

`demo` limita cada partición a 800 filas, utiliza hasta dos semillas y diez
épocas máximas; puede combinarse con `--modelo lineal`, `perceptron` o `conjunto`.
El orquestador cronometra la demo y avisa si supera 120 segundos. Los resultados
de demo están separados y no deben reportarse como resultados finales.
Los modelos y el preprocesador se guardan en la subcarpeta `models` de cada
ejecución, por ejemplo `outputs\perceptron\models\`.

El caché alineado `parte2\data\processed\lending_matured_lgd.parquet` y su
metadata están incluidos en el repositorio. La Parte 2 puede correr desde ese
caché sin conexión ni CSV original. Para reconstruirlo deliberadamente, coloca
el archivo original en `data\raw\loan.csv` y usa el menú (opción 7) o ejecuta:

```powershell
.\.venv\Scripts\python.exe parte2\orquestador.py reconstruir-cache --si
```

La regeneración reemplaza el caché existente y por eso requiere confirmación
explícita. El archivo comprimido `lending_resueltos.csv.gz` no sustituye al
caché alineado porque no contiene `id` ni las particiones usadas por la Parte 1.

Los archivos de predicción
`parte1\salidas\resultados\pd_logistic_predicciones.parquet` y
`parte1\salidas\resultados\pd_mlp_predicciones.parquet` deben corresponder al
modo `matured` de la Parte 1. Para generarlos o actualizarlos:

```powershell
.\.venv\Scripts\python.exe parte1\run.py --etapas logistica mlp
```

## Archivos generados

- `parte2\data\processed\lending_matured_lgd.parquet`: caché comprimido,
  alineado con los `id` y `split` de Parte 1.
- `parte2\data\processed\lending_matured_lgd.json`: versión y metadatos para
  comprobar la compatibilidad del caché.
- `parte2\outputs\<lineal|perceptron|conjunto>\`: predicciones, métricas,
  figuras y modelos serializados de cada método o del análisis conjunto.
- `parte2\outputs\conjunto\perdida_esperada_test.parquet`: predicciones de PD y
  LGD unidas por `id`, con las combinaciones del proxy de pérdida.
- `parte2\outputs_demo\<lineal|perceptron|conjunto>\`: salidas rápidas aisladas
  de los resultados completos.

`resumen`, `figuras` y `proxy` leen archivos guardados; no reentrenan. Para
consultar una corrida de demo se agregan `--demo` y el modelo correspondiente:

```powershell
.\.venv\Scripts\python.exe parte2\orquestador.py resumen --demo --modelo perceptron
```

La severidad observada se calcula únicamente en préstamos `Charged Off` con
`funded_amnt > 0`, `EAD_obs > 0` y recuperaciones disponibles. Primero se
estima `EAD_obs = funded_amnt - total_rec_prncp` y la recuperación neta como
`rec_net = recoveries - collection_recovery_fee`. El objetivo de LGD usado por
los modelos es `(EAD_obs - rec_net) / funded_amnt`, recortado a `[0, 1]`.
Esta variable mide pérdida observada acumulada como fracción del principal
financiado al origen. No es una LGD estándar sobre el EAD a la fecha de default:
`EAD_obs` es una aproximación a partir de pagos acumulados, no necesariamente el
saldo exacto al primer atraso o al default, y las recuperaciones pueden estar
censuradas por la fecha de corte. El recorte a `[0, 1]` limita el objetivo del
modelo, pero también elimina la magnitud de recuperaciones netas superiores al
saldo y de pérdidas mayores que el principal inicial.

Para cada crédito de prueba se calcula un **proxy indicativo**:
`PD * LGD_predicha * funded_amnt`:

- `PD` es la probabilidad de incumplimiento predicha por cada modelo de la
  Parte 1, unida por `id`. El target de Parte 1 considera malos `Charged Off`,
  `Default`, `Late (31-120 days)` y el estado de política equivalente a
  `Charged Off`.
- `LGD` es la predicción del modelo lineal o el promedio de predicciones MLP de
  30 semillas. No se usa la LGD observada del crédito de prueba para calcular
  el proxy, evitando incorporar el desenlace observado en la estimación.
- `funded_amnt` es el monto realmente financiado al origen. Es una base coherente
  con esta medida de severidad sobre el principal inicial, pero no es el saldo
  expuesto al default. La LGD observada se entrena solo con `Charged Off`, mientras
  que la PD también considera estados `Late (31-120 days)`; por esa diferencia,
  el producto no debe presentarse como pérdida esperada estándar ni como una
  estimación validada de pérdidas realizadas. En el parquet, el monto se llama
  `EAD_origin` y los productos `EL_proxy_*` para hacer explícita esta convención.

El archivo de salida contiene el proxy por crédito para cada combinación
PD/LGD y muestra tanto el promedio por crédito como la suma para la cartera de
prueba. Son estimaciones retrospectivas; la suma no debe interpretarse como una
validación contra pérdida realizada, especialmente porque las recuperaciones
pueden estar censuradas. El modo demo reduce semillas y épocas y no debe usarse
para reportar resultados finales.
