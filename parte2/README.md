# Parte 2: LGD y proxy de pérdida

La Parte 2 obtiene su muestra con `build_pd_dataset()` de la Parte 1 en modo
`matured`. Reutiliza su limpieza, variable objetivo y particiones temporales.
Conserva las columnas posteriores al otorgamiento que necesita la LGD, pero no
las usa como predictoras.

## Ejecución y caché

La primera ejecución necesita el CSV original en `data\raw\loan.csv`. Construye
un caché Parquet comprimido con predictores, `id`, particiones y columnas
auxiliares de LGD. Las ejecuciones siguientes cargan ese caché, sin necesitar el
CSV original ni acceso a internet:

```powershell
python parte2\analisis.py
```

Para regenerar el caché después de cambiar el CSV o la definición de LGD,
coloca el original en la ruta anterior y ejecuta:

```powershell
python parte2\analisis.py --rebuild-cache
```

El archivo anterior `data\processed\lending_resueltos.csv.gz` no sirve como
caché alineado: no contiene `id` ni `split`. Por eso se debe crear el nuevo
Parquet una vez con el CSV original disponible. Después, se puede mover o quitar
el CSV y ejecutar la Parte 2 desde el Parquet.

Los archivos de predicción
`results\pd_logistic_predicciones.parquet` y
`results\pd_mlp_predicciones.parquet` deben corresponder al modo `matured` de la
Parte 1. Para generarlos o actualizarlos:

```powershell
python run.py --etapas logistica mlp
```

## Archivos generados

- `parte2\data\processed\lending_matured_lgd.parquet`: caché comprimido,
  alineado con los `id` y `split` de Parte 1.
- `parte2\data\processed\lending_matured_lgd.json`: versión y metadatos para
  comprobar la compatibilidad del caché.
- `parte2\outputs\lgd_predicciones_test.parquet`: predicciones de LGD lineal y
  MLP para todos los créditos del test, también los que no tienen LGD observada.
- `parte2\outputs\perdida_esperada_test.parquet`: predicciones de PD y LGD unidas
  por `id`, con sus combinaciones de pérdida esperada.
- `parte2\outputs\lgd_metricas.json` y `lgd_mlp_semillas.csv`: métricas,
  configuración, presupuesto de búsqueda y resultados por semilla.
- `parte2\outputs\lgd_distribuciones.png`: comparación de distribuciones.

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
