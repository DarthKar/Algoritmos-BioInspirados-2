# Resumen de resultados, Parte 1 (generado automáticamente)

No editar a mano: lo reescribe `python run.py`. Las rutas de figuras son relativas a `parte1/salidas/`.

## 1. Datos

- Registros: 78,302; variables guardadas: 57; originación 2007-06-01 a 2012-12-01; descarga de Kaggle: 2026-10-24.
- Partición temporal: entrenamiento 49,844 (13.4 % malos), validación 13,322 (14.2 %), prueba 15,136 (13.1 %).

Calidad de las variables numéricas en entrenamiento (antes del preprocesamiento):

| Variable | Faltantes | Ceros | Negativos | Mínimo | P1 | Mediana | P99 | Máximo |
|---|---|---|---|---|---|---|---|---|
| loan_amnt | 0.0% | 0.0% | 0 | 500 | 1,200 | 9,000 | 3.3e+04 | 3.5e+04 |
| term | 0.0% | 0.0% | 0 | 36 | 36 | 36 | 60 | 60 |
| emp_length | 2.9% | 11.6% | 0 | 0 | 0 | 4 | 10 | 10 |
| annual_inc | 0.0% | 0.0% | 0 | 1,896 | 1.44e+04 | 5.7e+04 | 2.35e+05 | 6e+06 |
| dti | 0.0% | 0.4% | 0 | 0 | 0.4043 | 13.71 | 28.07 | 34.92 |
| delinq_2yrs | 0.0% | 89.4% | 0 | 0 | 0 | 0 | 2 | 13 |
| inq_last_6mths | 0.0% | 48.0% | 0 | 0 | 0 | 1 | 6 | 33 |
| mths_since_last_delinq | 0.0% | 0.0% | 0 | 1 | 4 | 93 | 396 | 657 |
| open_acc | 0.0% | 0.0% | 0 | 1 | 2 | 9 | 22 | 49 |
| pub_rec | 0.0% | 95.2% | 0 | 0 | 0 | 0 | 1 | 5 |
| revol_util | 0.2% | 2.1% | 0 | 0 | 0 | 52.7 | 98.2 | 119 |
| total_acc | 0.0% | 0.0% | 0 | 1 | 4 | 20 | 55 | 90 |
| credit_history_months | 0.0% | 0.0% | 0 | 6 | 39 | 147 | 428 | 785 |
| never_delinquent | 0.0% | 34.0% | 0 | 0 | 0 | 1 | 1 | 1 |
| revol_limit | 2.2% | 0.1% | 0 | 0 | 1,000 | 2.027e+04 | 2.559e+05 | 4.414e+07 |

## 2. Hipótesis: EDA univariado y efecto controlado en los modelos

EDA en entrenamiento (Mann-Whitney, α = 0,01, efecto mínimo). Δ PD: cambio medio de la PD al mover la variable de un valor bajo a uno alto dejando las demás fijas (pp = puntos porcentuales). Figura: `figuras/pd_eda_hipotesis.png`.

| Id | Hipótesis | Tasa de malos bajo → alto | AUC univ. | EDA | Escenario | Δ PD LR (pp) | Δ PD MLP (pp) | LR | MLP | Rango SHAP |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Peor subcalificación de Lending Club (sub_grade_num alto) -> más riesgo | 5.9 % → 22.8 % | 0.645 | Confirmada | sub_grade_num: P25 5 -> P75 13 | — | — | Fuera del modelo | Fuera del modelo | — |
| H2 | Mayor utilización del crédito rotativo (revol_util) -> más riesgo | 9.5 % → 18.0 % | 0.572 | Confirmada | revol_util: P25 28 -> P75 74.1 | +6.32 | +6.06 | Confirmada | Confirmada | 2.0 |
| H3 | Mayor ingreso anual (annual_inc) -> menos riesgo | 18.0 % → 9.8 % | 0.432 | Confirmada | annual_inc: P25 40000 -> P75 80000 | -4.53 | -5.36 | Confirmada | Confirmada | 1.0 |
| H4 | Más consultas recientes (inq_last_6mths) -> más riesgo | 11.8 % → 22.5 % | 0.576 | Confirmada | inq_last_6mths: 0 -> 2 | +4.17 | +5.44 | Confirmada | Confirmada | 3.0 |
| H5 | Mayor monto del crédito (loan_amnt) -> más riesgo | 14.1 % → 14.0 % | 0.499 | Sin evidencia suficiente | loan_amnt: P25 5000 -> P75 14000 | +2.54 | +3.68 | Confirmada | Confirmada | 4.0 |
| H6 | Mayor endeudamiento sobre ingreso (dti) -> más riesgo | 11.5 % → 15.3 % | 0.533 | Confirmada | dti: P25 8.46 -> P75 18.95 | +0.29 | +0.71 | Efecto despreciable | Confirmada | 8.0 |
| H7 | Tener moras o registros públicos previos (delinq_2yrs o pub_rec > 0) -> más riesgo | 12.8 % → 17.4 % | 0.525 | Confirmada | sin moras -> 1 mora hace 12 meses | +2.36 | +3.24 | Confirmada | Confirmada | 12.0 |
| H8 | Historial crediticio más largo (credit_history_months) -> menos riesgo | 15.2 % → 11.9 % | 0.472 | Confirmada | credit_history_months: P25 105 -> P75 201 | -0.42 | -1.53 | Efecto despreciable | Confirmada | 7.0 |
| H9 | No informar antigüedad laboral (emp_length faltante) -> más riesgo | 13.2 % → 20.0 % | 0.508 | Confirmada | emp_length: mediana 4 -> faltante | +5.34 | +3.01 | Confirmada | Confirmada | 11.0 |
| H10 | Mayor antigüedad laboral (emp_length) -> menos riesgo | 13.5 % → 13.4 % | 0.501 | Sin evidencia suficiente | emp_length: P25 2 -> P75 9 | +1.52 | +1.74 | Contraria | Contraria | 11.0 |
| H11 | Ingreso verificado -> menos riesgo | 13.6 % → 13.3 % | 0.497 | Sin evidencia suficiente | Not Verified -> Verified | -0.60 | -0.75 | Confirmada | Confirmada | 15.0 |
| H12 | Crédito para pequeño negocio (purpose = small_business) -> más riesgo | 13.0 % → 23.5 % | 0.518 | Confirmada | debt_consolidation -> small_business | +11.62 | +9.94 | Confirmada | Confirmada | 5.0 |
| H13 | Plazo de 60 meses (term) -> más riesgo | 12.8 % → 22.5 % | 0.526 | Confirmada | term: 36 -> 60 meses | +8.27 | +11.34 | Confirmada | Confirmada | 16.0 |

## 3. Logística contra MLP en prueba

MLP: media y desviación de 30 semillas; peor y mejor semilla. Figuras: `figuras/pd_*_roc.png`, `figuras/pd_*_calibracion.png`, `figuras/pd_mlp_perdidas.png`.

| Métrica | Tasa base | Logística | MLP media | MLP d.e. | MLP peor | MLP mejor | MLP guardado |
|---|---|---|---|---|---|---|---|
| auc | 0.5000 | 0.6222 | 0.6258 | 0.0022 | 0.6202 | 0.6291 | 0.6287 |
| gini | 0.0000 | 0.2444 | 0.2516 | 0.0043 | 0.2405 | 0.2582 | 0.2575 |
| ks | 0.0000 | 0.1805 | 0.1897 | 0.0048 | 0.1783 | 0.1964 | 0.1954 |
| brier | 0.1137 | 0.1116 | 0.1118 | 0.0002 | 0.1123 | 0.1114 | 0.1115 |
| log_loss | 0.3880 | 0.3781 | 0.3783 | 0.0006 | 0.3798 | 0.3774 | 0.3775 |

Bootstrap pareado en prueba (1000 réplicas), MLP guardado menos logística: ΔAUC +0.0066 (IC 95 % +0.0003 a +0.0126); ΔBrier -0.00003 (IC 95 % -0.00043 a +0.00033).

| Costo | Logística | MLP |
|---|---|---|
| parametros | 37 | 13,057 |
| configuraciones_evaluadas | 5 | 25 |
| semillas | 1 | 30 |
| segundos_totales | 1.0499 | 693.12 |
| interpretabilidad | coeficientes; puntos exactos por unidad | puntos por variable con SHAP |

## 4. Accuracy, F1 y matriz de confusión (prueba)

Umbral 0,5 y umbral que maximiza F1 en validación. Figuras: `figuras/pd_matriz_confusion.png` y `figuras/pd_distribucion_por_clase.png`.

| Modelo | Umbral | Valor | Accuracy | Precisión | Sensibilidad | Especificidad | F1 | Rechazados | VN | FP | FN | VP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pd_logistic | 0.5 | 0.500 | 0.869 | 0.500 | 0.001 | 1.000 | 0.001 | 0.000 | 13155 | 1 | 1979 | 1 |
| pd_logistic | max F1 en validación | 0.139 | 0.617 | 0.178 | 0.534 | 0.630 | 0.268 | 0.392 | 8286 | 4870 | 922 | 1058 |
| pd_mlp | 0.5 | 0.500 | 0.869 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 13155 | 1 | 1980 | 0 |
| pd_mlp | max F1 en validación | 0.156 | 0.652 | 0.189 | 0.505 | 0.674 | 0.275 | 0.349 | 8867 | 4289 | 981 | 999 |
| referencia_todos_buenos | ninguno | 1.000 | 0.869 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 13156 | 0 | 1980 | 0 |

## 5. Variables de riesgo (SHAP, |valor| medio sobre el logit)

Figura: `figuras/pd_mlp_shap.png`.

| Variable | MLP | Rango MLP | Logística | Rango LR |
|---|---|---|---|---|
| annual_inc | 0.266 | 1 | 0.247 | 2 |
| revol_util | 0.229 | 2 | 0.266 | 1 |
| inq_last_6mths | 0.198 | 3 | 0.165 | 3 |
| loan_amnt | 0.176 | 4 | 0.144 | 4 |
| purpose | 0.148 | 5 | 0.118 | 5 |
| open_acc | 0.110 | 6 | 0.107 | 6 |
| credit_history_months | 0.063 | 7 | 0.036 | 12 |
| dti | 0.060 | 8 | 0.018 | 17 |
| revol_limit | 0.060 | 9 | 0.038 | 10 |
| total_acc | 0.050 | 10 | 0.047 | 9 |
| emp_length | 0.044 | 11 | 0.084 | 7 |
| delinq_2yrs | 0.037 | 12 | 0.034 | 13 |
| never_delinquent | 0.022 | 13 | 0.080 | 8 |
| home_ownership | 0.021 | 14 | 0.038 | 11 |
| verification_status | 0.015 | 15 | 0.028 | 15 |
| term | 0.015 | 16 | 0.031 | 14 |
| mths_since_last_delinq | 0.011 | 17 | 0.016 | 18 |
| pub_rec | 0.005 | 18 | 0.027 | 16 |

## 6. Scorecard del MLP (con la de la logística como comparación)

Puntaje base 600 con odds 7:1 y PDO 20: puntaje = 543.85 + 28.85 · ln(odds buenos/malos). Los puntos por variable salen de SHAP sobre el logit: cada variable aporta −factor · φ puntos (`parte1/puntaje/shap_points.py`). Puntos base del MLP: 600.4; error máximo de aditividad: 5.4e-06 puntos en 2,000 créditos. AUC en prueba con el puntaje: MLP 0.6287, logística 0.6222.

Puntos medios absolutos por variable (cuánto mueve cada variable el puntaje, en promedio):

| Variable | MLP | Logística |
|---|---|---|
| annual_inc | 8.09 | 7.21 |
| revol_util | 7.84 | 7.92 |
| inq_last_6mths | 6.41 | 5.00 |
| loan_amnt | 4.74 | 3.71 |
| open_acc | 3.98 | 3.21 |
| purpose | 3.82 | 3.79 |
| term | 2.54 | 1.93 |
| total_acc | 2.12 | 1.46 |
| revol_limit | 2.10 | 1.33 |
| credit_history_months | 2.00 | 0.93 |
| emp_length | 1.70 | 2.12 |
| dti | 1.52 | 0.45 |
| delinq_2yrs | 1.24 | 0.73 |
| home_ownership | 1.20 | 1.03 |
| verification_status | 1.14 | 0.81 |
| pub_rec | 1.07 | 1.04 |
| never_delinquent | 1.04 | 2.01 |
| mths_since_last_delinq | 0.80 | 0.44 |

Puntos medios del MLP por tramo de las cinco variables más importantes:

| Variable | Tramo | Créditos | Puntos medios |
|---|---|---|---|
| loan_amnt | 500 a 5,000 | 432 | +6.3 |
| loan_amnt | 5,100 a 8,000 | 417 | +3.2 |
| loan_amnt | 8,050 a 10,625 | 354 | +0.4 |
| loan_amnt | 10,750 a 15,000 | 400 | -2.7 |
| loan_amnt | 15,050 a 35,000 | 397 | -10.6 |
| inq_last_6mths | 0 a 1 | 1556 | +5.0 |
| inq_last_6mths | 2 a 2 | 250 | -6.7 |
| inq_last_6mths | 3 a 15 | 194 | -17.1 |
| open_acc | 2 a 6 | 475 | +5.7 |
| open_acc | 7 a 8 | 386 | +2.4 |
| open_acc | 9 a 10 | 371 | +0.1 |
| open_acc | 11 a 13 | 387 | -2.6 |
| open_acc | 14 a 38 | 381 | -8.1 |
| revol_util | 0 a 28 | 400 | +13.5 |
| revol_util | 28.1 a 47.5 | 400 | +5.2 |
| revol_util | 47.6 a 62.7 | 401 | -0.5 |
| revol_util | 62.8 a 78.4 | 398 | -6.1 |
| revol_util | 78.5 a 104 | 400 | -13.0 |
| revol_util | faltante | 1 | -9.8 |
| annual_inc | 2,000 a 36,000 | 412 | -14.4 |
| annual_inc | 36,396 a 50,000 | 434 | -5.3 |
| annual_inc | 50,004 a 64,000 | 357 | -0.2 |
| annual_inc | 64,170 a 87,000 | 399 | +5.1 |
| annual_inc | 87,079 a 1,440,000 | 398 | +13.8 |

Bandas en prueba con el puntaje del MLP:

| Banda (1 = más riesgo) | Créditos | Puntaje mín. | Puntaje máx. | PD media | Tasa observada |
|---|---|---|---|---|---|
| 1 | 1514 | 534.9 | 579.3 | 0.268 | 0.217 |
| 2 | 1514 | 579.3 | 585.5 | 0.207 | 0.199 |
| 3 | 1513 | 585.5 | 590.3 | 0.178 | 0.163 |
| 4 | 1514 | 590.3 | 594.5 | 0.157 | 0.162 |
| 5 | 1513 | 594.6 | 598.6 | 0.139 | 0.132 |
| 6 | 1514 | 598.6 | 602.9 | 0.122 | 0.122 |
| 7 | 1513 | 602.9 | 607.4 | 0.107 | 0.095 |
| 8 | 1514 | 607.4 | 612.8 | 0.092 | 0.089 |
| 9 | 1513 | 612.8 | 620.9 | 0.075 | 0.075 |
| 10 | 1514 | 620.9 | 668.5 | 0.049 | 0.055 |

## 7. MLP: búsqueda y configuración elegida

Método: búsqueda aleatoria (25 configuraciones, 327 s); 30 semillas (364 s). Parámetros entrenables: 13,057. Configuración elegida: `{"hidden": [128, 64], "activation": "elu", "dropout": 0.1, "l2": 1.328e-07, "optimizer": "rmsprop", "lr": 0.0008925903, "momentum": 0.9, "batch_size": 512, "max_epochs": 100, "patience": 8, "seed": 0}`. Logística: C = 1 entre 5 valores.

| Capas | Activación | Dropout | L2 | Optimizador | Tasa | Lote | Mejor época | Log-loss val. | AUC val. |
|---|---|---|---|---|---|---|---|---|---|
| [128, 64] | elu | 0.1 | 1.3e-07 | rmsprop | 8.9e-04 | 512 | 28 | 0.3910 | 0.6529 |
| [64, 32] | relu | 0.0 | 1.4e-04 | adam | 4.3e-03 | 256 | 7 | 0.3918 | 0.6503 |
| [128, 64, 32] | relu | 0.2 | 0.0e+00 | sgd | 8.6e-03 | 256 | 48 | 0.3920 | 0.6482 |
| [16, 8] | gelu | 0.3 | 1.2e-06 | rmsprop | 3.5e-03 | 1024 | 19 | 0.3920 | 0.6479 |
| [128, 64, 32] | gelu | 0.5 | 1.7e-05 | rmsprop | 5.7e-04 | 256 | 19 | 0.3921 | 0.6474 |
