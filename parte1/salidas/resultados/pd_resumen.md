# Resumen de resultados, Parte 1 (generado automáticamente)

No editar a mano: lo reescribe `python run.py`. Las conclusiones redactadas están en `parte1/documentos/eda_y_conclusiones.md`. Las rutas de figuras son relativas a `parte1/salidas/`.

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
| sub_grade_num | 0.0% | 0.0% | 0 | 1 | 1 | 9 | 29 | 35 |

## 2. Hipótesis: EDA univariado y efecto controlado en los modelos

EDA en entrenamiento (Mann-Whitney, α = 0,01, efecto mínimo). Δ PD: cambio medio de la PD al mover la variable de un valor bajo a uno alto dejando las demás fijas (pp = puntos porcentuales). Figura: `figuras/pd_eda_hipotesis.png`.

| Id | Hipótesis | Tasa de malos bajo → alto | AUC univ. | EDA | Escenario | Δ PD LR (pp) | Δ PD MLP (pp) | LR | MLP | Rango SHAP |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Peor subcalificación de Lending Club (sub_grade_num alto) -> más riesgo | 5.9 % → 22.8 % | 0.645 | Confirmada | sub_grade_num: P25 5 -> P75 13 | +4.01 | +5.78 | Confirmada | Confirmada | 2 |
| H2 | Mayor utilización del crédito rotativo (revol_util) -> más riesgo | 9.5 % → 18.0 % | 0.572 | Confirmada | revol_util: P25 28 -> P75 74.1 | +3.68 | +3.01 | Confirmada | Confirmada | 5 |
| H3 | Mayor ingreso anual (annual_inc) -> menos riesgo | 18.0 % → 9.8 % | 0.432 | Confirmada | annual_inc: P25 40000 -> P75 80000 | -4.57 | -5.41 | Confirmada | Confirmada | 1 |
| H4 | Más consultas recientes (inq_last_6mths) -> más riesgo | 11.8 % → 22.5 % | 0.576 | Confirmada | inq_last_6mths: 0 -> 2 | +2.97 | +4.00 | Confirmada | Confirmada | 3 |
| H5 | Mayor monto del crédito (loan_amnt) -> más riesgo | 14.1 % → 14.0 % | 0.499 | Sin evidencia suficiente | loan_amnt: P25 5000 -> P75 14000 | +1.54 | +2.58 | Confirmada | Confirmada | 6 |
| H6 | Mayor endeudamiento sobre ingreso (dti) -> más riesgo | 11.5 % → 15.3 % | 0.533 | Confirmada | dti: P25 8.46 -> P75 18.95 | +0.39 | +0.49 | Efecto despreciable | Efecto despreciable | 8 |
| H7 | Tener moras o registros públicos previos (delinq_2yrs o pub_rec > 0) -> más riesgo | 12.8 % → 17.4 % | 0.525 | Confirmada | sin moras -> 1 mora hace 12 meses | +0.14 | -0.12 | Efecto despreciable | Efecto despreciable | 9 |
| H8 | Historial crediticio más largo (credit_history_months) -> menos riesgo | 15.2 % → 11.9 % | 0.472 | Confirmada | credit_history_months: P25 105 -> P75 201 | +0.17 | -0.35 | Efecto despreciable | Efecto despreciable | 10 |
| H9 | No informar antigüedad laboral (emp_length faltante) -> más riesgo | 13.2 % → 20.0 % | 0.508 | Confirmada | emp_length: mediana 4 -> faltante | +5.44 | +2.79 | Confirmada | Confirmada | 11 |
| H10 | Mayor antigüedad laboral (emp_length) -> menos riesgo | 13.5 % → 13.4 % | 0.501 | Sin evidencia suficiente | emp_length: P25 2 -> P75 9 | +1.67 | +1.74 | Contraria | Contraria | 11 |
| H11 | Ingreso verificado -> menos riesgo | 13.6 % → 13.3 % | 0.497 | Sin evidencia suficiente | Not Verified -> Verified | -0.65 | -0.80 | Confirmada | Confirmada | 15 |
| H12 | Crédito para pequeño negocio (purpose = small_business) -> más riesgo | 13.0 % → 23.5 % | 0.518 | Confirmada | debt_consolidation -> small_business | +9.84 | +8.90 | Confirmada | Confirmada | 4 |
| H13 | Plazo de 60 meses (term) -> más riesgo | 12.8 % → 22.5 % | 0.526 | Confirmada | term: 36 -> 60 meses | +4.47 | +7.06 | Confirmada | Confirmada | 19 |

## 3. Logística contra MLP en prueba

MLP: media y desviación de 30 semillas; peor y mejor semilla. Figuras: `figuras/pd_*_roc.png`, `figuras/pd_*_calibracion.png`, `figuras/pd_mlp_perdidas.png`.

| Métrica | Tasa base | Logística | MLP media | MLP d.e. | MLP peor | MLP mejor | MLP guardado |
|---|---|---|---|---|---|---|---|
| auc | 0.5000 | 0.6380 | 0.6413 | 0.0020 | 0.6371 | 0.6451 | 0.6451 |
| gini | 0.0000 | 0.2761 | 0.2826 | 0.0040 | 0.2741 | 0.2902 | 0.2902 |
| ks | 0.0000 | 0.2060 | 0.2141 | 0.0049 | 0.2061 | 0.2244 | 0.2200 |
| brier | 0.1137 | 0.1110 | 0.1111 | 0.0002 | 0.1116 | 0.1106 | 0.1106 |
| log_loss | 0.3880 | 0.3755 | 0.3753 | 0.0007 | 0.3769 | 0.3739 | 0.3739 |

Bootstrap pareado en prueba (1000 réplicas), MLP guardado menos logística: ΔAUC +0.0071 (IC 95 % +0.0020 a +0.0123); ΔBrier -0.00040 (IC 95 % -0.00077 a -0.00005).

| Costo | Logística | MLP |
|---|---|---|
| parametros | 38 | 13,185 |
| configuraciones_evaluadas | 5 | 25 |
| semillas | 1 | 30 |
| segundos_totales | 2.0781 | 963.44 |
| interpretabilidad | coeficientes y scorecard directa | SHAP (aproximado) |

## 4. Accuracy, F1 y matriz de confusión (prueba)

Umbral 0,5 y umbral que maximiza F1 en validación. Figuras: `figuras/pd_matriz_confusion.png` y `figuras/pd_distribucion_por_clase.png`.

| Modelo | Umbral | Valor | Accuracy | Precisión | Sensibilidad | Especificidad | F1 | Rechazados | VN | FP | FN | VP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pd_logistic | 0.5 | 0.500 | 0.869 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 13156 | 0 | 1980 | 0 |
| pd_logistic | max F1 en validación | 0.139 | 0.659 | 0.195 | 0.512 | 0.681 | 0.282 | 0.344 | 8965 | 4191 | 967 | 1013 |
| pd_mlp | 0.5 | 0.500 | 0.869 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 13155 | 1 | 1980 | 0 |
| pd_mlp | max F1 en validación | 0.155 | 0.660 | 0.201 | 0.537 | 0.678 | 0.292 | 0.350 | 8921 | 4235 | 917 | 1063 |
| referencia_todos_buenos | ninguno | 1.000 | 0.869 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 13156 | 0 | 1980 | 0 |

## 5. Variables de riesgo (SHAP, |valor| medio sobre el logit)

Figura: `figuras/pd_mlp_shap.png`.

| Variable | MLP | Rango MLP | Logística | Rango LR |
|---|---|---|---|---|
| annual_inc | 0.281 | 1 | 0.252 | 1 |
| sub_grade_num | 0.225 | 2 | 0.177 | 2 |
| inq_last_6mths | 0.150 | 3 | 0.118 | 4 |
| purpose | 0.127 | 4 | 0.104 | 5 |
| revol_util | 0.109 | 5 | 0.156 | 3 |
| loan_amnt | 0.103 | 6 | 0.088 | 7 |
| open_acc | 0.060 | 7 | 0.053 | 9 |
| dti | 0.052 | 8 | 0.025 | 13 |
| mths_since_last_delinq | 0.051 | 9 | 0.056 | 8 |
| credit_history_months | 0.051 | 10 | 0.021 | 15 |
| emp_length | 0.042 | 11 | 0.091 | 6 |
| revol_limit | 0.036 | 12 | 0.012 | 19 |
| delinq_2yrs | 0.026 | 13 | 0.017 | 17 |
| home_ownership | 0.019 | 14 | 0.027 | 11 |
| verification_status | 0.013 | 15 | 0.026 | 12 |
| total_acc | 0.007 | 16 | 0.012 | 18 |
| never_delinquent | 0.005 | 17 | 0.040 | 10 |
| pub_rec | 0.004 | 18 | 0.021 | 14 |
| term | 0.004 | 19 | 0.018 | 16 |

## 6. Scorecard (desde la logística)

Puntaje base 600 con odds 7:1 y PDO 20: score = 543.85 + 28.85 · ln(odds buenos/malos). AUC en prueba con el puntaje: 0.6380.

| Banda (1 = más riesgo) | Créditos | Puntaje mín. | Puntaje máx. | PD media | Tasa observada |
|---|---|---|---|---|---|
| 1 | 1514 | 545.5 | 583.1 | 0.251 | 0.229 |
| 2 | 1514 | 583.1 | 589.8 | 0.185 | 0.194 |
| 3 | 1513 | 589.8 | 594.5 | 0.158 | 0.172 |
| 4 | 1514 | 594.5 | 598.5 | 0.139 | 0.155 |
| 5 | 1513 | 598.5 | 602.3 | 0.124 | 0.137 |
| 6 | 1514 | 602.3 | 606.0 | 0.110 | 0.116 |
| 7 | 1513 | 606.0 | 610.1 | 0.098 | 0.109 |
| 8 | 1514 | 610.1 | 614.9 | 0.085 | 0.080 |
| 9 | 1513 | 614.9 | 621.8 | 0.071 | 0.062 |
| 10 | 1514 | 621.8 | 661.1 | 0.051 | 0.054 |

## 7. MLP: búsqueda y configuración elegida

Método: búsqueda aleatoria (25 configuraciones, 463 s); 30 semillas (499 s). Parámetros entrenables: 13,185. Configuración elegida: `{"hidden": [128, 64], "activation": "elu", "dropout": 0.1, "l2": 1.328e-07, "optimizer": "rmsprop", "lr": 0.0008925903, "momentum": 0.9, "batch_size": 512, "max_epochs": 100, "patience": 8, "seed": 0}`. Logística: C = 0.1 entre 5 valores.

| Capas | Activación | Dropout | L2 | Optimizador | Tasa | Lote | Mejor época | Log-loss val. | AUC val. |
|---|---|---|---|---|---|---|---|---|---|
| [128, 64] | elu | 0.1 | 1.3e-07 | rmsprop | 8.9e-04 | 512 | 28 | 0.3860 | 0.6725 |
| [64, 32] | relu | 0.0 | 1.4e-04 | adam | 4.3e-03 | 256 | 4 | 0.3866 | 0.6681 |
| [128, 64, 32] | relu | 0.2 | 0.0e+00 | sgd | 8.6e-03 | 256 | 43 | 0.3868 | 0.6684 |
| [64, 32] | leaky_relu | 0.5 | 2.8e-06 | adam | 3.3e-04 | 256 | 29 | 0.3868 | 0.6682 |
| [16] | relu | 0.0 | 1.8e-05 | adam | 1.4e-03 | 1024 | 39 | 0.3870 | 0.6687 |
