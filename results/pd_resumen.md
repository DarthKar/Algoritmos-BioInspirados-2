# Resumen de resultados, Parte 1 (generado automáticamente)

No editar a mano: lo reescribe `python run.py`. Las conclusiones redactadas están en `docs/parte1_eda_y_conclusiones.md`.

## 1. Datos

- Registros: 74,930; variables guardadas: 57; originación 2007-06-01 a 2012-12-01; descarga de Kaggle: 2026-10-24.
- Partición temporal: entrenamiento 46,472 (12.8 % malos), validación 13,322 (14.2 %), prueba 15,136 (13.1 %).

## 2. Hipótesis: EDA univariado y efecto controlado en los modelos

EDA en entrenamiento (Mann-Whitney, α = 0,01, efecto mínimo). Δ PD: cambio medio de la PD al mover la variable de un valor bajo a uno alto dejando las demás fijas (pp = puntos porcentuales). Figura: `figures/pd_eda_hipotesis.png`.

| Id | Hipótesis | Tasa de malos bajo → alto | AUC univ. | EDA | Escenario | Δ PD LR (pp) | Δ PD MLP (pp) | LR | MLP | Rango SHAP |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Peor subcalificación de Lending Club (sub_grade_num alto) -> más riesgo | 5.9 % → 21.1 % | 0.640 | Confirmada | sub_grade_num: P25 5 -> P75 13 | +4.10 | +6.18 | Confirmada | Confirmada | 2 |
| H2 | Mayor utilización del crédito rotativo (revol_util) -> más riesgo | 8.8 % → 17.3 % | 0.575 | Confirmada | revol_util: P25 28 -> P75 74 | +3.57 | +3.26 | Confirmada | Confirmada | 3 |
| H3 | Mayor ingreso anual (annual_inc) -> menos riesgo | 17.2 % → 9.2 % | 0.430 | Confirmada | annual_inc: P25 40000 -> P75 80000 | -4.33 | -5.07 | Confirmada | Confirmada | 1 |
| H4 | Más consultas recientes (inq_last_6mths) -> más riesgo | 11.3 % → 21.5 % | 0.574 | Confirmada | inq_last_6mths: 0 -> 2 | +2.83 | +3.29 | Confirmada | Confirmada | 4 |
| H5 | Mayor monto del crédito (loan_amnt) -> más riesgo | 13.3 % → 13.1 % | 0.496 | Sin evidencia suficiente | loan_amnt: P25 5000 -> P75 13650 | +1.53 | +1.56 | Confirmada | Confirmada | 6 |
| H6 | Mayor endeudamiento sobre ingreso (dti) -> más riesgo | 10.8 % → 14.8 % | 0.536 | Confirmada | dti: P25 8.43 -> P75 18.95 | +0.56 | +0.11 | Confirmada | Efecto despreciable | 9 |
| H7 | Tener moras o registros públicos previos (delinq_2yrs o pub_rec > 0) -> más riesgo | 12.1 % → 16.5 % | 0.524 | Confirmada | sin moras -> 1 mora hace 12 meses | +0.16 | +0.23 | Efecto despreciable | Efecto despreciable | 12 |
| H8 | Historial crediticio más largo (credit_history_months) -> menos riesgo | 14.8 % → 11.1 % | 0.467 | Confirmada | credit_history_months: P25 105 -> P75 201 | +0.04 | +0.15 | Efecto despreciable | Efecto despreciable | 11 |
| H9 | No informar antigüedad laboral (emp_length faltante) -> más riesgo | 12.6 % → 19.3 % | 0.509 | Confirmada | emp_length: mediana 4 -> faltante | +5.56 | +6.35 | Confirmada | Confirmada | 7 |
| H10 | Mayor antigüedad laboral (emp_length) -> menos riesgo | 13.0 % → 12.7 % | 0.499 | Sin evidencia suficiente | emp_length: P25 2 -> P75 9 | +1.72 | +1.95 | Contraria | Contraria | 7 |
| H11 | Ingreso verificado -> menos riesgo | 13.2 % → 12.4 % | 0.490 | Sin evidencia suficiente | Not Verified -> Verified | -0.78 | -0.80 | Confirmada | Confirmada | 15 |
| H12 | Crédito para pequeño negocio (purpose = small_business) -> más riesgo | 12.4 % → 22.6 % | 0.518 | Confirmada | debt_consolidation -> small_business | +9.85 | +8.44 | Confirmada | Confirmada | 5 |

## 3. Logística contra MLP en prueba

MLP: media y desviación de 30 semillas; peor y mejor semilla. Figuras: `figures/pd_*_roc.png`, `figures/pd_*_calibracion.png`, `figures/pd_mlp_perdidas.png`.

| Métrica | Tasa base | Logística | MLP media | MLP d.e. | MLP peor | MLP mejor | MLP guardado |
|---|---|---|---|---|---|---|---|
| auc | 0.5000 | 0.6379 | 0.6396 | 0.0013 | 0.6365 | 0.6418 | 0.6398 |
| gini | 0.0000 | 0.2758 | 0.2791 | 0.0026 | 0.2730 | 0.2837 | 0.2797 |
| ks | 0.0000 | 0.2081 | 0.2133 | 0.0040 | 0.2017 | 0.2195 | 0.2150 |
| brier | 0.1137 | 0.1111 | 0.1108 | 0.0001 | 0.1110 | 0.1107 | 0.1109 |
| log_loss | 0.3880 | 0.3756 | 0.3749 | 0.0003 | 0.3755 | 0.3744 | 0.3751 |

Bootstrap pareado en prueba (1000 réplicas), MLP guardado menos logística: ΔAUC +0.0019 (IC 95 % -0.0037 a +0.0075); ΔBrier -0.00016 (IC 95 % -0.00053 a +0.00021).

| Costo | Logística | MLP |
|---|---|---|
| parametros | 37 | 15,105 |
| configuraciones_evaluadas | 5 | 25 |
| semillas | 1 | 30 |
| segundos_totales | 1.0757 | 816.41 |
| interpretabilidad | coeficientes y scorecard directa | SHAP (aproximado) |

## 4. Accuracy, F1 y matriz de confusión (prueba)

Umbral 0,5 y umbral que maximiza F1 en validación. Figura: `figures/pd_matriz_confusion.png`.

| Modelo | Umbral | Valor | Accuracy | Precisión | Sensibilidad | Especificidad | F1 | Rechazados | VN | FP | FN | VP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pd_logistic | 0.5 | 0.500 | 0.869 | 1.000 | 0.001 | 1.000 | 0.002 | 0.000 | 13156 | 0 | 1978 | 2 |
| pd_logistic | max F1 en validación | 0.140 | 0.658 | 0.195 | 0.516 | 0.680 | 0.283 | 0.346 | 8941 | 4215 | 958 | 1022 |
| pd_mlp | 0.5 | 0.500 | 0.869 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 13156 | 0 | 1980 | 0 |
| pd_mlp | max F1 en validación | 0.153 | 0.636 | 0.192 | 0.559 | 0.647 | 0.286 | 0.380 | 8513 | 4643 | 874 | 1106 |
| referencia_todos_buenos | ninguno | 1.000 | 0.869 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 13156 | 0 | 1980 | 0 |

## 5. Variables de riesgo (SHAP, |valor| medio sobre el logit)

Figura: `figures/pd_mlp_shap.png`.

| Variable | MLP | Rango MLP | Logística | Rango LR |
|---|---|---|---|---|
| annual_inc | 0.284 | 1 | 0.244 | 1 |
| sub_grade_num | 0.248 | 2 | 0.182 | 2 |
| revol_util | 0.158 | 3 | 0.160 | 3 |
| inq_last_6mths | 0.139 | 4 | 0.123 | 5 |
| purpose | 0.129 | 5 | 0.124 | 4 |
| loan_amnt | 0.082 | 6 | 0.094 | 7 |
| emp_length | 0.080 | 7 | 0.097 | 6 |
| open_acc | 0.047 | 8 | 0.052 | 8 |
| dti | 0.041 | 9 | 0.037 | 9 |
| revol_limit | 0.028 | 10 | 0.025 | 14 |
| credit_history_months | 0.027 | 11 | 0.016 | 15 |
| mths_since_last_delinq | 0.025 | 12 | 0.034 | 10 |
| delinq_2yrs | 0.023 | 13 | 0.027 | 13 |
| home_ownership | 0.020 | 14 | 0.027 | 12 |
| verification_status | 0.019 | 15 | 0.032 | 11 |
| total_acc | 0.011 | 16 | 0.016 | 16 |
| never_delinquent | 0.010 | 17 | 0.011 | 17 |
| pub_rec | 0.004 | 18 | 0.010 | 18 |

## 6. Scorecard (desde la logística)

Puntaje base 600 con odds 7:1 y PDO 20: score = 543.85 + 28.85 · ln(odds buenos/malos). AUC en prueba con el puntaje: 0.6379.

| Banda (1 = más riesgo) | Créditos | Puntaje mín. | Puntaje máx. | PD media | Tasa observada |
|---|---|---|---|---|---|
| 1 | 1514 | 538.4 | 582.6 | 0.257 | 0.229 |
| 2 | 1514 | 582.6 | 589.5 | 0.187 | 0.192 |
| 3 | 1513 | 589.5 | 594.3 | 0.159 | 0.176 |
| 4 | 1514 | 594.3 | 598.5 | 0.139 | 0.147 |
| 5 | 1513 | 598.5 | 602.4 | 0.123 | 0.137 |
| 6 | 1514 | 602.4 | 606.2 | 0.110 | 0.123 |
| 7 | 1513 | 606.2 | 610.4 | 0.097 | 0.104 |
| 8 | 1514 | 610.4 | 615.4 | 0.084 | 0.081 |
| 9 | 1513 | 615.4 | 622.4 | 0.070 | 0.065 |
| 10 | 1514 | 622.4 | 663.7 | 0.050 | 0.053 |

## 7. MLP: búsqueda y configuración elegida

Método: búsqueda aleatoria (25 configuraciones, 301 s); 30 semillas (514 s). Parámetros entrenables: 15,105. Configuración elegida: `{"hidden": [128, 64, 32], "activation": "relu", "dropout": 0.2, "l2": 0.0, "optimizer": "sgd", "lr": 0.0085791055, "momentum": 0.9, "batch_size": 256, "max_epochs": 100, "patience": 8, "seed": 9}`. Logística: C = 10 entre 5 valores.

| Capas | Activación | Dropout | L2 | Optimizador | Tasa | Lote | Mejor época | Log-loss val. | AUC val. |
|---|---|---|---|---|---|---|---|---|---|
| [128, 64, 32] | relu | 0.2 | 0.0e+00 | sgd | 8.6e-03 | 256 | 67 | 0.3859 | 0.6719 |
| [128, 64, 32] | gelu | 0.5 | 1.7e-05 | rmsprop | 5.7e-04 | 256 | 23 | 0.3869 | 0.6698 |
| [64, 32] | relu | 0.0 | 1.4e-04 | adam | 4.3e-03 | 256 | 13 | 0.3870 | 0.6680 |
| [128] | elu | 0.2 | 2.4e-07 | rmsprop | 2.9e-03 | 1024 | 11 | 0.3870 | 0.6679 |
| [256, 128] | relu | 0.3 | 0.0e+00 | adam | 1.1e-04 | 1024 | 55 | 0.3871 | 0.6668 |
