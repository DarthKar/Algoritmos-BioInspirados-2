# Parte 1: hipótesis, resultados y conclusiones

Borrador para el reporte. Todas las cifras salen de `results/` y se regeneran con `python run.py`. Las tablas completas, actualizadas en cada corrida, están en `results/pd_resumen.md`.

## 1. Datos

- **Origen:** `loan.csv` (Lending Club, Kaggle). Fecha de descarga: 2026-10-24.
- **Muestra B:** créditos a 36 meses originados entre 2007-06 y 2012-12 que ya habían vencido en el corte (2016-01). Son 74.930 créditos con 57 columnas guardadas; 13,1 % son malos.
- **Partición temporal por fecha de originación:**

| Conjunto | Periodo | Créditos | Tasa de malos |
|---|---|---|---|
| Entrenamiento | 2007-06 a 2012-06 | 46.472 | 12,8 % |
| Validación | 2012 T3 | 13.322 | 14,2 % |
| Prueba | 2012 T4 | 15.136 | 13,1 % |

## 2. Hipótesis del análisis descriptivo

**Método** (`src/pd_model/eda.py`). Las hipótesis se formularon antes de modelar y se contrastaron solo con entrenamiento, para que validación y prueba no influyan en ellas. Para cada variable se calculan tres cosas:
- el AUC univariado, que mide cuánto separa la variable sola a buenos de malos;
- la prueba de Mann-Whitney (α = 0,01);
- un tamaño de efecto mínimo: |AUC − 0,5| ≥ 0,01 para variables continuas, o 1 punto porcentual de diferencia en la tasa de malos para variables sí/no.

Una hipótesis queda *Confirmada* si el efecto es significativo, supera el mínimo y va en la dirección esperada. Las tasas por quintil o grupo se ven en la figura `figures/pd_eda_hipotesis.png`.

| Id | Hipótesis | Tasa de malos: grupo bajo → grupo alto | AUC univariado | Veredicto EDA |
|---|---|---|---|---|
| H1 | Peor subcalificación (`sub_grade_num` alto) → más riesgo | 5,9 % → 21,1 % | 0,640 | Confirmada |
| H2 | Mayor utilización del rotativo (`revol_util`) → más riesgo | 8,8 % → 17,3 % | 0,575 | Confirmada |
| H3 | Mayor ingreso anual → menos riesgo | 17,2 % → 9,2 % | 0,430 | Confirmada |
| H4 | Más consultas en 6 meses → más riesgo | 11,3 % → 21,5 % | 0,574 | Confirmada |
| H5 | Mayor monto del crédito → más riesgo | 13,3 % → 13,1 % | 0,496 | Sin evidencia suficiente |
| H6 | Mayor `dti` → más riesgo | 10,8 % → 14,8 % | 0,536 | Confirmada |
| H7 | Moras o registros públicos previos → más riesgo | 12,1 % → 16,5 % | 0,524 | Confirmada |
| H8 | Historial crediticio más largo → menos riesgo | 14,8 % → 11,1 % | 0,467 | Confirmada |
| H9 | No informar antigüedad laboral → más riesgo | 12,6 % → 19,3 % | 0,509 | Confirmada |
| H10 | Mayor antigüedad laboral → menos riesgo | 13,0 % → 12,7 % | 0,499 | Sin evidencia suficiente |
| H11 | Ingreso verificado → menos riesgo | 13,2 % → 12,4 % | 0,490 | Sin evidencia suficiente |
| H12 | Propósito `small_business` → más riesgo | 12,4 % → 22,6 % | 0,518 | Confirmada |

La variable sola que más discrimina es la subcalificación de Lending Club (AUC 0,64). Le siguen la utilización del rotativo, las consultas recientes y el ingreso. Ninguna supera 0,65, así que se esperaba de antemano que el problema tuviera poca señal.

## 3. Resultados de los modelos (prueba, 2012 T4)

### 3.1 Discriminación y calibración

| Métrica | Tasa base (referencia) | Regresión logística | MLP, media ± d.e. (30 semillas) | MLP, peor / mejor semilla |
|---|---|---|---|---|
| AUC | 0,500 | 0,638 | 0,6396 ± 0,0013 | 0,637 / 0,642 |
| Gini | 0 | 0,276 | 0,279 ± 0,003 | 0,273 / 0,283 |
| KS | 0 | 0,208 | 0,213 ± 0,004 | 0,202 / 0,221 |
| Brier | 0,1137 | 0,1111 | 0,1108 ± 0,0001 | 0,1110 / 0,1107 |
| Log-loss | 0,3880 | 0,3756 | 0,3749 ± 0,0003 | 0,3755 / 0,3744 |
| Parámetros entrenables | | 37 | 15.105 | |
| Configuraciones probadas | | 5 | 25 + 30 semillas | |
| Tiempo total | | 1 s | 816 s (~14 min) | |

- **Diferencia entre modelos:** el bootstrap pareado en prueba da una diferencia de AUC (MLP − LR) de +0,002, con IC 95 % [−0,004; 0,008]. El intervalo incluye el 0, así que no hay diferencia significativa.
- **Calibración global:**
  - La PD media de la logística es 12,8 %, contra una tasa observada de 13,1 %.
  - La del MLP guardado es 13,7 %.
  - Por deciles ambos siguen la diagonal; ver `figures/pd_logistic_calibracion.png` y `figures/pd_mlp_calibracion.png`.

### 3.2 Accuracy, F1 y matriz de confusión

Código en `src/pd_model/classification.py`; tabla completa en la sección 4 de `results/pd_resumen.md`.

El modelo entrega una probabilidad, no una clase. Para calcular accuracy o F1 hay que fijar un umbral y marcar como "malo" a quien tenga PD ≥ umbral.

- **Umbral 0,5.** Casi ninguna PD llega a 0,5, porque la tasa de malos es 13 %. Ambos modelos aprueban a todos y obtienen un accuracy de 86,9 %, idéntico al de aprobar a todos sin modelo, con F1 = 0. Por eso el accuracy no sirve para comparar modelos con clases desbalanceadas.
- **Umbral que maximiza F1, elegido en validación y aplicado sin cambios en prueba:**

| Prueba | Umbral | Accuracy | Precisión | Sensibilidad (malos detectados) | Especificidad (buenos aceptados) | F1 | Créditos rechazados |
|---|---|---|---|---|---|---|---|
| Aprobar a todos | — | 86,9 % | — | 0 % | 100 % | 0 | 0 % |
| Logística | 0,140 | 65,8 % | 19,5 % | 51,6 % | 68,0 % | 0,283 | 34,6 % |
| MLP | 0,153 | 63,6 % | 19,2 % | 55,9 % | 64,7 % | 0,286 | 38,0 % |

La figura `figures/pd_matriz_confusion.png` muestra las matrices de confusión en prueba.

**Cómo leerlo:**
- Para detectar la mitad de los malos hay que rechazar uno de cada tres créditos.
- Dos de cada tres buenos se aceptan.
- De cada cinco rechazados, solo uno habría sido malo.
- El MLP detecta 84 malos más (1.106 frente a 1.022), pero a cambio rechaza 428 buenos más.

El umbral de máximo F1 es una convención. Al inversionista le importa el dinero: el umbral correcto sale de comparar la pérdida esperada (PD × LGD × EAD, de la Parte 2) con el interés que gana. Por eso conviene que la scorecard y la app muestren PD y puntaje, no una etiqueta de "aprobado".

## 4. Hipótesis contra lo que aprendieron los modelos

El EDA mira cada variable por separado. Para el veredicto final se pregunta lo mismo a los modelos, pero controlando el resto de variables (código en `src/pd_model/hypothesis_check.py`; tabla en la sección 2 de `results/pd_resumen.md`):
1. Se toma cada cliente de entrenamiento.
2. Se fija la variable en un valor bajo (P25) y luego en uno alto (P75), sin tocar las demás.
3. Se promedia el cambio de la PD.

Es un efecto *ceteris paribus*, el mismo principio de la dependencia parcial (Friedman, 2001). Un cambio menor de 0,5 puntos porcentuales se considera despreciable. El rango SHAP es la importancia de la variable en el MLP (1 = la más importante; sección 5 de `results/pd_resumen.md`).

| Id | Escenario | EDA (univariado) | Δ PD logística | Δ PD MLP | Rango SHAP (MLP) | Veredicto final |
|---|---|---|---|---|---|---|
| H1 | sub_grade 5 → 13 | Confirmada | +4,1 pp | +6,2 pp | 2 | **Confirmada** |
| H2 | revol_util 28 → 74 % | Confirmada | +3,6 pp | +3,3 pp | 3 | **Confirmada** |
| H3 | ingreso 40k → 80k USD | Confirmada | −4,3 pp | −5,1 pp | 1 | **Confirmada** |
| H4 | consultas 0 → 2 | Confirmada | +2,8 pp | +3,4 pp | 4 | **Confirmada** |
| H5 | monto 5.000 → 13.650 USD | Sin evidencia | +1,5 pp | +1,6 pp | 6 | **Confirmada solo controlando el ingreso** |
| H6 | dti 8,4 → 19,0 | Confirmada | +0,6 pp | +0,1 pp | 9 | Débil: no sobrevive al control |
| H7 | sin moras → 1 mora hace 12 meses | Confirmada | +0,2 pp | +0,2 pp | 12 | **No se confirma al controlar** |
| H8 | historial 105 → 201 meses | Confirmada | +0,0 pp | +0,2 pp | 11 | **No se confirma al controlar** |
| H9 | antigüedad 4 años → no informada | Confirmada | +5,6 pp | +6,4 pp | 7 | **Confirmada** |
| H10 | antigüedad 2 → 9 años | Sin evidencia | +1,7 pp | +2,0 pp | 7 | **Refutada: efecto contrario** |
| H11 | no verificado → verificado | Sin evidencia | −0,8 pp | −0,8 pp | 15 | Confirmada, efecto pequeño |
| H12 | consolidación → pequeño negocio | Confirmada | +9,9 pp | +8,5 pp | 5 | **Confirmada** |

**Lo que cambia entre el EDA y los modelos, y por qué:**
- **H5, el monto.** Por sí solo no discrimina, porque quien pide más suele ganar más: la correlación de Spearman entre monto e ingreso es 0,41 y los dos efectos se cancelan. A igual ingreso, pedir más sí es más riesgoso. Lo confirma la razón monto/ingreso en entrenamiento: la tasa de malos sube de 11,0 % en el quintil más bajo a 16,6 % en el más alto. Es un caso de confusión: el ingreso esconde el efecto del monto.
- **H7, las moras y registros públicos.** Su efecto desaparece porque Lending Club ya los incorpora en la subcalificación. La subcalificación media es 8,9 para quien no tiene moras y 12,4 para quien sí. El inversionista ya recibe esa información a través de `sub_grade`.
- **H8, el historial.** Se explica por el ingreso (correlación 0,35 con el logaritmo del ingreso) y por la subcalificación. Una vez descontados, la antigüedad del historial no aporta.
- **H6, el dti.** Su efecto se reparte con la utilización del rotativo (correlación 0,27) y con la subcalificación. Queda un efecto marginal.
- **H10, la antigüedad laboral.** A igual ingreso y subcalificación, más años de empleo se asocian con algo **más** de riesgo, al revés de lo esperado. Ambos modelos coinciden, así que no es ruido de un solo modelo. Todavía no hay una explicación verificada; queda como hallazgo abierto. Ojo: `emp_length` está truncado en "10+ años", así que un valor de 10 mezcla antigüedades muy distintas.
- **H9, la antigüedad no informada.** Con +6 pp, su efecto es de los más grandes de la tabla, comparable al de pasar de una subcalificación P25 a una P75. Probablemente refleja perfiles sin empleo formal o solicitudes incompletas.

## 5. Conclusiones

1. **El techo es la información, no el modelo.**
   - La logística, el MLP y un gradient boosting de prueba quedan en AUC 0,64 en prueba y 0,66 a 0,67 en validación.
   - La variable sola más fuerte ya da 0,64.
   - Faltan las variables más predictivas del sector: el puntaje FICO no está en el CSV, y la tasa de interés se excluyó por redundante con la subcalificación.
2. **La red no aporta una mejora demostrable sobre la logística.**
   - El MLP gana +0,002 de AUC y −0,0003 de Brier en promedio, pero el intervalo de confianza de la diferencia incluye 0.
   - Cuesta 400 veces más parámetros (15.105 frente a 37) y cientos de veces más cómputo (14 min frente a 1 s en la misma máquina).
   - Pierde la lectura directa de los coeficientes; para interpretarlo hace falta SHAP.
   - Es muy estable entre semillas (d.e. de AUC 0,0013), así que la falta de ventaja no se debe a mala suerte con la semilla.
3. **Los dos modelos cuentan la misma historia.**
   - Las direcciones coinciden en las 12 hipótesis.
   - Las cinco variables más importantes por SHAP son las mismas: ingreso, subcalificación, utilización, consultas y propósito.
   - La no linealidad que podría captar la red no aparece en estos datos.
4. **El análisis univariado engaña en ambos sentidos.**
   - Sobreestima variables que la subcalificación ya resume (moras, historial).
   - Oculta otras que necesitan un control: el monto, que solo pesa a igual ingreso.
   - De las 9 hipótesis que confirmó el EDA, 6 se mantienen al controlar, 2 desaparecen y 1 queda marginal.
5. **El accuracy no sirve aquí.**
   - Con 13 % de malos, aprobar a todos da 86,9 % de accuracy.
   - Cualquier umbral útil baja el accuracy (65 %) a cambio de detectar la mitad de los malos.
   - Para comparar modelos se usan AUC, KS y Brier; el umbral se decide con la pérdida esperada de la Parte 2.
6. **Hay deriva temporal.**
   - El AUC baja de 0,66 a 0,67 en validación (2012 T3) a 0,64 en prueba (2012 T4).
   - Con validación aleatoria se habría reportado un desempeño optimista.

## 6. Reproducir

```bash
python run.py                                               # todo (~40 min)
python run.py --etapas eda clasificacion hipotesis informe  # solo estas etapas, sin reentrenar
```

Las etapas `clasificacion` e `hipotesis` usan los modelos guardados en `models/` y no reentrenan.

## Referencias (verificar formato APA antes del reporte)

- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics, 18*(1), 50–60. https://doi.org/10.1214/aoms/1177730491
