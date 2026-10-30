# Parte 1: hipótesis, resultados y conclusiones (resumen)

Resumen corto para el equipo. El desarrollo completo, con figuras, ecuaciones y referencias, está en `reporte_parte1.md` (y su versión Word, `Reporte_Parte1.docx`). Las tablas completas se regeneran en cada corrida en `parte1/salidas/resultados/pd_resumen.md`.

## Datos

- **Origen:** `loan.csv` (Lending Club, Kaggle), descargado el 2026-10-24.
- **Muestra:** créditos cuyo plazo ya había terminado en la fecha de corte (enero de 2016). Son 78.302 créditos, con 13,5 % de malos:
  - 74.930 a 36 meses, originados entre 2007-06 y 2012-12;
  - 3.372 a 60 meses, todos de 2010.
- **Por qué no más créditos a 60 meses:** los de 2011 y 2012 que ya tenían desenlace están sesgados. En esas cohortes seguía vigente entre 24 % y 38 % de los créditos, y entre los ya resueltos la tasa de malos supera en 7 a 14 puntos su cota inferior real.

| Conjunto | Periodo | Créditos | Tasa de malos |
|---|---|---|---|
| Entrenamiento | 2007-06 a 2012-06 | 49.844 (3.372 a 60 meses) | 13,4 % |
| Validación | 2012 T3 | 13.322 | 14,2 % |
| Prueba | 2012 T4 | 15.136 | 13,1 % |

Validación y prueba no tienen créditos a 60 meses, así que el efecto del plazo no se puede validar fuera de tiempo.

## Hipótesis: veredicto final

El EDA (`parte1/analisis/eda.py`) contrasta cada hipótesis en entrenamiento con la prueba de Mann-Whitney y un tamaño de efecto mínimo. El veredicto final (`parte1/analisis/hypothesis_check.py`) mueve cada variable de un valor bajo a uno alto en los modelos ya entrenados, dejando las demás fijas.

| Id | Hipótesis | EDA | Δ PD logística | Δ PD MLP | Veredicto final |
|---|---|---|---|---|---|
| H1 | Peor subcalificación → más riesgo | Confirmada | +4,0 pp | +5,8 pp | Confirmada |
| H2 | Mayor utilización rotativa → más riesgo | Confirmada | +3,7 pp | +3,0 pp | Confirmada |
| H3 | Mayor ingreso → menos riesgo | Confirmada | −4,6 pp | −5,4 pp | Confirmada |
| H4 | Más consultas recientes → más riesgo | Confirmada | +3,0 pp | +4,0 pp | Confirmada |
| H5 | Mayor monto → más riesgo | Sin evidencia | +1,5 pp | +2,6 pp | Confirmada al controlar por el ingreso |
| H6 | Mayor `dti` → más riesgo | Confirmada | +0,4 pp | +0,5 pp | No se confirma al controlar |
| H7 | Moras o registros públicos → más riesgo | Confirmada | +0,1 pp | −0,1 pp | No se confirma al controlar |
| H8 | Historial más largo → menos riesgo | Confirmada | +0,2 pp | −0,4 pp | No se confirma al controlar |
| H9 | Antigüedad laboral no informada → más riesgo | Confirmada | +5,4 pp | +2,8 pp | Confirmada |
| H10 | Mayor antigüedad laboral → menos riesgo | Sin evidencia | +1,7 pp | +1,7 pp | Refutada (efecto contrario) |
| H11 | Ingreso verificado → menos riesgo | Sin evidencia | −0,7 pp | −0,8 pp | Confirmada (efecto pequeño) |
| H12 | Pequeño negocio → más riesgo | Confirmada | +9,8 pp | +8,9 pp | Confirmada |
| H13 | Plazo de 60 meses → más riesgo | Confirmada | +4,5 pp | +7,1 pp | Confirmada, sin validación fuera de tiempo |

De las diez hipótesis que el EDA confirmó, siete se mantienen al controlar por el resto. Las moras, el historial y el `dti` pierden su efecto porque la subcalificación de Lending Club y el ingreso ya los recogen. El monto solo aparece al controlar por el ingreso. La antigüedad laboral va al revés de lo esperado y no tiene explicación verificada.

## Resultados en prueba (2012 T4)

| Métrica | Tasa base | Logística | MLP, media ± d.e. (30 semillas) | MLP guardado |
|---|---|---|---|---|
| AUC | 0,500 | 0,638 | 0,641 ± 0,002 | 0,645 |
| KS | 0 | 0,206 | 0,214 ± 0,005 | 0,220 |
| Brier | 0,1137 | 0,1110 | 0,1111 ± 0,0002 | 0,1106 |
| Parámetros | — | 38 | 13.185 | 13.185 |

- **MLP guardado frente a la logística:** el bootstrap pareado da ΔAUC +0,007 (IC 95 % [+0,002; +0,012]) y ΔBrier −0,0004. Son diferencias significativas pero pequeñas.
- **Media de las 30 semillas:** mejora el AUC en 0,003 y no mejora el Brier.
- **Umbral de máximo F1:** la logística obtiene accuracy 65,9 % y F1 0,282; el MLP, 66,0 % y 0,292. Con umbral 0,5 ambos aprueban a todos y quedan en 86,9 %, lo mismo que aprobar a todos sin modelo.
- **Variables más importantes en ambos modelos:** ingreso, subcalificación, consultas recientes, propósito y utilización rotativa.

## Conclusiones

1. **La red aporta poco:** una mejora pequeña y significativa en discriminación, ninguna en calibración, a cambio de 347 veces más parámetros, unas 500 veces más cómputo y menos interpretabilidad.
2. **El límite es la información disponible.** Un gradient boosting con las mismas variables obtiene AUC 0,634 en prueba, y la subcalificación sola llega a 0,645.
3. **La regresión logística convertida en scorecard es la mejor opción para el caso de uso:** buena calibración, puntos por variable directos y casi el mismo ordenamiento de riesgo que la red.
4. **El efecto del plazo de 60 meses es real pero incierto:** +4,5 pp en la logística y +7,1 pp en el MLP, estimado con una sola cohorte.
5. **Hay deriva temporal.** El AUC baja de 0,66–0,67 en validación a 0,64–0,65 en prueba. Con validación cruzada aleatoria se habría reportado 0,674.
