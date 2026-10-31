---
layout: post
title: "Entrega 2, parte 5: resultados"
date: 2026-10-29
---

## 6. Resultados

### 6.1 Entrenamiento del MLP

En la semilla guardada, la pérdida de validación baja de 0,397 en la primera época a 0,394 en la quinta, oscila y alcanza su mínimo, 0,391, en la época 25. La de entrenamiento baja de forma continua, de 0,373 a 0,365, y la de prueba se mueve entre 0,377 y 0,380. La red aprende rápido lo que los datos permiten; después las mejoras son pequeñas y ruidosas y la parada temprana corta el entrenamiento. En promedio, la mejor época fue la 18 (entre la 3 y la 36).

<figure>
<img src="{{ site.baseurl }}/assets/img/figura-2.png" alt="Pérdida por época del MLP en entrenamiento, validación y prueba" />
<figcaption>Figura 2. Pérdida (entropía cruzada) por época. Línea gruesa: semilla guardada; tenues: las otras 29. La pérdida de entrenamiento se mide sin dropout ni penalización. Elaboración propia.</figcaption>
</figure>

### 6.2 Discriminación y calibración

**Tabla 8. Métricas en prueba (octubre a diciembre de 2012)**

| Métrica  | Tasa base | Regresión logística | MLP, media ± d.e. (30 semillas) | MLP, peor / mejor | MLP guardado |
|----------|-----------|---------------------|---------------------------------|-------------------|--------------|
| AUC      | 0,500     | 0,622               | 0,626 ± 0,002                   | 0,620 / 0,629     | 0,629        |
| Gini     | 0         | 0,244               | 0,252 ± 0,004                   | 0,241 / 0,258     | 0,258        |
| KS       | 0         | 0,181               | 0,190 ± 0,005                   | 0,178 / 0,196     | 0,195        |
| Brier    | 0,1137    | 0,1116              | 0,1118 ± 0,0002                 | 0,1123 / 0,1114   | 0,1115       |
| Log-loss | 0,3880    | 0,3781              | 0,3783 ± 0,0006                 | 0,3798 / 0,3774   | 0,3775       |

Nota. La tasa base asigna a todos los créditos la tasa de malos de entrenamiento. Elaboración propia.

**Discriminación.** La logística obtiene un AUC de 0,622 y el MLP, en promedio, 0,626; el guardado llega a 0,629. El bootstrap pareado da una diferencia de AUC de +0,007, con IC 95 % de \[+0,0003; +0,013\], y una diferencia de Brier de −0,00003, con IC \[−0,0004; +0,0003\]. **La diferencia de AUC es significativa, aunque pequeña; la de Brier no lo es.**

<figure>
<img src="{{ site.baseurl }}/assets/img/figura-3.png" alt="Curva ROC de la regresión logística" />
<figcaption>Figura 3. Curva ROC de la regresión logística. Validación: AUC 0,642, KS 0,203; prueba: AUC 0,622, KS 0,180.</figcaption>
</figure>

<figure>
<img src="{{ site.baseurl }}/assets/img/figura-4.png" alt="Curva ROC del MLP" />
<figcaption>Figura 4. Curva ROC del MLP (semilla guardada). Validación: AUC 0,653, KS 0,225; prueba: AUC 0,629, KS 0,195.</figcaption>
</figure>

**Calibración** (curvas por deciles, Figuras 5 y 6):

- **Logística:** sigue la diagonal en casi todo el rango. Su PD media en prueba (13,2 %) queda 0,2 puntos por encima de la tasa observada (13,1 %).
- **MLP:** su PD media es 13,9 %, 0,9 puntos por encima. En los dos deciles de mayor riesgo sobreestima: predice 20,7 % y 26,8 % frente a 20,0 % y 21,7 % observados.

Para una scorecard que se usará como probabilidad, la logística está algo mejor calibrada en los extremos.

<figure>
<img src="{{ site.baseurl }}/assets/img/figura-5.png" alt="Calibración de la regresión logística" />
<figcaption>Figura 5. Curva de calibración de la regresión logística. Cada punto es un decil de PD estimada. Elaboración propia.</figcaption>
</figure>

<figure>
<img src="{{ site.baseurl }}/assets/img/figura-6.png" alt="Calibración del MLP" />
<figcaption>Figura 6. Curva de calibración del MLP (semilla guardada). Elaboración propia.</figcaption>
</figure>

### 6.3 Variabilidad entre semillas y costo de cómputo

El MLP es estable: la desviación estándar del AUC es 0,002, pero entre la peor y la mejor semilla hay 0,009. Esa distancia es comparable a la ventaja sobre la logística, así que elegir la corrida por validación importa: con otra semilla la ventaja podría ser menor.

En costo, el MLP requiere 353 veces más parámetros, 30 corridas en lugar de una y unos 12 minutos de cómputo frente a un segundo.

**Tabla 9. Costo y complejidad de cada modelo**

| Concepto                                 | Regresión logística              | MLP                          |
|------------------------------------------|----------------------------------|------------------------------|
| Parámetros entrenables                   | 37                               | 13.057                       |
| Configuraciones evaluadas en la búsqueda | 5                                | 25                           |
| Corridas de la configuración final       | 1                                | 30                           |
| Tiempo total de cómputo                  | 1,0 s                            | 693 s                        |
| Interpretación                           | coeficientes y scorecard directa | puntos por variable con SHAP |

Nota. Tiempos medidos en un mismo computador, solo con CPU. Elaboración propia.

## Conclusión

El MLP separa un poco mejor buenos de malos (+0,007 de AUC, con intervalo que apenas excluye el cero), pero no mejora el Brier, sobreestima el riesgo en los deciles más altos y cuesta mucho más. Con estos datos, la regresión logística ofrece casi el mismo poder de discriminación, mejor calibración en los extremos, interpretación directa y una scorecard sin discretizar variables.

