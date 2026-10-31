---
layout: post
title: "Parte 1: metodología experimental"
date: 2026-10-25
---

## 5. Metodología experimental

### 5.1 Regresión logística

Se ajustó (22) para cinco valores de C, de 0,001 a 10, y se eligió el de menor log-loss en validación (Tabla 5). Desde C = 0,01 la diferencia entre valores es menor que 10⁻⁴: la regularización casi no cambia el resultado, porque hay 49.844 observaciones para 37 parámetros. Se eligió **C = 1**.

**Tabla 5. Búsqueda del parámetro de regularización**

| C     | Log-loss en validación | AUC en validación |
|-------|------------------------|-------------------|
| 0,001 | 0,39584                | 0,6382            |
| 0,01  | 0,39424                | 0,6417            |
| 0,1   | 0,39419                | 0,6417            |
| **1** | **0,39416**            | 0,6419            |
| 10    | 0,39419                | 0,6419            |

### 5.2 Perceptrón multicapa

La arquitectura y los hiperparámetros se eligieron por **búsqueda aleatoria**. Con el mismo presupuesto, explora mejor que una rejilla cuando pocos hiperparámetros influyen de verdad: la rejilla repite valores de los importantes, mientras que la aleatoria prueba uno distinto en cada intento. Las capas forman un embudo: cada una tiene la mitad de neuronas que la anterior, con mínimo de 8.

**Tabla 6. Espacio de búsqueda del MLP**

| Hiperparámetro              | Valores                                                                          |
|-----------------------------|----------------------------------------------------------------------------------|
| Capas ocultas               | 1, 2 o 3                                                                         |
| Neuronas en la primera capa | 16, 32, 64, 128 o 256                                                            |
| Activación                  | ReLU, Leaky ReLU, ELU, tanh o GELU                                               |
| Dropout                     | 0; 0,1; 0,2; 0,3 o 0,5                                                           |
| Penalización L2 (λ)         | 0 con probabilidad 0,2; si no, log-uniforme en \[10⁻⁷, 10⁻³\]                    |
| Optimizador                 | Adam, RMSprop o SGD con momento 0,9                                              |
| Tasa de aprendizaje         | log-uniforme en \[10⁻⁴, 10⁻²·³\] para Adam y RMSprop; en \[10⁻³, 10⁻¹\] para SGD |
| Tamaño del minilote         | 256, 512 o 1.024                                                                 |
| Parada temprana             | paciencia de 8 épocas, máximo 100                                                |

Se entrenaron 25 configuraciones y se eligió la de menor log-loss en validación, igual que en la logística. Todas quedaron en un rango estrecho (0,3910 a 0,3948): con estos datos, la arquitectura importa poco.

**Tabla 7. Las cinco mejores configuraciones**

| Capas     | Activación | Dropout | λ        | Optimizador | η      | Lote  | Parámetros | Log-loss val. | AUC val. |
|-----------|------------|---------|----------|-------------|--------|-------|------------|---------------|----------|
| 128-64    | ELU        | 0,1     | 1,3·10⁻⁷ | RMSprop     | 0,0009 | 512   | 13.057     | **0,3910**    | 0,6529   |
| 64-32     | ReLU       | 0       | 1,4·10⁻⁴ | Adam        | 0,0043 | 256   | 4.481      | 0,3918        | 0,6503   |
| 128-64-32 | ReLU       | 0,2     | 0        | SGD         | 0,0086 | 256   | 15.105     | 0,3920        | 0,6482   |
| 16-8      | GELU       | 0,3     | 1,2·10⁻⁶ | RMSprop     | 0,0035 | 1.024 | 737        | 0,3920        | 0,6479   |
| 128-64-32 | GELU       | 0,5     | 1,7·10⁻⁵ | RMSprop     | 0,0006 | 256   | 15.105     | 0,3921        | 0,6474   |

**Configuración final.** Dos capas ocultas (128 y 64 neuronas) con ELU, dropout 0,1, L2 casi nula (λ = 1,3·10⁻⁷), RMSprop, tasa de aprendizaje 0,00089 y minilotes de 512. Se entrenó con 30 semillas (0 a 29), que cambian la inicialización, el orden de los minilotes y las máscaras de dropout. Como modelo final se guardó la corrida con menor log-loss en validación: la semilla 0.

### 5.3 Reglas de comparación justa

1.  **Mismos datos:** variables, preprocesamiento y particiones idénticos, generados por la misma función.
2.  **Presupuesto explícito:** la logística probó 5 configuraciones en 1,0 s; el MLP, 25 configuraciones (327 s) y luego 30 semillas (364 s), en total 693 s, en el computador de un integrante del equipo (solo CPU).
3.  **Variabilidad:** el MLP se reporta con media, desviación estándar, mejor y peor valor de las 30 semillas. La logística es determinista dado C.
4.  **Complejidad:** 37 parámetros en la logística y 13.057 en el MLP.

La diferencia entre modelos se evaluó con un **bootstrap pareado** sobre la prueba (Efron y Tibshirani, 1993): 1.000 remuestras con reemplazo de los créditos de prueba, la diferencia de la métrica (MLP guardado menos logística) en cada una, y un intervalo de confianza del 95 % con los percentiles 2,5 y 97,5.

### 5.4 Reproducibilidad

Todo corre con un solo comando, `python parte1/run.py`, a partir de `parte1/config.yaml`. Con `python parte1/run.py --demo` corre una versión reducida en menos de dos minutos. Las semillas están fijadas y las versiones de las librerías, documentadas.

