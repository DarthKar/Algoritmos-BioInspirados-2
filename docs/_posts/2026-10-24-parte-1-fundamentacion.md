---
layout: post
title: "Parte 1: fundamentación teórica y matemática"
date: 2026-10-24
---

## 4. Fundamentación teórica y matemática

### 4.1 Neurona artificial

Con entradas x ∈ ℝᵈ, una neurona calcula una combinación afín y le aplica una activación no lineal φ, donde w ∈ ℝᵈ son los pesos, b ∈ ℝ el sesgo y a la activación:

```text
z = wᵀx + b,    a = φ(z)                                   (1)
```

### 4.2 Perceptrón multicapa

Un MLP con L capas ocultas organiza neuronas en capas densas. Con h⁽⁰⁾ = x y nₗ neuronas en la capa l:

```text
z⁽ˡ⁾ = W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾,    h⁽ˡ⁾ = φ(z⁽ˡ⁾),    l = 1, …, L     (2)
```

con W⁽ˡ⁾ ∈ ℝ^(nₗ×nₗ₋₁). La capa de salida tiene una neurona cuyo valor es el logit de la PD; la sigmoide lo convierte en probabilidad:

```text
z_sal = w_salᵀ h⁽ᴸ⁾ + b_sal,    p̂ = σ(z_sal) = 1 / (1 + e^(−z_sal))   (3)
```

El número de parámetros entrenables es:

```text
Σ (de l = 1 a L+1) (nₗ₋₁ + 1) · nₗ                         (4)
```

Para la arquitectura final (36 → 128 → 64 → 1): 37·128 + 129·64 + 65·1 = **13.057 parámetros**. La regresión logística es el caso L = 0: una sola neurona con activación sigmoide.

### 4.3 Funciones de activación

La búsqueda consideró cinco activaciones para las capas ocultas; la sigmoide se usa en la salida. Se muestran con sus derivadas, que necesita la retropropagación:

```text
ReLU:        φ(z) = max(0, z)                 φ'(z) = 1[z > 0]                    (5)
Leaky ReLU:  φ(z) = max(αz, z), α = 0,01      φ'(z) = 1 si z > 0; α si no         (6)
ELU (final): φ(z) = z si z > 0; α(eᶻ − 1)      φ'(z) = 1 si z > 0; φ(z) + α       (7)
             si no, con α = 1
tanh:        φ(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ)     φ'(z) = 1 − tanh²(z)                (8)
GELU:        φ(z) = z·Φ(z)                    φ'(z) = Φ(z) + z·ϕ(z)              (9)
             (Φ y ϕ: distribución y densidad de la normal estándar)
Sigmoide:    σ(z) = 1/(1 + e⁻ᶻ)               σ'(z) = σ(z)(1 − σ(z))              (10)
```

### 4.4 Pérdida: entropía cruzada binaria por máxima verosimilitud

Modelamos la etiqueta de cada crédito como Bernoulli, yᵢ \| xᵢ ~ Bernoulli(pᵢ), con pᵢ = p̂(xᵢ; θ). Si las observaciones son independientes, la verosimilitud es:

```text
L(θ) = Π (i = 1..n) pᵢ^yᵢ · (1 − pᵢ)^(1 − yᵢ)               (11)
```

Maximizar L equivale a minimizar el negativo de su logaritmo promediado, es decir, la entropía cruzada binaria (*log-loss*):

```text
l(θ) = −(1/n) Σ [ yᵢ ln pᵢ + (1 − yᵢ) ln(1 − pᵢ) ]           (12)
```

Con pᵢ = σ(zᵢ), la derivada respecto al logit es notablemente simple:

```text
∂lᵢ/∂zᵢ = pᵢ − yᵢ                                           (13)
```

Para evitar desbordamientos, la implementación calcula la pérdida desde el logit: lᵢ = max(zᵢ, 0) − zᵢyᵢ + ln(1 + e^(−\|zᵢ\|)).

### 4.5 Retropropagación

El gradiente se obtiene con la regla de la cadena, de la salida hacia la entrada. Con δ⁽ˡ⁾ = ∂lᵢ/∂z⁽ˡ⁾, en la salida δ_sal = p̂ᵢ − yᵢ, y hacia atrás:

```text
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ φ'(z⁽ˡ⁾)                          (14)

∂J/∂W⁽ˡ⁾ = (1/|B|) Σ δᵢ⁽ˡ⁾ hᵢ⁽ˡ⁻¹⁾ᵀ + 2λW⁽ˡ⁾,   ∂J/∂b⁽ˡ⁾ = (1/|B|) Σ δᵢ⁽ˡ⁾   (15)
```

donde ⊙ es el producto elemento a elemento, B un minilote y W⁽ᴸ⁺¹⁾ = w_salᵀ. El término 2λW⁽ˡ⁾ viene de la penalización L2. Con dropout, h⁽ˡ⁾ incluye la máscara de esa iteración. PyTorch calcula (14) y (15) por diferenciación automática.

### 4.6 Optimizadores

Con gₜ = ∇J en el minilote del paso t y tasa de aprendizaje η, la búsqueda consideró tres reglas, tal como las implementa PyTorch (operaciones elemento a elemento):

```text
SGD con momento (Polyak, 1964), μ = 0,9:
  vₜ = μ vₜ₋₁ + gₜ,    θₜ = θₜ₋₁ − η vₜ                         (16)

RMSprop (Tieleman y Hinton, 2012), α = 0,99, ε = 10⁻⁸ (optimizador final):
  sₜ = α sₜ₋₁ + (1 − α) gₜ²,    θₜ = θₜ₋₁ − η gₜ / (√sₜ + ε)     (17)

Adam (Kingma y Ba, 2015), β₁ = 0,9, β₂ = 0,999, ε = 10⁻⁸:
  mₜ = β₁ mₜ₋₁ + (1 − β₁) gₜ,    vₜ = β₂ vₜ₋₁ + (1 − β₂) gₜ²    (18)
  m̂ₜ = mₜ/(1 − β₁ᵗ),  v̂ₜ = vₜ/(1 − β₂ᵗ),  θₜ = θₜ₋₁ − η m̂ₜ/(√v̂ₜ + ε)   (19)
```

### 4.7 Regularización

**Penalización L2.** Se suma a la pérdida la norma de Frobenius al cuadrado de las matrices de pesos (los sesgos no se penalizan). Encoge los pesos hacia cero y reduce la varianza:

```text
J(θ) = (1/|B|) Σ lᵢ + λ Σₗ ‖W⁽ˡ⁾‖²_F                       (20)
```

**Dropout.** En entrenamiento, cada neurona oculta se apaga con probabilidad q y las activas se reescalan para conservar el valor esperado (Srivastava et al., 2014):

```text
rⱼ ~ Bernoulli(1 − q),    h̃ⱼ = rⱼ hⱼ / (1 − q)                (21)
```

**Parada temprana.** Al final de cada época se calcula la pérdida (12) en validación y se guardan los mejores pesos. El entrenamiento se detiene tras P = 8 épocas sin mejora mayor que 10⁻⁶, o al llegar a 100 épocas, y se restauran los mejores pesos. Así se limita cuánto se alejan los pesos de su valor inicial (Prechelt, 1998).

### 4.8 Regresión logística de referencia

Modela logit(p) = β₀ + βᵀx y se estima minimizando la log-loss con penalización L2, como en scikit-learn:

```text
min (β₀, β)  ½‖β‖₂² + C Σ [ −yᵢ ln pᵢ − (1 − yᵢ) ln(1 − pᵢ) ]    (22)
```

donde C es el inverso de la fuerza de regularización. Como las variables están estandarizadas, la magnitud de los coeficientes es comparable, y e^(βⱼ) es la razón de odds asociada a una desviación estándar de la variable j.

### 4.9 Métricas de evaluación

Sean n₁ malos, n₀ buenos y s la PD estimada.

- **AUC:** área bajo la curva ROC (TPR contra FPR al variar el umbral). Es la probabilidad de que un malo elegido al azar reciba una PD mayor que un bueno elegido al azar (Hanley y McNeil, 1982; Fawcett, 2006), y coincide con el estadístico de Mann-Whitney normalizado.
- **Gini:** reescala del AUC, de 0 (azar) a 1 (separación perfecta).
- **KS:** máxima distancia entre las distribuciones acumuladas de la PD en malos (F₁) y buenos (F₀), equivalente a la máxima diferencia entre TPR y FPR (Thomas et al., 2002).
- **Brier:** error cuadrático medio de la probabilidad (Brier, 1950); mide a la vez discriminación y calibración.

```text
AUC = (1/(n₁n₀)) Σ(i: yᵢ=1) Σ(j: yⱼ=0) ( 1[sᵢ > sⱼ] + ½·1[sᵢ = sⱼ] )   (23)
Gini = 2·AUC − 1                                                         (24)
KS = máx_t |F₁(t) − F₀(t)| = máx_t (TPR(t) − FPR(t))                      (25)
Brier = (1/n) Σ (p̂ᵢ − yᵢ)²                                               (26)
```

### 4.10 Derivación de la scorecard

La scorecard expresa el riesgo como un puntaje en el que más puntos significan menos riesgo. Se define sobre las odds de ser bueno, odds = (1 − p)/p, con una transformación lineal de su logaritmo (Siddiqi, 2006):

```text
puntaje = offset + factor · ln(odds)                                    (29)
```

Los parámetros salen de dos condiciones: las odds de referencia o₀ reciben el puntaje base S₀, y duplicar las odds suma PDO puntos (*points to double the odds*):

```text
S₀ = offset + factor · ln o₀,    S₀ + PDO = offset + factor · ln(2o₀)    (30)
factor = PDO / ln 2,    offset = S₀ − factor · ln o₀                    (31)
```

Para la logística, ln(odds) = −logit(p) = −(β₀ + Σ βⱼx̃ⱼ), con x̃ⱼ = (xⱼ − μⱼ)/sⱼ estandarizada con la media y desviación de entrenamiento. Al sustituir en (29), el puntaje queda como suma de puntos por variable en sus unidades originales:

```text
puntaje = offset − factor · (β₀ − Σ βⱼμⱼ/sⱼ)   +   Σ (−factor·βⱼ/sⱼ) · xⱼ   (32)
          └──────── puntos base ────────┘        └─ puntos por unidad de xⱼ ─┘
```

Cada variable continua aporta una cantidad fija de puntos por unidad y cada categoría aporta una cantidad fija frente a su referencia. La scorecard admite variables continuas sin discretizarlas.

