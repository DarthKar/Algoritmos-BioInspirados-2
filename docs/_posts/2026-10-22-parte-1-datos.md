---
layout: post
title: "Parte 1: los datos de Lending Club"
date: 2026-10-22
---

> Construimos dos modelos de probabilidad de incumplimiento (PD) con créditos de consumo de Lending Club: una regresión logística que se puede convertir en scorecard y un perceptrón multicapa (MLP). Ambos se compararon con los mismos datos, partición temporal y presupuesto explícito. El MLP discrimina un poco mejor (AUC 0,629 frente a 0,622), pero la logística queda mejor calibrada en los extremos y cuesta 1 segundo frente a 693.

## 1. Datos

### 1.1 Fuente

Usamos el conjunto *Credit Risk Analysis* de Kaggle (Ranadeep, s. f.), con los créditos de consumo que Lending Club originó en Estados Unidos.

- **Archivo:** `loan.csv`, descargado el 24 de octubre de 2026.
- **Tamaño:** 887.379 registros y 74 variables.
- **Originación (`issue_d`):** junio de 2007 a diciembre de 2015.

Los significados de las variables vienen del diccionario incluido (`LCDataDictionary.xlsx`). Al compararlo con los datos hallamos tres diferencias:

- La verificación del ingreso se llama `is_inc_v` en el diccionario y `verification_status` en los datos.
- El diccionario describe el puntaje FICO, pero ninguna columna del archivo lo contiene.
- En los créditos de 2007 y 2008, la ausencia de eventos en las variables `mths_since_last_*` aparece como 0 y no como vacío.

### 1.2 Variable objetivo

Se construyó a partir de `loan_status`. Los créditos sin desenlace conocido se excluyen porque su comportamiento final no se observa. Quedan 268.530 créditos con desenlace, de los cuales 58.819 (21,9 %) son malos.

**Tabla 1. Codificación de la variable objetivo**

| Categoría original                                  | Créditos | Codificación  |
|-----------------------------------------------------|----------|---------------|
| Fully Paid                                          | 207.723  | 0 (bueno)     |
| Does not meet the credit policy. Status:Fully Paid  | 1.988    | 0 (bueno)     |
| Charged Off                                         | 45.248   | 1 (malo)      |
| Default                                             | 1.219    | 1 (malo)      |
| Late (31-120 days)                                  | 11.591   | 1 (malo)      |
| Does not meet the credit policy. Status:Charged Off | 761      | 1 (malo)      |
| Current                                             | 601.779  | NA (excluido) |
| Issued                                              | 8.460    | NA (excluido) |
| In Grace Period                                     | 6.253    | NA (excluido) |
| Late (16-30 days)                                   | 2.357    | NA (excluido) |

Nota. Elaboración propia con datos de Lending Club (Ranadeep, s. f.).

### 1.3 Sesgo de censura y selección de la muestra

Excluir los créditos vigentes introduce un sesgo. Con corte en enero de 2016, un crédito de 2015 solo tiene desenlace si terminó antes de tiempo: se pagó por anticipado o cayó en impago muy pronto. Las cohortes recientes quedan representadas solo por sus créditos de desenlace temprano, y su tasa de malos no es comparable con la de las antiguas.

Para evitarlo trabajamos con la **muestra de créditos vencidos**: aquellos cuya fecha de originación más el plazo es anterior a enero de 2016. Así todos tuvieron tiempo de mostrar su desenlace.

- A 36 meses: créditos originados hasta diciembre de 2012.
- A 60 meses: solo los de 2010, porque Lending Club ofrece ese plazo desde mayo de 2010 (3.372 créditos, 22,5 % de malos).

La muestra queda en 78.331 créditos de junio de 2007 a diciembre de 2012. Se eliminaron 29 sin información de buró; quedan **78.302 créditos con 13,5 % de malos**.

### 1.4 Variables

Una variable solo entra al modelo si el inversionista la conoce al decidir (sección 2). Por eso se excluyeron cinco grupos:

1.  **Información posterior a la originación** (pagos, saldos, recuperaciones, fechas del último pago): sería fuga de información.
2.  **Evaluaciones de riesgo de Lending Club:** `sub_grade`, `grade`, `int_rate` e `installment`. Son salidas del modelo de la plataforma o funciones de ellas: usarlas sería alimentar nuestro modelo con otro modelo, y el solicitante no las conoce antes de pedir el crédito.
3.  **Estado de residencia** (`addr_state`): puede ser sustituto de características protegidas y tiene muchas categorías pequeñas.
4.  **Variables que solo existen desde 2012**, como `tot_cur_bal`, `total_rev_hi_lim` e `initial_list_status`.
5.  **Variables con más de 50 % de faltantes** en entrenamiento, o con un solo valor en más del 99,5 % de los casos.

La Tabla 2 lista las 18 variables que usan ambos modelos.

**Tabla 2. Variables de entrada y definición según el diccionario**

| Variable                           | Definición (LCDataDictionary.xlsx)                                                                                                                                                                         | Tratamiento                                                  |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `loan_amnt`                        | “The listed amount of the loan applied for by the borrower.”                                                                                                                                               | Numérica                                                     |
| `term`                             | “The number of payments on the loan. Values are in months and can be either 36 or 60.”                                                                                                                     | Numérica (36 o 60)                                           |
| `annual_inc`                       | “The self-reported annual income provided by the borrower during registration.”                                                                                                                            | ln(1 + x)                                                    |
| `emp_length`                       | “Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.”                                                                        | Numérica, con indicador de faltante                          |
| `dti`                              | “A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.” | Numérica                                                     |
| `delinq_2yrs`                      | “The number of 30+ days past-due incidences of delinquency in the borrower’s credit file for the past 2 years”                                                                                             | Numérica                                                     |
| `inq_last_6mths`                   | “The number of inquiries in past 6 months (excluding auto and mortgage inquiries)”                                                                                                                         | Numérica                                                     |
| `mths_since_last_delinq`           | “The number of months since the borrower’s last delinquency.”                                                                                                                                              | Si nunca hubo mora, se llena con la antigüedad del historial |
| `open_acc`                         | “The number of open credit lines in the borrower’s credit file.”                                                                                                                                           | Numérica                                                     |
| `pub_rec`                          | “Number of derogatory public records”                                                                                                                                                                      | Numérica                                                     |
| `revol_util`                       | “Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.”                                                                               | Recortada en los percentiles 1 y 99                          |
| `total_acc`                        | “The total number of credit lines currently in the borrower’s credit file”                                                                                                                                 | Numérica                                                     |
| `home_ownership`                   | “The home ownership status provided by the borrower during registration.”                                                                                                                                  | Categórica                                                   |
| `verification_status` (`is_inc_v`) | “Indicates if income was verified by LC, not verified, or if the income source was verified”                                                                                                               | Categórica                                                   |
| `purpose`                          | “A category provided by the borrower for the loan request.”                                                                                                                                                | Categórica                                                   |
| `credit_history_months`            | Derivada de `earliest_cr_line` (“The month the borrower’s earliest reported credit line was opened”) e `issue_d`                                                                                           | Meses entre ambas fechas                                     |
| `never_delinquent`                 | Derivada de `mths_since_last_delinq`                                                                                                                                                                       | 1 si nunca registró mora                                     |
| `revol_limit`                      | Derivada de `revol_bal` (“Total credit revolving balance”) y `revol_util`                                                                                                                                  | revol_bal / (revol_util / 100), en ln(1 + x)                 |

Nota. Definiciones citadas de LCDataDictionary.xlsx, incluido en el conjunto de datos (Ranadeep, s. f.).

### 1.5 Limpieza y preprocesamiento

La limpieza fija, que no depende de estadísticas de los datos, hace cuatro cosas: corrige los ceros de 2007 y 2008 que significaban “nunca” en `mths_since_last_*`; crea las variables derivadas de la Tabla 2; agrupa las categorías raras de vivienda (NONE, ANY) en OTHER; y elimina los 29 registros sin información de buró.

El preprocesamiento estadístico se ajusta **solo con el conjunto de entrenamiento** y se aplica igual a validación y prueba:

- **Numéricas:** logaritmo ln(1 + x) para las de cola larga (ingreso y límite rotativo); recorte en los percentiles 1 y 99 para la utilización rotativa; imputación por la mediana con indicador binario de faltante; estandarización a media 0 y desviación 1.
- **Categóricas:** codificación *one-hot* omitiendo la primera categoría como referencia (`MORTGAGE`, `Not Verified` y `car`). Las categorías con menos de 0,5 % de los casos se agrupan.

El resultado es una matriz de 36 columnas, la misma para ambos modelos.

### 1.6 Partición temporal

La partición sigue la fecha de originación. Entrenar con el pasado y evaluar con el futuro reproduce el uso real del modelo y expone la deriva temporal, que una partición aleatoria esconde.

**Tabla 3. Partición temporal de la muestra**

| Conjunto      | Periodo de originación            | Créditos | Tasa de malos |
|---------------|-----------------------------------|----------|---------------|
| Entrenamiento | junio de 2007 a junio de 2012     | 49.844   | 13,4 %        |
| Validación    | julio a septiembre de 2012        | 13.322   | 14,2 %        |
| Prueba        | octubre a diciembre de 2012       | 15.136   | 13,1 %        |
| Total         | junio de 2007 a diciembre de 2012 | 78.302   | 13,5 %        |

Nota. Los 3.372 créditos a 60 meses, todos de 2010, quedan en entrenamiento, donde son el 6,8 %. Elaboración propia.

La validación sirve para elegir hiperparámetros, detener el entrenamiento y fijar umbrales. La prueba se usa una sola vez, para reportar.

