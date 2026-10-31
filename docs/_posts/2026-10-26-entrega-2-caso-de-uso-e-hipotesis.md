---
layout: post
title: "Entrega 2, parte 2: caso de uso e hipótesis"
date: 2026-10-26
---

## 2. Caso de uso

- **Quién usa el modelo:** un inversionista que compra fracciones de créditos de consumo publicados en Lending Club.
- **Para qué decisión:** fondear o no un crédito publicado, y con qué exposición, según la probabilidad de que el solicitante incumpla.
- **En qué momento:** cuando el crédito aparece publicado, antes de comprometer dinero y antes de que exista cualquier pago.
- **Con qué información:** datos de la solicitud (incluido el plazo) y del historial crediticio. Aunque el inversionista ve la subcalificación (`sub_grade`), el modelo no la usa, ni nada que se conozca después de la originación.

Este caso de uso explica tres decisiones de modelamiento:

- Ninguna evaluación de riesgo de Lending Club (subcalificación, calificación ni tasa) entra al modelo.
- La partición es temporal.
- La salida principal es una probabilidad calibrada, no una etiqueta de aprobado o rechazado: con ella, el inversionista compara la pérdida esperada con el interés que recibiría.

## 3. Análisis descriptivo e hipótesis

La Figura 1 muestra la tasa de malos por trimestre de originación. Hay dos regularidades: las cohortes de 2007 y 2008 tienen tasas mucho más altas (cercanas a la crisis financiera, y con créditos que no cumplían la política de crédito), y desde 2009 la tasa se estabiliza entre 10 % y 15 %. Esa variación en el tiempo es otra razón para usar una partición temporal.

<figure>
<img src="{{ site.baseurl }}/assets/img/figura-1.png" alt="Tasa de malos por trimestre de originación" />
<figcaption>Figura 1. Tasa de malos por trimestre de originación (se omiten trimestres con menos de 100 créditos). Elaboración propia con datos de Lending Club.</figcaption>
</figure>

Antes de modelar formulamos trece hipótesis sobre qué hace más riesgosa a una persona (Tabla 4). Se contrastaron solo con el conjunto de entrenamiento, con tres criterios:

- **Prueba de Mann-Whitney** (Mann y Whitney, 1947): compara la distribución de la variable entre buenos y malos.
- **AUC univariado:** cuánto separa la variable por sí sola a buenos de malos (sección 4.9).
- **Tamaño de efecto mínimo:** \|AUC − 0,5\| ≥ 0,01 para variables continuas, o una diferencia de al menos un punto porcentual en la tasa de malos para variables sí/no.

Una hipótesis se considera *confirmada* si el efecto es significativo (α = 0,01), supera el tamaño mínimo y va en la dirección esperada.

**Tabla 4. Hipótesis y su contraste en entrenamiento**

| Id  | Hipótesis                                              | Tasa de malos: grupo bajo → alto | AUC univariado | Veredicto     |
|-----|--------------------------------------------------------|----------------------------------|----------------|---------------|
| H1  | Peor subcalificación de Lending Club → más riesgo      | 5,9 % → 22,8 %                   | 0,645          | Confirmada    |
| H2  | Mayor utilización del crédito rotativo → más riesgo    | 9,5 % → 18,0 %                   | 0,572          | Confirmada    |
| H3  | Mayor ingreso anual → menos riesgo                     | 18,0 % → 9,8 %                   | 0,432          | Confirmada    |
| H4  | Más consultas de crédito en seis meses → más riesgo    | 11,8 % → 22,5 %                  | 0,576          | Confirmada    |
| H5  | Mayor monto del crédito → más riesgo                   | 14,1 % → 14,0 %                  | 0,499          | Sin evidencia |
| H6  | Mayor endeudamiento sobre ingreso (`dti`) → más riesgo | 11,5 % → 15,3 %                  | 0,534          | Confirmada    |
| H7  | Moras o registros públicos previos → más riesgo        | 12,8 % → 17,4 %                  | 0,525          | Confirmada    |
| H8  | Historial crediticio más largo → menos riesgo          | 15,2 % → 11,9 %                  | 0,472          | Confirmada    |
| H9  | No informar la antigüedad laboral → más riesgo         | 13,2 % → 20,0 %                  | 0,508          | Confirmada    |
| H10 | Mayor antigüedad laboral → menos riesgo                | 13,5 % → 13,4 %                  | 0,501          | Sin evidencia |
| H11 | Ingreso verificado → menos riesgo                      | 13,6 % → 13,3 %                  | 0,497          | Sin evidencia |
| H12 | Crédito para pequeño negocio → más riesgo              | 13,0 % → 23,5 %                  | 0,518          | Confirmada    |
| H13 | Plazo de 60 meses → más riesgo                         | 12,8 % → 22,5 %                  | 0,526          | Confirmada    |

Nota. Para variables continuas, los grupos son el primer y el quinto quintil; para las demás, “no” y “sí”. Un AUC menor que 0,5 indica que valores altos se asocian con menos riesgo. La subcalificación (H1) se contrasta solo de forma descriptiva: no entra a los modelos (sección 1.4). Elaboración propia.

Diez de las trece hipótesis se confirman. Monto del crédito, antigüedad laboral e ingreso verificado no muestran evidencia de efecto.

