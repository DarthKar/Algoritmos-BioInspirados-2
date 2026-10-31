---
layout: post
title: "Cacería de la alucinación"
date: 2026-10-28
---

## Cacería de la alucinación

Durante el trabajo usamos asistentes de IA y contrastamos sus respuestas con los datos, el diccionario y el enunciado. Estas son cinco verificaciones, cada una con el prompt, la respuesta, la sospecha, la evidencia, la corrección y la lección.

### Verificación 1. Definición errónea de `total_rec_prncp` y `out_prncp`

*Tipo: datos del mundo real.*

**1. Prompt.** Según mi propio análisis del dataset, las columnas que consulté tienen estos significados:

- `loan_amnt` = capital requerido de financiamiento por el cliente
- `funded_amnt` = capital efectivamente financiado
- `out_prncp` = valor faltante por financiar hasta la fecha
- `total_rec_prncp` = valor total abonado al monto base financiado incluyendo intereses
- `total_rec_int` = valor total de los intereses abonados

**2. Respuesta.** `out_prncp` se describió como “valor faltante por financiar” y `total_rec_prncp` como “valor total abonado al monto financiado incluyendo intereses”.

**3. Sospecha.** Contradice el diccionario de datos, que define `out_prncp` como capital pendiente y `total_rec_prncp` como capital recibido, con los intereses en `total_rec_int`.

**4. Evidencia.**

- `funded_amnt = total_rec_prncp + out_prncp` se cumple en el 100 % de los créditos *Current* y en el 99,7 % de los *Fully Paid*.
- `total_rec_prncp` nunca supera a `funded_amnt` (0 casos). Si incluyera intereses, los superaría.
- En *Fully Paid*, el 99,7 % tiene `total_rec_prncp` igual a `funded_amnt`, aunque pagaron intereses.
- La igualdad `funded_amnt == total_rec_prncp − total_rec_int` se cumple en 0 de 58.819 incumplidos. Ejemplo: 2.500 contra 456,46 − 435,17 = 21,29.

**5. Corrección.** Usar la definición del diccionario: `total_rec_prncp` es solo capital, y los intereses van aparte.

**6. Lección.** Contrastar la definición de cada columna con el diccionario y con identidades contables entre columnas antes de usarla en una fórmula.

### Verificación 2. Igualdad exacta entre flotantes

*Tipo: código.*

**1. Prompt.** “Primero que todo la equivalencia pagos totales no se cumple.” La captura muestra la comparación que se ejecutó en el notebook:

<img src="{{ site.baseurl }}/assets/img/caceria-verificacion-2.png" alt="Captura del notebook: comparación total_pymnt == total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries, con resultado array([True, False])" />

**2. Respuesta.** La comparación `total_pymnt == total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries` arrojó `[True, False]`, y se tomó como evidencia de que la identidad no se cumple.

**3. Sospecha.** Los `False` no deberían existir si los pagos totales se construyen de esos componentes.

**4. Evidencia.**

- Con `==`: 43.523 `True` y 15.296 `False`.
- En los `False`, la diferencia mediana es de unos 9·10⁻¹³ y la máxima, de 0,01.
- Ejemplo: `total_pymnt = 2270.7` y la suma da `2270.7000000000003`.
- Con `np.isclose(..., atol=0.01)`, el 100 % cumple.

**5. Corrección.** Comparar flotantes con tolerancia, no con `==`.

**6. Lección.** Nunca usar igualdad exacta con números decimales. Mirar el tamaño de la diferencia antes de concluir que una identidad falla.

### Alucinación 3. EAD: “grave error” e invención de lo que pide el profesor

*Tipo: datos del mundo real y conceptos.*

**1. Prompt.** Le mostramos a la IA lo que había dicho otra IA sobre el EAD:

> El profesor pide justificar la Exposición al Incumplimiento (EAD) precisamente porque no corresponde simplemente al préstamo inicial, sino al saldo que el cliente realmente debía en el momento exacto en que dejó de pagar.
>
> Si asumes que la exposición es el monto original del préstamo (`funded_amnt`), estarías cometiendo un error financiero grave: estaríamos ignorando todo el capital que el cliente sí alcanzó a pagar antes de entrar en mora.
>
> Por eso debes definir y justificar. En el caso de créditos a plazos (*installment*), la justificación que espera tu profesor es la siguiente:
>
> La EAD no es estática: un cliente que pide $10,000 y entra en mora en la primera cuota expone al banco a una pérdida mucho mayor que un cliente que pide los mismos $10,000 pero entra en mora en la última cuota.
>
> La definición matemática: la exposición real al momento del incumplimiento se define calculando el monto inicial financiado (`funded_amnt`) menos el capital que ya había sido devuelto por el prestatario hasta la fecha (`total_rec_prncp`).
>
> Al redactar el reporte, justificar la EAD significa explicarle al profesor que entiendes esta dinámica de amortización y que no estás cobrándole al cliente (ni calculando pérdidas sobre) dinero que ya había sido devuelto al banco.

Y preguntamos: cuál es tu opinión y, si tuviéramos que modificar el código, cómo lo modificarías para sacar correctamente el cálculo y que corresponda con la realidad de que un cliente que entra en mora después de la séptima cuota debería significar una menor pérdida que uno que entra en mora en la primera cuota con el mismo monto fondeado.

**2. Respuesta.** La IA afirmó que el profesor “pide justificar la EAD precisamente porque no corresponde simplemente al préstamo inicial” y que usar `funded_amnt` es “un error financiero grave” porque ignora el capital ya pagado.

**3. Sospecha.** El enunciado solo dice “definiendo y justificando la exposición al incumplimiento (EAD) que usan”; no indica qué definición espera. Además, el capital ya pagado se descuenta en el numerador de la pérdida, no en el denominador.

**4. Evidencia.**

- Sobre 46.009 castigados con exposición positiva, la pérdida real suma 474,5 millones.
- La LGD medida sobre `funded_amnt`, multiplicada por `funded_amnt`, da 474,53 millones, es decir, la pérdida real. Usar `funded_amnt` como EAD no ignora lo pagado cuando la LGD está definida sobre la misma base.
- La IA sí acertó en algo: al incumplir, la exposición es `funded_amnt − total_rec_prncp`, y la fracción que sigue expuesta baja de 0,96 (incumplimiento en los primeros 3 meses) a 0,41 (más de 24 meses).

**5. Corrección.** Elegir una definición coherente.

- **Opción A:** LGD sobre `funded_amnt` con EAD = `funded_amnt`.
- **Opción B:** modelar aparte la fracción expuesta *f* y usar EAD = `funded_amnt` × *f̂*.

**6. Lección.** Contrastar lo que una IA dice que pide el profesor con el texto literal del enunciado, y verificar sus afirmaciones financieras con una identidad numérica.

### Verificación 4. Una LGD que se ejecutaba sin errores pero sobreestimaba la pérdida

*Tipo: código y conceptos.*

**1. Prompt.** “Esta es la descripción de lo que hace el código en este momento, ¿podrías darme un conjunto de sugerencias para que esa fracción de pérdida total del monto inicial dado incumplimiento quede bien definida?”

**2. Respuesta.** El código definía `LGD = 1 − rec_net / EAD_obs` y luego calculaba `E[L] = PD × LGD × funded_amnt`. Se ejecutaba sin errores.

**3. Sospecha.** La LGD promedio era de 0,93 y la pérdida esperada parecía alta. Además, la descripción del propio código señalaba que la base económica no era consistente.

**4. Evidencia.**

| Cálculo | Pérdida total |
|---|---|
| Pérdida real | 474,5 millones |
| LGD sobre deuda × EAD_obs | 474,53 millones |
| LGD sobre deuda × `funded_amnt` (el código) | 616,1 millones |

El código sobreestimaba la pérdida en un 29,8 %.

**5. Corrección.** Definir `LGD = (funded_amnt − total_rec_prncp − rec_net) / funded_amnt`, o bien modelar `EAD = funded_amnt × f̂`.

**6. Lección.** Probar una identidad de coherencia: para los incumplidos observados, Σ LGD × EAD debe igualar la pérdida real. Este hallazgo es el más peligroso si pasa desapercibido, porque el código funciona sin errores y el resultado se ve razonable.

### Verificación 5. Conflicto de dependencias en `requirements.txt`

*Tipo: código y librerías.*

**1. Prompt.** “Hola. Junto con algunos compañeros de la universidad debo realizar el trabajo descrito en el archivo docx. A mí me corresponde realizar la aplicación web, así que me gustaría que me digas si al decir que debe estar publicada significa que existe una forma de publicarla mediante GitHub o si crees que el profesor se refiere a otra forma de publicarla. Entiendo que la primera parte del trabajo debe estar completada antes de poder realizar la aplicación web, así que te pido que revises el proyecto para decirme si crees que ya puedo comenzar a trabajar y, de ser así, quiero que me ayudes a desarrollar la aplicación con los requisitos que solicita la descripción del trabajo, potencialmente creándose de tal forma que pueda seguir funcionando aunque mis compañeros cambien o añadan nuevos elementos al proyecto.”

**2. Respuesta.** La IA ofreció un archivo comprimido con los archivos básicos para correr un entorno de desarrollo, que incluía un `requirements.txt` con versiones exactas como `pandas==3.1.0`, `shap==0.54.1` y `streamlit==1.55.0`.

**3. Sospecha.** En un inicio no hubo sospecha visual, pero al intentar la instalación, `pip` arrojó un error de incompatibilidad entre las versiones recomendadas por la IA.

**4. Evidencia.**

<img src="{{ site.baseurl }}/assets/img/caceria-verificacion-5.png" alt="Error de pip: no se puede instalar pandas==3.1.0 porque shap 0.54.1 depende de pandas y streamlit 1.55.0 depende de pandas&lt;3 y &gt;=1.4.0" />

El error de `pip` indica que la librería `shap` 0.54.1 requiere `pandas` 3.1.0, pero `streamlit` 1.55.0 exige versiones de `pandas` anteriores a la 3.0 (`pandas<3` y `>=1.4.0`).

**5. Corrección.** Actualizar `streamlit` a la versión 1.66.1, la cual es compatible con `pandas` 3.1.0.

**6. Lección.** Verificar que las versiones de librerías sugeridas por una IA no generen conflictos de dependencias en el entorno de desarrollo antes de dar por válido un archivo de requisitos.
