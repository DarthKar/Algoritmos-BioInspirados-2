# Caso de uso (Parte 1)

**Quién.** Un inversionista que compra fracciones de créditos de consumo publicados en la plataforma Lending Club.

**Decisión.** Fondear o no un crédito publicado, y con qué exposición, según la probabilidad de que el cliente incumpla (PD).

**Momento.** Cuando el crédito aparece publicado con su calificación, antes de comprometer dinero y antes de conocer cualquier pago.

**Información disponible en ese momento.** Los datos del anuncio y del historial crediticio del solicitante: monto, antigüedad laboral, ingreso, tipo de vivienda, verificación de ingreso, propósito, endeudamiento (dti), moras y consultas recientes, utilización y límite del crédito rotativo, antigüedad del historial y la subcalificación asignada por Lending Club (`sub_grade`). No se usa nada que se conozca después de la originación (pagos, saldos, recuperaciones, fechas de último pago), por ser fuga de información.

**Alcance.** Créditos que ya habían terminado su plazo en la fecha de corte de los datos (enero de 2016):
- a 36 meses, originados entre 2007-06 y 2012-12;
- a 60 meses, originados en 2010. Lending Club ofrece ese plazo desde mayo de 2010, y los posteriores no habían vencido.

El plazo entra como variable. Su efecto se estima con una sola cohorte (2010) y no se puede validar fuera de tiempo, porque no hay créditos a 60 meses vencidos en validación ni en prueba. Las predicciones para 60 meses deben leerse con esa cautela.

**Variable objetivo.** 1 = Charged Off, Default, Late (31-120) o sus equivalentes "Does not meet the credit policy"; 0 = Fully Paid. Los créditos sin desenlace conocido se excluyen.

**Decisiones de variables que se derivan del caso de uso.**
- `sub_grade` entra: es información legítima para el inversionista. `int_rate` e `installment` quedan fuera por ser casi redundantes con `sub_grade` (correlación de 0,97).
- `addr_state` queda fuera (proxy de características protegidas y muchas categorías pequeñas); `initial_list_status` queda fuera (solo existe desde 2012).