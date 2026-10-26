# Caso de uso (Parte 1)

**Quién.** Un inversionista que compra fracciones de créditos de consumo publicados en la plataforma Lending Club.

**Decisión.** Fondear o no un crédito publicado, y con qué exposición, según la probabilidad de que el cliente incumpla (PD).

**Momento.** Cuando el crédito aparece publicado con su calificación, antes de comprometer dinero y antes de conocer cualquier pago.

**Información disponible en ese momento.** Los datos del anuncio y del historial crediticio del solicitante: monto, antigüedad laboral, ingreso, tipo de vivienda, verificación de ingreso, propósito, endeudamiento (dti), moras y consultas recientes, utilización y límite del crédito rotativo, antigüedad del historial y la subcalificación asignada por Lending Club (`sub_grade`). No se usa nada que se conozca después de la originación (pagos, saldos, recuperaciones, fechas de último pago), por ser fuga de información.

**Alcance.** Créditos a 36 meses originados entre 2007-06 y 2012-12 que ya alcanzaron su desenlace (muestra B). El modelo no aplica a créditos a 60 meses.

**Variable objetivo.** 1 = Charged Off, Default, Late (31-120) o sus equivalentes "Does not meet the credit policy"; 0 = Fully Paid. Los créditos sin desenlace conocido se excluyen.

**Decisiones de variables que se derivan del caso de uso.**
- `sub_grade` entra: es información legítima para el inversionista. `int_rate` e `installment` quedan fuera por ser casi redundantes con `sub_grade` (correlación de 0,97).
- `addr_state` queda fuera (proxy de características protegidas y muchas categorías pequeñas); `initial_list_status` queda fuera (solo existe desde 2012).