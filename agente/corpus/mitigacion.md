# Opciones de mitigación

> **Ilustrativo.** Las opciones y su coste son representativos. No son datos de
> un fabricante.

## Opciones ordenadas por coste

### 1. Reducir la profundidad `G` — coste: bajo
Menos puertas ruidosas ⇒ menos infidelidad acumulada. Con `G=2` en vez de `G=4`
la infidelidad cae de `~(1-p/2)^5` a `~(1-p/2)^3`.
**Aplicar cuando:** la tasa de acierto está entre 50 % y 95 % y el circuito
lo permite.

### 2. Reordenar las puertas — coste: bajo
El modelo `shared` muestra que **dónde** cae el Pauli importa: al final
preserva el invariante exactamente; intercalado lo degrada. Mover las puertas
de mayor error fuera de la trayectoria que define el bundle puede recuperar
tasa de acierto sin tocar el hardware.
**Aplicar cuando:** el análisis por puerta muestra una puerta dominante.

### 3. Recalibración diaria — coste: medio
La deriva intradía (~2 MHz/h) degrada `p`. Recalibrar antes de cada serie
reduce `p` de ~0,10 a ~0,02.
**Aplicar cuando:** `p` medido supera el presupuesto.

### 4. Dynamical decoupling — coste: medio
Secuencias de pulsos que cancelan el ruido de baja frecuencia. Reduce `p`
efectivo a costa de puertas extra (más profundidad).
**Aplicar cuando:** el ruido es de baja frecuencia dominante.

### 5. Mitigación de errores por post-procesado — coste: alto
Extrapolación a ruido cero oProbabilistic error cancellation. Requiere
caracterizar el ruido y multiplica el número de ejecuciones.
**Aplicar cuando:** las opciones 1–4 no bastan y el experimento lo justifica.

## Cómo citarlo

```
mitigacion.md > Opciones ordenadas por coste
mitigacion.md > 1. Reducir la profundidad
mitigacion.md > 2. Reordenar las puertas
mitigacion.md > 3. Recalibración diaria
```
