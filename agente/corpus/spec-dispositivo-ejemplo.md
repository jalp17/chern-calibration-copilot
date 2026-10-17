# Especificación de dispositivo — ejemplo ilustrativo

> **Ilustrativo.** Valores representativos de un procesador superconductor de
> 6 cúbits de la literatura de 2023–2024. No es un dispositivo real.

## Arquitectura

| Parámetro | Valor |
|---|---|
| Tecnología | transmón superconductor |
| Cúbits | 6 |
| Conectividad | anillo (cada cúbit acoplado a 2 vecinos) |
| Puerta de 1 cúbit | RY, RZ (nativa por rotación de marco) |
| Puerta de 2 cúbits | CNOT |
| Calibración | diaria, con deriva intradía |

## Fidelidades medidas

| Magnitud | Valor | Nota |
|---|---:|---|
| Fidelidad de puerta 1q (promedio) | 0,995 | Ramsey + RB |
| Fidelidad de puerta 1q (peor cúbit) | 0,992 | |
| Fidelidad de puerta CNOT | 0,985 | |
| Fidelidad de lectura | 0,97 | |
| T1 (promedio) | 45 µs | |
| T2 (promedio) | 60 µs | |

## Límites operativos

| Magnitud | Valor | Consecuencia |
|---|---:|---|
| Duración de puerta 1q | 20 ns | |
| Duración de puerta CNOT | 40 ns | |
| Ventana de coherencia útil | ~30 µs | **limita la profundidad** |
| Deriva de frecuencia intradía | ~2 MHz/h | obliga a recalibrar |

**Presupuesto de tiempo por experimento:** con 20 ns por puerta de 1 cúbit,
30 µs de coherencia permiten **~1500 puertas de 1 cúbit** antes de que la
coherencia se agote. Un circuito con `G` capas de `RY+RZ` por cúbit usa
`2·G` puertas de 1 cúbit, luego el límite de coherencia es `G ≤ 750`. En la
práctica el límite real lo pone el ruido, no la coherencia.

## Cómo citarlo

```
spec-dispositivo-ejemplo.md > Fidelidades medidas
spec-dispositivo-ejemplo.md > Límites operativos
```
