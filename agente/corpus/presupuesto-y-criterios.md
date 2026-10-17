# Presupuesto de error y criterio de aceptación

> **Ilustrativo.** Cifras coherentes con la especificación de dispositivo del
> corpus. No son mediciones reales.

## Presupuesto de error por experimento

La infidelidad total de una preparación de `G+1` puertas ruidosas es, para el
modelo `shared` (Pauli fijo por puerta), aproximadamente `(1-p/2)^(G+1)`:

| p por puerta | G+1 puertas | infidelidad total | ¿viable? |
|---:|---:|---:|---|
| 0,005 | 5 | 0,012 | sí |
| 0,010 | 5 | 0,025 | sí |
| 0,040 | 5 | 0,096 | en el límite |
| 0,100 | 5 | 0,226 | no |

**Regla operativa:** la infidelidad total de preparación debe mantenerse por
debajo de **0,10** para que el invariante topológico sea fiable.

## Criterio de aceptación del invariante

El invariante (número de Chern) se acepta cuando:

1. La **tasa de acierto** supera el **95 %** en el conjunto de semillas.
2. La preparación ideal (sin ruido) reproduce el valor exacto del modelo.
3. El Chern sale **entero** (FHS es válido sobre un bundle coherente).

> Si la tasa de acierto está entre 50 % y 95 %, el invariante es **indeciso**:
> no se puede afirmar ni negar la fase topológica con una sola medición.

## Traducción a decisiones

| Tasa de acierto | Decisión |
|---|---|
| ≥ 95 % | Aceptar. El invariante es fiable. |
| 50 – 95 % | Indeciso. Reducir `G`, mejorar `p`, o repetir con más semillas. |
| < 50 % | Rechazar. El ruido destruye el bundle; no hay señal. |

## Cómo citarlo

```
presupuesto-y-criterios.md > Presupuesto de error por experimento
presupuesto-y-criterios.md > Criterio de aceptación del invariante
presupuesto-y-criterios.md > Traducción a decisiones
```
