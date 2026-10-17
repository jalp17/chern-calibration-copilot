# Corpus local de especificaciones de dispositivo

Este corpus es lo que el agente lee **por defecto**. Contiene los hechos
externos que **no están en el JSON del barrido** y sin los cuales no hay
recomendación posible: presupuesto de error de puerta, coherencia del equipo,
y criterio de aceptación.

> **Aviso de honestidad:** estos valores son **ilustrativos**, redactados para
> este proyecto. No son mediciones de un dispositivo real y no deben citarse
> como tales. Si el agente tiene acceso a la web, puede sustituirlos por
> especificaciones reales; si no, opera con estos y lo declara.

## Ficheros

| fichero | contenido |
|---|---|
| `spec-dispositivo-ejemplo.md` | dispositivo de 6 cúbits superconductor: fidelidad de puerta, T1/T2, conectividad |
| `presupuesto-y-criterios.md` | presupuesto de error por experimento y criterio de aceptación del invariante |
| `mitigacion.md` | opciones de mitigación con su coste y cuándo aplicarlas |

## Cómo se usan

El agente recibe el resultado del barrido (infidelidad de preparación y tasa
de error topológico) y busca en este corpus:

1. Si la infidelidad medida **supera** el presupuesto de puerta del
   dispositivo, la preparación no es viable y hay que reducirla.
2. Si el dispositivo tiene un límite de profundidad (por coherencia), el
   número de puertas `G` debe reducirse.
3. Las opciones de mitigación se priorizan por coste.

Cada afirmación del plan lleva la **fuente** (fichero + sección) de este
corpus.
