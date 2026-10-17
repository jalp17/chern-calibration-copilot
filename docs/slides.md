# Chern Calibration Copilot

AMD Developer Hackathon ACT III · Pista 1 · Intelligent Industry

---

## El problema

El número de Chern es un invariante topológico que clasifica fases de materia cuántica. Cuando preparamos el estado con ruido, ¿cuánta infidelidad tolera antes de dejar de ser el valor exacto?

No hay respuesta publicada. Los ordenadores cuánticos actuales operan con fidelidades del 95–99 %, y no se sabe cuándo el invariante se rompe.

---

## Lo que hicimos

Simulador statevector en PyTorch/ROCm, validado contra PennyLane a error 1e-15.

Preparamos el estado de la banda ocupada del modelo de Haldane con ruido depolarizante por puerta y medimos el Chern con el método Fukui-Hatsugai-Suzuki.

Probamos tres modelos de ruido con 256 semillas y barras de error.

---

## Resultado principal

| Modelo de ruido | 1−F | % acierto (Chern exacto) |
|---|---|---|
| `final` — control coherente | 0,42 | **100 %** |
| `shared` — Pauli fijo por puerta | 0,42 | 28 % |
| `indep` — ruido por (puerta, k) | 0,42 | 0 % |

**Hallazgo:** el invariante es exactamente robusto ante error coherente, se degrada suavemente ante error en posiciones fijas de puerta, y se destruye ante ruido incoherente. No hay un umbral único: depende de la estructura del error, no solo de su tamaño.

---

## El agente de calibración

Un agente LLM (Gemma vía AMD Token Factory) lee las especificaciones externas del dispositivo (presupuesto de error, coherencia, criterio de aceptación) y emite un plan de mitigación **con fuentes citadas**. No calcula el Chern: interpreta resultados.

---

## AMD en el producto

Simulación en GPU AMD Radeon Pro W7900D (ROCm 7.15, 48 GiB). Justificada por memoria: 256 semillas × n=14 × malla 12×12 requieren ~45 GiB, que superan los 6 GiB de CPU.

---

## Limitaciones declaradas

- `n=1` da el mismo Chern que `n=14`: el estado de Haldane es 1 cúbit por k.
- No simulamos el efecto Hall cuántico físico: calculamos un invariante.
- Los valores del corpus del agente son ilustrativos.

---

## Demo

`streamlit run demo/app.py` — barra de ruido, modelo de ruido, semillas, curva de acierto y plan del agente.

Repo: github.com/jalp17/chern-calibration-copilot

---

## Referencias

Lado et al., Phys. Rev. Research 6, 043288 (2024) · Fukui et al., JPSJ 74, 1674 (2005) · Haldane, PRL 61, 2015 (1988)
