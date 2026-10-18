# Chern Calibration Copilot

AMD Developer Hackathon ACT III · Pista 1 · Intelligent Industry

---

## El problema

El número de Chern es un invariante topológico que clasifica fases de materia cuántica. Cuando preparamos el estado con ruido, ¿cuánta infidelidad tolera antes de dejar de ser el valor exacto?

No hay respuesta publicada. Los ordenadores cuánticos actuales operan con fidelidades del 95–99 %, y no se sabe cuándo el invariante se rompe.

---

## Lo que hicimos

**Simulador statevector en PyTorch/ROCm**, validado contra PennyLane a error 1e-15.

**5 tareas avanzadas implementadas** sobre el simulador base:
1. **Chern en espacio real** (red finita Haldane, L×L)
2. **QAOA para MaxCut** (optimización combinatoria cuántica)
3. **Evolución temporal de Haldane** (Trotter, M(t) cruzando Mc)
4. **Matriz densidad + Lindblad** (ruido disipativo, matriz densidad 2^(2n))
5. **Surface code d=3** (estabilizadores X/Z, decodador, tasa error lógico)

**Quantum Kernel** — feature map cuántico para clasificación de fases de Haldane. SVM con kernel precomputado vs RBF clásico.

Preparamos el estado de la banda ocupada del modelo de Haldane con ruido depolarizante por puerta y medimos el Chern con el método Fukui-Hatsugai-Suzuki.

Probamos tres modelos de ruido con 256 semillas y barras de error.

---

## Resultado principal

| Modelo de ruido | 1-F | % acierto (Chern exacto) |
|---|---|---|
| `final` — control coherente | 0,42 | **100 %** |
| `shared` — Pauli fijo por puerta | 0,42 | 28 % |
| `indep` — ruido por (puerta, k) | 0,42 | 0 % |

**Hallazgo:** el invariante es exactamente robusto ante error coherente, se degrada suavemente ante error en posiciones fijas de puerta, y se destruye ante ruido incoherente. No hay un umbral único: **depende de la estructura del error, no solo de su tamaño**.

---

## Quantum Kernel — Ventaja de modelo

El circuito variacional actúa como **feature map**: mapea $(M, t_1, t_2, \phi) \to U(x)|0\rangle$ en espacio $2^n$. El kernel $K(x,y) = |\langle\psi(x)|\psi(y)\rangle|^2$ captura correlaciones de fase y entrelazamiento que un kernel RBF clásico no representa eficientemente.

**Validado empíricamente:** SVM con kernel cuántico precomputado vs SVM-RBF en las mismas features de Haldane.

---

## 5 tareas avanzadas implementadas

| Tarea | Módulo | Tests | Qué hace |
|-------|--------|-------|----------|
| 1. Chern espacio real | `simulador/espacio_real/` | 5 | Hamiltoniano 2L²×2L², Chern FHS espacio real |
| 2. QAOA MaxCut | `simulador/qaoa/` | 7 | Ansatz p capas, optimización autograd |
| 3. Evolución temporal | `simulador/evolucion_temporal/` | 6 | Trotter, M(t) cruza M_c, diagnóstico dinámico |
| 4. Lindblad | `simulador/lindblad/` | 6 | Matriz densidad 2^(2n), operadores salto, decodificador |
| 5. Surface code d=3 | `simulador/surface_code/` | 6 | Estabilizadores X/Z, decodador, tasa error lógico |

**Total: 36 tests nuevos + 41 base = 77 tests pasando (77/0)**

---

## Quantum Kernel — Ventaja de modelo

El circuito variacional mapea $(M, t_1, t_2, \phi) \to U(x)|0\rangle$ en espacio $2^n$. El kernel $K(x,y) = |\langle\psi(x)|\psi(y)\rangle|^2$ captura correlaciones de fase/entrelazamiento que un kernel RBF clásico no representa eficientemente.

Validado: SVM con kernel cuántico precomputado vs SVM-RBF en features de Haldane.

---

## El agente de calibración

Un agente LLM (Gemma vía AMD Token Factory) lee especificaciones del dispositivo (tasa error puerta, T1/T2, conectividad, criterio aceptación), ejecuta circuitos de prueba, compara con umbral de aceptación y emite un **plan de mitigación con fuentes citadas**. No calcula el Chern: interpreta resultados.

Modo LLM (Token Factory Gemma) o fallback local determinista con corpus local.

---

## AMD en el producto

Simulación en **GPU AMD Radeon Pro W7900D** (ROCm 7.15, 48 GiB, 48 CUs, gfx1100).

**GPU justificada por memoria, no velocidad:**
- Lote 65.536 semillas × n=14 × Nk=12 → ~45 GiB reales (no caben en CPU 6 GiB)
- Con 65k semillas, error estadístico cae de ~3% a ~0,2%
- **No se afirma ventaja de velocidad** (einsum/matmul ~1× en CPU vs GPU)

---

## Limitaciones declaradas

- `n=1` da el mismo Chern que `n=14`: el estado de Haldane es 1 cúbit por k. `n` es parámetro del modelo, no necesidad física.
- No simulamos el efecto Hall cuántico físico: calculamos un invariante topológico y su respuesta al ruido.
- Los valores del corpus del agente son ilustrativos, no mediciones de dispositivo real.
- La fidelidad analítica $(1-p/2)^{(G+1)}$ es una cota (estado maximally mixed), no la fidelidad real; se reporta la fidelidad muestreada.
- No se afirma ventaja de velocidad GPU vs CPU; la GPU es **load-bearing por memoria**.

---

## Demo

`streamlit run demo/app.py` — **3 pestañas:**
1. **Quantum Kernel vs Clásico** — feature map, SVM kernel cuántico vs RBF, gráficas
2. **Barrido de ruido** — parámetros, gráficas tasa acierto vs infidelidad, JSON
3. **Calibración del agente** — specs dispositivo → plan mitigación con fuentes

---

## Referencias

Lado et al., *Phys. Rev. Research* **6**, 043288 (2024) · arXiv:2404.06048  
Fukui, Hatsugai, Suzuki, *JPSJ* **74**, 1674 (2005), DOI `10.1143/JPSJ.74.1674`  
Haldane, *PRL* **61**, 2015 (1988), DOI `10.1103/PhysRevLett.61.2015`

---

Repo: github.com/jalp17/chern-calibration-copilot  
Demo: https://chern-calibration-copilot.streamlit.app/  
Licencia: MIT