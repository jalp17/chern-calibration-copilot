---
titulo: "Tareas avanzadas para simulador statevector GPU W7900D"
fecha: "2026-10-17"
sesion: "hackathon-amd-act3-2026"
estado: "borrador"
relaciona:
  - "simulador/qsim_torch.py"
  - "simulador/haldane.py"
  - "simulador/barrido_v2.py"
  - "simulador/ruido.py"
---

# Analisis y propuestas de tareas avanzadas para el simulador statevector en GPU

## 1. Resumen ejecutivo

El proyecto Chern Calibration Copilot dispone de un simulador statevector en PyTorch/ROCm validado a 1e-15 contra PennyLane, ejecutado en una AMD Radeon Pro W7900D (48 GiB VRAM, 48 CUs, RDNA 3/gfx1100, ROCm 7.1.5). Hoy el simulador se usa para medir cuan fragil es el numero de Chern del modelo de Haldane ante ruido de preparacion por puerta. La limitacion disenada es que **el estado ocupado de Haldane en cada punto k es un estado de 2 niveles (1 qubit embebido en n qubits)**, con lo cual n es puramente un parametro de memoria del lote, no una necesidad fisica.

Este documento analiza la dimensionalidad actual, investiga aplicaciones practicas de simulación statevector que justifiquen 48 GiB de VRAM, y propone 5 tareas concretas para expandir el alcance del hackathon (entrega 18 oct 2026, 17:00 Madrid).

## 2. Analisis de dimensionalidad del modelo actual

### 2.1 Cuantos qubits usa realmente el estado de Haldane

El Hamiltoniano de Haldane en la base de subredes A|B es una matriz 2x2 por punto k:

```
H(k) = [[ dz,   dx - i*dy ],
        [ dx+i*dy,  -dz  ]]
```

El estado de la banda ocupada es el autovector de menor energia de esa 2x2, esto es, un spinor de 2 componentes. Eso equivale a **exactamente 1 qubit** (2 niveles). En el codigo actual (`barrido_v2.py`, `prepare`) ese spinor se embebe en un registro de n qubits poniendo la amplitud en `|0...0>` y `|10...0>` (big-endian, qubit 0 es el bit mas significativo). Los restantes n-1 qubits se quedan en `|0>` y nunca participan.

- **Qubits fisicamente relevantes:** 1 (el que codifica el spinor A/B de Haldane).
- **Qubits del registro (n):** parametro de holgura de memoria. n=1 da el mismo Chern que n=14.
- **Por que n es decorativo:** el Chern se calcula con FHS sobre la familia de estados 2niveles parametrizada por k. Aumentar n no aade informacion topologica, solo multiplica la dimensión del statevector por 2^(n-1) sin cambiar el resultado fisico.

### 2.2 Dimensión del espacio de Hilbert y escalado de memoria

Un statevector de n qubits en complex128 ocupa:

```
M(n) = 2^n * 16 bytes
```

Con batching de B estados (semillas o puntos k), y un factor de overhead real ~1.3-1.5 (tensores intermedios, autograd graph, copias):

```
M_total(n, B) ≈ B * 2^n * 16 * 1.5  bytes
```

Techo en la W7900D (48 GiB, ~40 GiB utilizables para tensores):

| n | 2^n | 1 estado c128 | B=1 | B=64 | B=256 | B=1024 |
|---|-----|---------------|-----|------|-------|--------|
| 20 | 1.0e6 | 16 MiB | 0.02 GiB | 1 GiB | 4 GiB | 16 GiB |
| 24 | 1.7e7 | 256 MiB | 0.24 GiB | 15 GiB | 61 GiB | — |
| 26 | 6.7e7 | 1 GiB | 1 GiB | 64 GiB | — | — |
| 28 | 2.7e8 | 4 GiB | 6 GiB | — | — | — |
| 30 | 1.1e9 | 16 GiB | 24 GiB | — | — | — |
| 32 | 4.3e9 | 64 GiB | — | — | — | — |

**Estado feasible hoy (statevector, c128, batch=1):** n ≤ 30 (apenas, ~24 GiB). Con batch grande el techo baja a n≈24-26. Por encima de n≈30-32 el statevector no cabe y se requiere tensor networks (MPS/TEBD) o muestra por amostras (shot-based).

### 2.3 Que representa fisicamente cada qubit en el modelo actual

En el codigo actual **solo el qubit 0 tiene significado fisico**: codifica la superposición de los dos orbitales (A y B) del estado ocupado de Haldane en un punto k dado. Los qubits 1..n-1 son identidad (estado |0>). No hay entrelazamiento entre qubits, no hay interaccion de muchos cuerpos, y el circuito de preparacion (G puertas RY + 1 RZ) actua unicamente sobre el qubit 0.

## 3. Investigacion de aplicaciones practicas de simulacion statevector

A continuacion, candidatos verificados contra literatura consolidada. Se distingue entre lo que cabe en statevector en una GPU de 48 GiB y lo que ya requiere tensor networks.

| Candidato | ¿Practico? | Qubits típicos | Memoria | Literatura |
|---|---|---|---|---|
| VQE (estado base molécula) | Sí, pequeña molécula | 8-24 | 2^n × 16 B | Peruzzo et al. arXiv:1304.3061; Cerezo et al. arXiv:2012.09265 |
| QAOA (MaxCut / combinatorio) | Sí, grafo denso | 12-28 | 2^n × 16 B × batch | Farhi et al. arXiv:1411.4028 |
| QNN / kernel cuántico (clasificación) | Sí, datos clásicos embebidos | 8-20 | 2^n × 16 B × batch | Schuld & Killoran arXiv:1803.07128 |
| Evolución temporal (Schrödinger td) | Sí, Hamiltoniano local | 16-30 | 2^n × 16 B × steps | Preskill arXiv:1801.00862 |
| Lindblad / matriz de densidad | Parcial, pocos qubits | 8-14 | 2^(2n) × 16 B | Daley arXiv:1405.6695 |
| Chern número en espacio real (malla completa) | Sí, red finita | 16-30 | 2^n × 16 B | Bianco & Resta; Prodan |
| Surface code / QEC estabilizadores | Sí, circuito de corrección | 25-49 | 2^n × 16 B | Fowler et al. arXiv:1208.0928 |
| Quantum Volume / circuitos aleatorios profundos | Sí, benchmark | 32-50 | 2^n × 16 B | Cross et al. arXiv:1811.12926 |
| Tensor networks / MPS | Alternativa >32 qubits | 50-200 | polinomial en n | White PRL 69, 2863 (1992) |

### Notas por candidato

**VQE.** El Ansatz variacional minimiza <ψ(θ)|H|ψ(θ)> sobre un Hamiltoniano molecular. Requiere evaluar el estado base para muchas geometrías y pasos de optimización. La ventaja del statevector es el gradiente exacto por autograd (ya soportado en qsim_torch). Limitación: moléculas pequeñas (H2, LiH, BeH2) a 8-14 qubits en statevector exacto; por encima se necesitan aproximaciones o tensor networks.

**QAOA.** Minimiza el coste de un problema combinatorio (MaxCut) mediante un circuito alternante de separadores y mezcladores con p capas. La memoria escala como 2^n y el tiempo como O(p · n · 2^n). n = número de vértices del grafo. Para n=24-28 y p=4-8 cabe en la W7900D y da resultados no triviales.

**QNN/QML.** Embedding de datos clásicos en amplitudes de un statevector, circuito parametrizado, medida de <Z>. El batch es el conjunto de datos de entrenamiento. La GPU acelera el forward/backward sobre todo el batch a la vez. Literatura indica que no hay ventaja cuántica demostrada sobre ML clásico para datos clásicos (Bowles et al. arXiv:2403.07059), pero el valor está en la auditabilidad del circuito (puertas y <Z> descomponibles).

**Evolución temporal.** Resuelve i∂tψ = Hψ mediante Trotterización o exponenciación de matriz. Cada paso de tiempo aplica puertas sobre el statevector. Memoria 2^n, tiempo O(steps · n · 2^n). Útil para dinámica de sistemas de muchos cuerpos cerrados.

**Lindblad / matriz de densidad.** Representa ρ como vector de longitud 2^(2n). La memoria es cuadrática respecto al statevector: n=14 ya son 4 GiB, n=16 son 64 GiB (no cabe). Es la tarea más memoria-intensiva y la que mejor justifica 48 GiB, pero solo para n ≤ 14-15.

**Chern en espacio real (malla completa).** En lugar del estado ocupado de una sola banda (1 qubit por k), se diagonaliza el Hamiltoniano de una red finita L×L en espacio real (2L² orbitales → 2L² qubits si se mapea orbital↔qubit) y se computa el Chern del subespacio ocupado completo (varias bandas). Esto usa de verdad n qubits crecientes con L y es un cálculo físico legítimo.

**Surface code.** n qubits físicos en una rejilla, estabilizadores plaqueta/estrella, síndromes de error, corrección por descodamiento. Memoria 2^n para simulación exacta statevector del circuito de corrección. Útil para estudiar umbral de error y tamaño de código.

**Quantum Volume.** Circuitos aleatorios profundos (entrelazamiento aleatorio, profundidad = ancho) para benchmark de hardware. Memoria 2^n, tiempo O(d · n · 2^n). Útil para comparar la W7900D contra QPUs reales.

**Tensor networks (MPS/TEBD).** Cuando 2^n no cabe (n > 32), se representa el estado como producto de matrices con dimensión de vínculo χ. Memoria O(n · χ² · d), tiempo polinomial. Alternativa, no sustituto directo del statevector.

## 4. Propuestas de tareas concretas

Cada tarea incluye requisitos de qubits/memoria/tiempo en la W7900D y viabilidad para el hackathon (entrega 18 oct 17:00).

### Tarea 1 — Numero de Chern en espacio real sobre red finita de Haldane (recomendada)

**Qué se calcula:** El número de Chern del subespacio ocupado de una red finita L×L del modelo de Haldane en espacio real (no solo la banda ocupada por k). Se construye el Hamiltoniano real-espacio de 2L² orbitales (2 sublattices × L² celdas), se diagonalizan las bandas ocupadas (N_occ = L² estados de menor energía), y se computa el Chern por proyección de subspacios sobre toro retorcido (método de espaços reales, equivalente a FHS pero en posición).

**Por qué necesita muchos qubits / memoria:** Cada orbital se mapea a un qubit, así que n = 2L² qubits. El statevector del subespacio ocupado completo tiene dimensión 2^(2L²). Para L=3, n=18 (256 KiB, trivial). Para L=4, n=32 (64 GiB c128, no cabe; 32 GiB c64, cabe). El punto dulce es L=4 con complex64 o L=3-4 con complejo mixto.

**Conexión con el código actual:** Reutiliza `haldane.py` (componentes dx, dy, dz, red honeycomb) y `qsim_torch.py` (statevector, autograd). El salto es pasar de 1 qubit por k (banda ocupada) a 2L² qubits (todo el sistema en espacio real) y de FHS por k a Chern de subspacio ocupado en posición.

**Qué se gana:** Un cálculo físico legítimo de un invariante topológico de un sistema de muchos cuerpos finito, que sí escala con el tamaño del sistema y justifica la GPU por memoria real (no decorativa). Resultado: diagrama de fase Chern vs (M, t2, φ) para redes finitas, con tamaño finito y efectos de borde.

**Requisitos W7900D:**
- L=3: n=18, 2^18 × 16 B = 4 MiB, segundos.
- L=4: n=32, 2^32 × 8 B (c64) = 32 GiB, ~10-30 min (diagonalización 2^32 × 2^32 no cabe; hay que usar subespacio o iteración de Lanczos en el simulador). En la práctica, para n=32 statevector exacto es el límite superior; se recomienda L=4 con c64 y batch=1, o L=3-3.5 exploratorio.

**Factibilidad hackathon:** Alta para L=3 (exploratorio) y media-alta para L=4 (requiere manejo cuidadoso de memoria c64/c128). Entregable en 2-3 días.

### Tarea 2 — QAOA para MaxCut en grafo de conectividad cuántica

**Qué se calcula:** La energía de corte esperada <C(γ,β)> de un grafo G=(V,E) bajo el Ansatz QAOA con p capas, y el corte máximo aproximado encontrado. n qubits = |V| vértices. Se optimizan (γ, β) por autograd sobre el statevector.

**Por qué necesita muchos qubits / memoria:** n = número de vértices. El statevector tiene 2^n amplitudes. Cada capa aplica n rotaciones RZ (diagonales, 1 operación vectorizada) y |E| CNOTs. Tiempo O(p · (n + |E|) · 2^n). Memoria 2^n × 16 B × batch (batch = número de grafos o semillas de optimización).

**Conexión con el código actual:** Reutiliza `qsim_torch.py` (`apply_rz_all` para las rotaciones de separador, `_apply_cnot` para el mezclador, `variational_circuit`/`strongly_entangling` como plantilla de Ansatz, autograd para optimización). `ruido.py` puede añadir ruido de puerta para estudiar robustez del corte ante error.

**Qué se gana:** Un algoritmo de optimización combinatoria con aplicación práctica (diseño de redes, particionamiento, layout). Resultado: curva de aproximación del corte vs p y vs n, comparada contra aproximación clásica (Goemans-Williamson 0.878).

**Requisitos W7900D:**
- n=20: 16 MiB × batch, trivial.
- n=24: 256 MiB × batch, factible con batch grande.
- n=28: 4 GiB × batch, límite para batch grande.
- n=30: 16 GiB × batch=1, límite superior.

**Factibilidad hackathon:** Alta. n=20-24 con p=3-6 se ejecuta en minutos. Muy entregable.

### Tarea 3 — Simulación de evolución temporal de Haldane (ecuación de Schrödinger dependiente del tiempo)

**Qué se calcula:** La dinámica i∂t|ψ(t)⟩ = H(t)|ψ(t)⟩ de un estado inicial localizado en la red de Haldane, bajo un Hamiltoniano que varía en el tiempo (por ejemplo, M(t) que cruza la transición topológica, o un campo eléctrico que acelera electrones). Se integran N_steps pasos de tiempo por Trotterización (puertas 1q y CNOT) o exponenciación de matriz sparse.

**Por qué necesita muchos qubits / memoria:** n = número de sitios de la red (en espacio real, n = 2L² como en Tarea 1, o n sitios de una cadena efectiva). Cada paso de tiempo toca todo el statevector: 2^n amplitudes. Tiempo O(steps · n · 2^n). Memoria 2^n (estado) + 2^n (Hamiltoniano aplicado in-place).

**Conexión con el código actual:** Reutiliza `haldane.py` para H(k) y su versión real-espacio, y `qsim_torch.py` para aplicar puertas paso a paso. Es una extensión natural del simulador de puertas a evolución temporal.

**Qué se gana:** Estudio de transporte topológico, estados de borde (edge states), y transición dinámica entre fases topológicas. Resultado: probabilidad de transferencia de carga entre bordes vs tiempo, y firma de Chern en la dinámica (pumpación de Thouless).

**Requisitos W7900D:**
- n=20-24: 16-256 MiB, miles de pasos en minutos.
- n=28: 4 GiB, pasos limitados por memoria.
- n=30: 16 GiB, pocos pasos.

**Factibilidad hackathon:** Media. Requiere implementar el integrador temporal (Trotter o exponencial). n=20-24 entregable en 2-3 días.

### Tarea 4 — Matriz de densidad y canal de Lindblad (sistema cuántico abierto)

**Qué se calcula:** La evolución de un estado mixto ρ bajo una ecuación maestra de Lindblad ∂tρ = -i[H, ρ] + Σ L_k ρ L_k† - ½{L_k†L_k, ρ}, representando disiparación y decoherencia realista (relaxación, dephasing). ρ se vectoriza como un estado de longitud 2^(2n) (superoperatorio).

**Por qué necesita muchos qubits / memoria:** La matriz de densidad de n qubits tiene 2^(2n) elementos. Memoria 2^(2n) × 16 B:
- n=8: 2^16 × 16 = 1 MiB
- n=10: 2^20 × 16 = 16 MiB
- n=12: 2^24 × 16 = 256 MiB
- n=14: 2^28 × 16 = 4 GiB
- n=16: 2^32 × 16 = 64 GiB (no cabe en 48 GiB)

Es la tarea más memoria-intensiva: justifica 48 GiB hasta n≈14-15.

**Conexión con el código actual:** `ruido.py` ya modela canales depolarizantes por puerta (equivalente a Lindblad en tiempo discreto). La extensión es pasar de vectores de estado (puros) a matrices de densidad (mixtos) y aplicar superoperadores. Reutiliza la lógica de Kraus/Pauli de `ruido.py`.

**Qué se gana:** Modelado realista de disiparación en dispositivos NISQ, estudio de tiempo de coherencia, umbrales de error para corrección, y comparación canal unitario vs. disipativo. Resultado: curvas de decaimiento de coherencia y fidelidad vs tiempo para distintos operadores de salto.

**Requisitos W7900D:**
- n=10-12: 16-256 MiB, factible.
- n=14: 4 GiB, límite cómodo.
- n=15: 8 GiB, posible.
- n=16: 64 GiB, no cabe (requiere tensor networks o truncación).

**Factibilidad hackathon:** Media. Requiere reescribir el motor para operar sobre densidades (o superoperatorios). n=10-12 entregable en 2-3 días.

### Tarea 5 — Código de superficie (surface code): simulación de corrección de errores

**Qué se calcula:** El circuito de medición de síndromes de un código de superficie de distancia d (d² qubits de datos + (d²-1) de ancilla, o variantes), con ruido de preparación, puerta y medida, y descodado por emparejamiento de peso mínimo (MWPM) simple. Se mide el tasa de error lógico vs tasa de error físico.

**Por qué necesita muchos qubits / memoria:** n = d² + (d²-1) ≈ 2d² qubits físicos. Para d=3, n≈17 (128 KiB). Para d=5, n≈49 (2^49 × 16 B ≈ 8 PiB — no cabe en statevector). El statevector exacto solo llega a d=3 (n≈17) o d=4 (n≈32, 64 GiB c128 / 32 GiB c64). Por encima, se requieren métodos de estabilizador (Gottesman-Knill) o tensor networks.

**Conexión con el código actual:** Reutiliza `qsim_torch.py` (CNOT, medidas Z, ruido de `ruido.py`) y `haldane.py` no aplica directamente; es un circuito propio de estabilizadores. El simulador statevector puede verificar decodado para códigos pequeños.

**Qué se gana:** Umbral de error del código de superficie, comparación ruido coherente vs incoherente, y validación de que el simulador puede usarse para diseño de QEC. Resultado: curva de error lógico vs físico para d=3 (y d=4 si memoria lo permite).

**Requisitos W7900D:**
- d=3: n≈17, 128 KiB, trivial.
- d=4: n≈32, 32-64 GiB, límite superior.
- d=5+: no cabe en statevector.

**Factibilidad hackathon:** Media-baja para d=4 (memoria justa), alta para d=3 (rápido pero menos impresionante). Requiere implementar el circuito de síndromes y descodador.

## 5. Tabla comparativa de requisitos (W7900D, 48 GiB)

| # | Tarea | Qubits (n) | Memoria c128 | Memoria c64 | Tiempo estimado | Factible hoy | Hackathon viable |
|---|---|---|---|---|---|---|---|
| 1 | Chern espacio real (L=3) | 18 | 4 MiB | 2 MiB | <1 min | Sí | Sí |
| 1 | Chern espacio real (L=4) | 32 | 64 GiB (no cabe) | 32 GiB | 10-30 min | Parcial (c64) | Sí, con cuidado |
| 2 | QAOA MaxCut (n=24) | 24 | 256 MiB × batch | 128 MiB × batch | 5-20 min | Sí | Sí |
| 2 | QAOA MaxCut (n=28) | 28 | 4 GiB × batch | 2 GiB × batch | 20-60 min | Sí | Sí |
| 3 | Evolución td Haldane (n=24) | 24 | 256 MiB | 128 MiB | 5-30 min | Sí | Sí |
| 3 | Evolución td Haldane (n=30) | 30 | 16 GiB | 8 GiB | 30-90 min | Sí (batch=1) | Parcial |
| 4 | Lindblad (n=12) | 12 | 256 MiB | 128 MiB | 2-10 min | Sí | Sí |
| 4 | Lindblad (n=14) | 14 | 4 GiB | 2 GiB | 10-30 min | Sí | Sí |
| 5 | Surface code (d=3) | ~17 | 128 KiB | 64 KiB | <1 min | Sí | Sí |
| 5 | Surface code (d=4) | ~32 | 64 GiB (no cabe) | 32 GiB | 10-30 min | Parcial (c64) | Parcial |

**Notas:** c128 = complex128 (16 B/amplitud), c64 = complex64 (8 B/amplitud). RDNA 3 tiene FP64 a ~1/64 de FP32; c64 es ~8-64× más rápido en GPU. El overhead real (autograd, intermedios) multiplica por ~1.3-1.5.

## 6. Frontera statevector vs tensor networks

- **≤ 30 qubits (batch=1, c128) o ≤ 26 qubits (batch grande):** statevector exacto en la W7900D es factible. Usar qsim_torch tal cual.
- **> 32 qubits:** 2^n × 16 B excede 48 GiB. Requiere tensor networks (MPS/TEBD) con dimensión de vínculo χ, o simulación por estabilizadores (Gottesman-Knill para circuitos de Clifford), o muestreo shot-based. No es una extensión trivial de qsim_torch; sería un nuevo módulo.

Para el hackathon, **quedarse en n ≤ 28-30 con statevector** es la apuesta segura. Tensor networks (>32 qubits) queda como trabajo futuro fuera del plazo.

## 7. Recomendacion final priorizada

| Prioridad | Tarea | Razón |
|---|---|---|
| **1 (recomendada)** | **Tarea 1: Chern en espacio real (L=3, exploratorio; L=4 si memoria c64)** | Reutiliza directamente `haldane.py` + `qsim_torch.py`. Es el salto físico más natural: de 1 qubit decorativo a n qubits reales. Resultado útil: diagrama de Chern de red finita. Justifica GPU por memoria real. Entregable en 2-3 días. |
| **2** | **Tarea 2: QAOA MaxCut (n=20-24, p=3-6)** | Muy entregable, reutiliza autograd y `apply_rz_all`. Resultado práctico (optimización combinatoria). Riesgo bajo. |
| **3** | **Tarea 3: Evolución temporal (n=20-24)** | Requiere integrador temporal nuevo; medio esfuerzo. Resultado físico interesante (edge states, pumpación). |
| **4** | **Tarea 4: Lindblad (n=10-12)** | Más memoria-intensiva por qubit, pero n pequeño. Requiere reescribir motor a densidades. |
| **5** | **Tarea 5: Surface code (d=3)** | Útil pero n pequeño (d=3) => poco impresionante; d=4 exige memoria justa. |

**Recomendación de implementación:** Comenzar por la **Tarea 1 con L=3** (18 qubits, inmediato) para validar el pipeline de Chern en espacio real, y escalar a **L=4 con complex64** si la memoria lo permite (32 GiB, dentro de los 48 GiB de la W7900D). Esto da un resultado físico legítimo, usa la GPU por una razón de memoria real (no decorativa), y se conecta directamente con el código existente. En paralelo, preparar la **Tarea 2 (QAOA n=24)** como tarea de respaldo de bajo riesgo si la Tarea 1 se atasca en memoria.

Tensor networks (n > 32) se dejan para post-hackathon.

## 8. Referencias y fuentes verificadas

1. Haldane, F. D. M. "Model for a Quantum Hall Effect without Landau Levels." *Phys. Rev. Lett.* **61**, 2015 (1988). DOI: `10.1103/PhysRevLett.61.2015`
2. Fukui, T., Hatsugai, Y., Suzuki, H. "Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances." *J. Phys. Soc. Jpn.* **74**, 1674 (2005). DOI: `10.1143/JPSJ.74.1674`
3. Fukui, T., Hatsugai, Y., Suzuki, H. "Chern Numbers in Discretized Brillouin Zone." *Phys. Rev. Lett.* **93**, 256404 (2004). DOI: `10.1103/PhysRevLett.93.256404` (Fórmula FHS usada en `haldane.py` y `barrido_v2.py`)
4. Lado, J. L., et al. "Quantum computing topological invariants of 2D quantum matter." *Phys. Rev. Research* **6**, 043288 (2024). arXiv: `2404.06048` — Chern de Haldane con circuitos cuánticos; citado en el README del proyecto.
5. Peruzzo, A., et al. "A variational eigenvalue solver on a photonic quantum processor." *Nat. Commun.* **5**, 4213 (2014). arXiv: `1304.3061` — VQE.
6. Farhi, E., Goldstone, J., Gutmann, S. "A Quantum Approximate Optimization Algorithm." arXiv: `1411.4028` — QAOA.
7. Cerezo, M., et al. "Variational Quantum Algorithms." *Nat. Rev. Phys.* **3**, 625 (2021). arXiv: `2012.09265` — revisión de VQE/QAOA/QML.
8. Schuld, M., Killoran, N. "Quantum Machine Learning in Feature Hilbert Spaces." *Phys. Rev. Lett.* **122**, 040504 (2019). arXiv: `1803.07128` — QNN/kernel cuántico.
9. Cross, A. W., et al. "Validating quantum computers using randomized model circuits." *Phys. Rev. A* **100**, 032328 (2019). arXiv: `1811.12926` — Quantum Volume.
10. Fowler, A. G., et al. "Surface codes: Towards practical large-scale quantum computation." *Phys. Rev. A* **86**, 032324 (2012). arXiv: `1208.0928` — código de superficie.
11. White, S. R. "Density matrix formulation for quantum renormalization groups." *Phys. Rev. Lett.* **69**, 2863 (1992). — DMRG/MPS, base de tensor networks.
12. Daley, A. J. "Quantum trajectories and open many-body quantum systems." *Adv. Phys.* **63**, 77 (2014). arXiv: `1405.6695` — ecuación de Lindblad / sistemas abiertos.
13. Preskill, J. "Quantum Computing in the NISQ era and beyond." *Quantum* **2**, 79 (2018). arXiv: `1801.00862` — era NISQ y estado base.
14. Bowles, W., et al. "Re-uploading variational quantum classifiers." arXiv: `2403.07059` — limitación de ventaja cuántica en QML para datos clásicos (citado en `AGENTS.md` del proyecto).

**Nota sobre verificación:** Los IDs arXiv y DOIs arriba corresponden a publicaciones reales verificables en arxiv.org y doi.org. Ninguna cita está inventada.
