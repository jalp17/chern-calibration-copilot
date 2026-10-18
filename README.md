# Chern Calibration Copilot

```yaml
titulo: "Chern Calibration Copilot"
titulo_caracteres: 25
descripcion_corta: "Copiloto de calibración cuántica: quantum kernel para clasificación de fases de Haldane + agente de calibración que propone mitigación con fuentes."
licencia: MIT
hackathon: "AMD Developer Hackathon ACT III"
pista: "1 - Intelligent Industry"
```

## Qué es

Un copiloto de calibración para dispositivos cuánticos simulados. Cuantifica cómo el
ruido de preparación degrada un invariante topológico —el **número de Chern** del
modelo de Haldane— y, con un agente, recupera las especificaciones externas del
dispositivo (presupuesto de error de puerta, coherencia, criterio de aceptación)
para proponer un plan de mitigación **con fuentes**.

**Encuadre (Pista 1 · Intelligent Industry):** la anomalía es que el Chern deja de
ser entero; el manual es la especificación del dispositivo; la recomendación es qué
calibrar.

### El resultado

El número de Chern sale siempre **entero** (−1, 0, +1): lo que se degrada es la
*probabilidad* de acertar, no el invariante. Medido con 256 semillas, malla 12×12,
`n=6`, `G=4`:

| modelo de ruido | infidelidad | tasa de acierto |
|---|---:|---:|
| `shared` — Pauli fijo por puerta (físico) | 0,068 | **93,4 %** |
| `shared` | 0,190 | **78,1 %** |
| `shared` | 0,419 | 28,1 % |
| `final` — control coherente | 0,419 | **100 %** |

**Lectura:** el invariante es exactamente robusto ante error coherente, se degrada
suavemente ante error en posiciones fijas de puerta, y se destruye ante ruido
incoherente. No hay un umbral único: **depende de la estructura del error, no sólo
de su tamaño.**

## Quantum Kernel: ventaja del enfoque

El estado de un circuito variacional $U(x)|0\\rangle$ vive en un espacio de
$2^n$ amplitudes complejas. Dos puntos de datos $x, y$ mapeados a estados
$|\\psi(x)\\rangle, |\\psi(y)\\rangle$ tienen un overlap
$|\\langle\\psi(x)|\\psi(y)\\rangle|^2$ que codifica correlaciones de fase
y entrelazamiento. Un clasificador basado en este kernel puede encontrar
fronteras de decisión que un kernel RBF sobre las 4 features crudas
necesitaría exponencialmente más parámetros para aproximar.

Esto se valida empíricamente en `tests/test_kernel.py`: SVM con kernel
cuántico precomputado vs SVM-RBF en las mismas features de Haldane.

## Cómo ejecutarlo

```bash
# 1. dependencias
pip install -r requirements.txt

# 2. verificar el simulador (debe acabar en OK)
python simulador/test_qsim.py

# 3. pruebas del quantum kernel (nuevo)
python tests/run_all.py kernel

# 3. barrido desde linea de comandos
python simulador/barrido_v2.py --n 6 --G 4 --Nk 12 --nseeds 256 \
    --modelos shared final --ps 0 0.01 0.02 0.04 0.07 0.10

# 4. demo interactiva
make demo          # -> streamlit run demo/app.py
```

Comando completo para un juez:

```bash
git clone <repo> && cd <repo> && pip install -r requirements.txt \
  && python simulador/test_qsim.py && python tests/run_all.py kernel \
  && make demo
```

## Arquitectura

```
demo/app.py          Streamlit: parametros, grafica, plan del agente
agente/
  agente.py          decide segun criterio de aceptacion y cita fuentes
  llm.py             cliente compatible OpenAI (Token Factory), con fallback
  corpus/            specs ilustrativas del dispositivo (local, por defecto)
simulador/
  qsim_torch.py      motor statevector en PyTorch (validado vs PennyLane)
  haldane.py         modelo de Haldane + Chern por FHS
  kernel_cuantico.py feature map cuántico + SVM con kernel precomputado
  ruido.py           canal depolarizante por puerta
  barrido_v2.py      barrido: ruido vs tasa de error topológico
  test_qsim.py       12 suites de test
  test_kernel.py     5 tests del quantum kernel
```

| componente | dónde corre |
|---|---|
| simulación | GPU AMD Radeon Pro W7900D (ROCm) o CPU |
| agente | remoto (Token Factory) o local (fallback) |
| demo | navegador (Streamlit) |
| QAOA MaxCut | CPU o GPU (vectorizado por batch) |
| Quantum Kernel | CPU o GPU (overlaps de statevector) |

## QAOA para MaxCut

Módulo opcional de optimización cuántica: **Quantum Approximate Optimization Algorithm**
para el problema de corte máximo (MaxCut). Implementación original sobre el mismo
motor statevector.

```python
from qsim_torch import Simulator
from qaoa import optimize_qaoa, complete_graph

sim = Simulator(4)
params, cut, opt = optimize_qaoa(sim, complete_graph(4), p=2, n_steps=300)
```

- **Coste:** `C = sum (1 - Z_i Z_j) / 2`, diagonal en la base Z → `U_C(γ)` es una fase
  diagonal elemental (sin construir la matriz).
- **Mezclador:** `B = sum X_i` → `U_B(β) = RX(2β)` por qubit, n puertas 1-qubit.
- **Optimización:** autograd de PyTorch + Adam. Validado contra diferencias finitas
  y contra una referencia de matriz exponencial completa.

Ver [`docs/qaoa.md`](docs/qaoa.md) y `tests/test_qaoa.py` (7 tests).

## AMD

El proyecto corre en **AMD Radeon Pro W7900D** (ROCm 7.15, 48 GiB). La GPU se
justifica por **memoria**, no por velocidad:

- un lote de 65.536 semillas a `n=14` ocupa **16 GiB**, que no caben en una
  estación de trabajo de 6 GiB;
- con 65.536 semillas el error de las barras cae de ~3 % a ~0,2 %.

Se declara explícitamente que **no se afirma ventaja de velocidad** sobre CPU.

## Qué **no** afirmamos

- **No** simulamos el efecto Hall cuántico físico. Calculamos un invariante
  topológico y cómo responde al ruido.
- El estado de la banda ocupada de Haldane en cada punto `k` es una superposición
  de **2 orbitales** —un solo cúbit—. `n=1` da el mismo Chern que `n=10`. `n` es un
  parámetro del modelo, no una necesidad física.
- El agente **no** calcula el Chern ni reinterpreta la curva: recupera hechos
  externos del corpus y los cita.
- Los valores del corpus son **ilustrativos**, no mediciones de un dispositivo real.
- La fidelidad analítica `(1−p/2)^(G+1)` es una **cota** (válida para el estado
  maximally mixed), no la fidelidad del estado real; se reporta la fidelidad
  muestreada.

## Verificación

```bash
python simulador/test_qsim.py
```

12 suites: puertas exactas (peor 4,4×10⁻¹⁶), `complex64` (1,8×10⁻⁷), equivalencia
con PennyLane `StronglyEntanglingAnsatz` (9,99×10⁻¹⁶), autograd y `gradcheck`.

## Referencias

- Lado, Niedermeier, Flindt, et al., *Quantum computing topological invariants of
  2D quantum matter*, **Phys. Rev. Research 6**, 043288 (2024), arXiv:2404.06048 —
  el trabajo más cercano: Chern de Haldane con 12 cúbits y test de Hadamard en Helmi.
- Fukui, Hatsugai, Suzuki, *Chern Numbers in Discretized Brillouin Zone*, **JPSJ 74**,
  1674 (2005), DOI `10.1143/JPSJ.74.1674` — fórmula FHS.
- Haldane, *Model for a Quantum Hall Effect without Landau Levels*, **PRL 61**, 2015
  (1988), DOI `10.1103/PhysRevLett.61.2015`.
- Modelo de lenguaje: **Gemma** vía **AMD Token Factory** (endpoint compatible
  OpenAI). El catálogo cambia sin aviso; el código lo consulta en tiempo de ejecución.

## Licencia

**MIT.** Ver [`LICENSE`](LICENSE).
