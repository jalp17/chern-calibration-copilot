# Evolución temporal de Haldane

Ecuación de Schrödinger dependiente del tiempo, `i ∂_t |ψ(t)⟩ = H(t) |ψ(t)⟩`, para el modelo de Haldane. Implementación original sobre el simulador statevector de PyTorch/ROCm (`simulador/qsim_torch.py`).

## Modulos

```
simulador/evolucion_temporal/
├── hamiltonian_td.py    # H(k, t) = d(k,t)·sigma con M(t) parametrico (rampa/seno/ciclico)
├── evolucion_td.py     # Integrador Trotter 1er orden (exacto: [H(k),H(k')]=0) + expm para validacion
├── estado_inicial.py    # |0...0>, |+...+>, estado localizado en sitio o borde
├── diagnosticos.py      # Norma, poblacion de sublattice, transferencia de carga, polarizacion de Berry, Chern dinamico
└── __init__.py          # Exporta todo
```

## Física

Cada punto `k` de la malla de Brillouin es un sistema de dos niveles (sublattice A|B), es decir, exactamente **1 qubit**. Los distintos `k` actúan sobre qubits distintos, luego `[H(k), H(k')] = 0` y la evolución total se factoriza en puertas de 1 qubit por `k`. Un paso de Trotter de 1er orden,

```
U(dt) = prod_k exp(-i H(k) dt),
```

es **exacto** (error de Trotter = 0). No se requieren CNOT entre sectores `k`.

`H(k, t) = [[dz, dx - i dy], [dx + i dy, -dz]]` con

```
dx = t1 * sum_i cos(k . d_i)
dy = t1 * sum_i sin(k . d_i)
dz = M(t) - 2 t2 sin(phi) * sum_i sin(k . b_i)
```

y `M_c = 3√3 · t2 · |sin φ|` (masa crítica donde se cierra el hueco en el punto de Dirac).

Modos de `M(t)` (clase `HaldaneTD`):
- **rampa**: lineal de `M_start` a `M_end` en `[0, T]`.
- **seno**: interpolación suave `cos` entre `M_start` y `M_end`.
- **ciclico**: `M = M_c sin(2π t / T)` (ciclo de bombeo de Thouless).

## Uso básico

```python
from qsim_torch import Simulator
from evolucion_temporal import (
    HaldaneTD, EvolucionTD, estado_producto_A,
    norma, transferencia_carga, chern_dinamico,
)

net = __import__('haldane').honeycomb()   # o from haldane import honeycomb
ham = HaldaneTD(M_start=-1.5, M_end=1.5, T=1.0, modo="rampa")
sim = Simulator(n_qubits=4)               # 1 qubit por punto k (N_mesh = 2)
evol = EvolucionTD(sim, ham, dt=0.01)

psi0 = estado_producto_A(sim)             # |0...0>
psi, historial = evol.evolucionar(psi0, t0=0.0, t1=1.0, n_steps=100,
                                  guardar_historial=True)

print("norma final:", norma(psi).item())                       # ~ 1
print("transferencia:", transferencia_carga(psi, sim).item())  # poblacion B total
print("chern dinamico:", chern_dinamico(historial, sim, evol.k_points))
```

## Convención de qubits

El qubit 0 es el factor más significativo (MSB), coherente con `qsim_torch` (`embed` empieza por el qubit 0 a la izquierda). La malla de puntos `k` se genera con `indexing="ij"` sobre `(u + 0.5)/N`, `u ∈ [0, N)`, en el paralelogramo de la zona de Brillouin generado por `G1, G2` de `haldane.honeycomb()`.

## Diagnosticos

- `norma(psi)`: `||ψ||` por lote. Estado normalizado → 1.
- `poblacion_sublattice(psi, sim, q)`: `(p_A, p_B)` para el qubit `q`.
- `transferencia_carga(psi, sim, q_borde=None)`: población total de sublattice B (o de un qubit-borde).
- `polarizacion_berry(psi, sim, k_points)`: polarización de Berry simplificada (teoría moderna, producto sobre `k`).
- `chern_dinamico(psi_hist, sim, k_points)`: `P_final − P_inicial` sobre el historial (aprox. carga bombeada; versión simplificada del flujo de Berry sobre el toro `(k_x, t)`).

## Tests

`tests/test_evolucion_temporal.py` (6 pruebas), CPU, `n <= 4` qubits, memoria < 2 GiB, sin red:

- `test_norma_conservada` — `||ψ|| = 1` tras 200 pasos con rampa.
- `test_evolucion_unitaria_H_constante` — `H` constante: Trotter coincide con `scipy.linalg.expm` (error ~1e-16).
- `test_paso_pequeno_lineal` — `dt` pequeño → evolución ≈ identidad, lineal en `dt`.
- `test_estado_localizado_h0` — `dt = 0`: estado localizado invariante exactamente.
- `test_chern_dinamico_rampa` — Chern dinámico finito y acotado en `(-1, 1)`; igual a `P_final − P_inicial`.
- `test_transferencia_carga` — transferencia finita en `[0, n_qubits]`; `dt = 0` la conserva.

Ejecutar:

```
cd entrega/
python3 tests/run_all.py evolucion_temporal
# o todo:
python3 tests/run_all.py
```

Resultado esperado: **60 pasados, 0 fallados** (54 previos + 6 nuevos).

## Memoria

Statevector de `2^n · 16 B` (complex128). Para `n = 4` (N_mesh = 2) son 256 amplitudes = 4 KiB. Las pruebas locales usan CPU con 2 hilos y consumen < 100 MiB de RSS.

## GPU (W7900D, RDNA 3, ROCm)

El código acepta `device` del `Simulator`, por lo que es portable a `device="cuda"` sin cambios lógicos (el statevector y las puertas se allocan en `sim.device`). No se ha ejecutado en la W7900D todavía: no se lanza el notebook de AMD hasta estar listo, porque la cuota de GPU (3 h/día/usuario) corre desde que se abre la sesión.

## Notas de implementación

Se corrigieron las importaciones intra-paquete de `evolucion_temporal/` (importos absolutos de hermanos → relativos `from .modulo import ...`), igual que el paquete `qaoa`. No se modificó `simulador/qsim_torch.py` ni `simulador/test_qsim.py`.

`chern_dinamico` y `polarizacion_berry` son implementaciones simplificadas, no un número de Chern de FHS sobre el toro `(k_x, t)`. Las pruebas verifican finito/acotado/consistencia interna, no igualdad exacta a un entero.
