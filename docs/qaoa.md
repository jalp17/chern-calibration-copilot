# QAOA para MaxCut

Quantum Approximate Optimization Algorithm aplicada al problema de MaxCut
(corte maximo de un grafo). Implementacion original sobre el simulador
statevector de PyTorch/ROCm (`simulador/qsim_torch.py`).

## Modulos

```
simulador/qaoa/
├── hamiltonian_maxcut.py   # operador de coste C = sum (1 - Z_i Z_j)/2
├── qaoa_maxcut.py          # Ansatz QAOA y optimizacion por autograd
├── dataset_qaoa.py         # grafos aleatorios + corte optimo por fuerza bruta
└── __init__.py
```

## Fisica

Para un grafo G = (V, E) con n = |V| qubits:

- **Coste:** C = sum_{(i,j) en E} (1 - Z_i Z_j) / 2. Cada arista contribuye 1
  si los qubits i y j estan en lados distintos del corte, 0 si estan en el mismo
  lado. El valor maximo de C es el corte maximo |E_max|.
- **Mezclador:** B = sum_i X_i.
- **Ansatz:** |psi(gamma, beta)> = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1)
  U_C(gamma_1) |+...+>, con U_C(gamma) = exp(-i gamma C) y U_B(beta) = exp(-i beta B).

Por que es eficiente:
- C es diagonal en la base Z, asi que U_C(gamma) solo multiplica cada amplitud
  por exp(-i gamma * C_diag[idx]). Una multiplicacion elemental, sin matriz.
- B = sum_i X_i implica exp(-i beta X) = RX(2 beta) (convencion estandar
  RX(t) = exp(-i t X / 2)). U_B(beta) es un producto de n puertas 1-qubit,
  sin entrelazamiento.

## Uso basico

```python
from qsim_torch import Simulator
from qaoa import qaoa_maxcut, optimize_qaoa, complete_graph

sim = Simulator(4)
edges = complete_graph(4)          # K4, 6 aristas
params, cut, opt = optimize_qaoa(sim, edges, p=2, n_steps=300)
print(f'corte QAOA={cut}, optimo={opt}')
```

Para un grafo aleatorio:

```python
from qaoa import random_graph, dataset
edges = random_graph(n=5, p_edge=0.5, seed=0)
grafos, cortes = dataset(n_samples=20, n=5)
```

## Convencion de puertas

`exp(-i beta X) = RX(2 beta)`, **no** RY(2 beta). El modulo define su propia
funcion `_rx(theta)` con la convencion estandar de la literatura
(RX(t) = exp(-i t X / 2)). Esto se verifica con un test de kron vs diagonal y
contra una referencia de matriz exponencial completa (`scipy.linalg.expm`).

## Tests

`tests/test_qaoa.py` (7 tests):
- grafo vacio -> corte 0
- K2 -> el optimizador alcanza el corte maximo = 1
- p=0 -> corte = |E/2| (estado |+...+>)
- gradientes finitos (autograd vs diferencias finitas, no NaN)
- maxcut_matrix (kron de Pauli Z) coincide con la version diagonal
- el corte QAOA optimizado nunca supera el optimo por fuerza bruta
- estado inicial normalizado para n par e impar

## Memoria

Para n <= 6 qubits el statevector es 2^n * 16 B (complex128) <= 1 KiB.
Las pruebas locales usan CPU con 2 hilos y consumen < 400 MiB de RSS.
La GPU (W7900D, 48 GiB) no es necessaria para MaxCut con n pequeno, pero el
codigo es nativamente vectorizado por batch: `qaoa_maxcut` acepta
gamma/beta de forma (batch, p) y procesa un lote de grafos en una sola pasada.