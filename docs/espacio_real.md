# Espacio real — Chern de Haldane en red finita

## Visión general

Módulo para calcular el número de Chern de un modelo de Haldane sobre una red finita L×L con condiciones de frontera periódicas, trabajando en espacio real (matriz Hamiltoniana 2L²×2L²) y aplicando la fórmula FHS para obtener el Chern.

## Módulos

### `hamiltonian_real_space`

Construye el Hamiltoniano de Haldane en espacio real:

```python
from espacio_real import hamiltonian_real_space

H = hamiltonian_real_space(L=3, M=1.0, t1=1.0, t2=0.20, phi=np.pi/2)
# H: matriz (18, 18) complex128 para L=3
```

### `chern_real_space`

Calcula el número de Chern por FHS:

```python
from espacio_real import chern_fhs_real_space

C_int, C_raw = chern_fhs_real_space(H, n_occ=9, L=3, M=1.0)
# C_int = -1 (fase topológica) o 0 (fase trivial)
```

### `dataset_real_space`

Genera dataset etiquetado con Chern exacto:

```python
from espacio_real import dataset_real_space

X, y, chern, theta = dataset_real_space(n_samples=100, L=3, seed=0)
# X: (100, 4) features [M/M_crit, log t1, t2, |sin phi|]
# y: (100,) etiquetas 0 trivial / 1 topológica
```

## Uso básico

```python
import numpy as np
from espacio_real import hamiltonian_real_space, chern_fhs_real_space, dataset_real_space

# 1. Construir Hamiltoniano
L = 3
H = hamiltonian_real_space(L, M=0.5, t1=1.0, t2=0.20, phi=np.pi/2)

# 2. Calcular Chern
C, C_raw = chern_fhs_real_space(H, n_occ=L*L, L=L, M=0.5)
print(f"Chern = {C}")  # -1 (topológico)

# 3. Generar dataset
X, y, chern, theta = dataset_real_space(n_samples=800, L=3, seed=0)
```

## Pruebas

Ejecutar desde `entrega/`:

```bash
python tests/run_all.py espacio_real
```

## GPU (W7900D)

El módulo usa NumPy para la construcción del Hamiltoniano y el cálculo FHS. Para aceleración en GPU, portar a PyTorch con `device='cuda'`. Los cálculos son de pequeña escala (L ≤ 20 → matriz ≤ 800×800) y adecuados para CPU.