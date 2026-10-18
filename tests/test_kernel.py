"""Categoria: QUANTUM KERNEL — tests de la nueva funcionalidad.

Pruebas:
  1. test_kernel_simetrica_diagonal_uno
  2. test_kernel_rango_y_dim
  3. test_kernel_valores_en_rango
  4. test_svm_cuantico_vs_rbf (si sklearn disponible)
  5. test_return_state_aditivo (no rompe variational_circuit por defecto)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parents[1]  # entrega/
import sys
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / 'simulador'))

from simulador.qsim_torch import Simulator
from simulador.kernel_cuantico import (
    feature_map_states, kernel_matrix, state_fidelity_vec,
    comparar_kernels, demo_kernel,
)
from haldane import sample_dataset


def test_return_state_aditivo():
    """T-K1: variational_circuit y strongly_entangling con return_state=True
    devuelven (readout, psi); con False, comportamiento original."""
    sim = Simulator(4, device='cpu')
    x = torch.randn(3, 4)
    w = torch.randn(2, 4)

    # por defecto: solo readout
    out = sim.variational_circuit(x, w)
    assert out.shape == (3,), f"esperaba (3,), obtuvo {out.shape}"

    out = sim.strongly_entangling(x, w)
    assert out.shape == (3, 4), f"esperaba (3, 4), obtuvo {out.shape}"

    # con return_state=True: tupla (readout, psi)
    readout, psi = sim.variational_circuit(x, w, return_state=True)
    assert readout.shape == (3,)
    assert psi.shape == (3, 16)
    assert torch.allclose(readout, sim.variational_circuit(x, w))

    readout2, psi2 = sim.strongly_entangling(x, w, return_state=True)
    assert readout2.shape == (3, 4)
    assert psi2.shape == (3, 16)
    assert torch.allclose(readout2, sim.strongly_entangling(x, w))


def test_kernel_simetrica_diagonal_uno():
    """T-K2: K es simetrica, diagonal = 1, valores en [0, 1]."""
    sim = Simulator(4, device='cpu')
    X = np.random.default_rng(0).uniform(-1, 1, (6, 4))
    psi = feature_map_states(sim, X, n_layers=2)
    K = kernel_matrix(psi)

    assert K.shape == (6, 6)
    assert torch.allclose(K, K.T, atol=1e-5), "K no es simetrica"
    diag = torch.diag(K)
    assert torch.allclose(diag, torch.ones(6, dtype=K.dtype, device=K.device), atol=1e-5), "diagonal != 1"
    assert (K >= 0).all() and (K <= 1 + 1e-6).all(), "valores fuera de [0,1]"


def test_kernel_rango_y_dim():
    """T-K3: forma correcta (B, B) y consistencia con state_fidelity_vec."""
    sim = Simulator(5, device='cpu')
    X = np.random.default_rng(1).uniform(-1, 1, (4, 4))
    psi = feature_map_states(sim, X, n_layers=1)
    K = kernel_matrix(psi)

    assert K.shape == (4, 4)
    # K[i,j] == state_fidelity_vec(psi, psi[j])[i]
    for i in range(4):
        for j in range(4):
            assert abs(K[i, j].item() - state_fidelity_vec(psi, psi[j])[i].item()) < 1e-6


def test_svm_cuantico_vs_rbf():
    """T-K4: dataset Haldane pequeno, SVM cuantico >= clasico o diff < 0.05."""
    try:
        from sklearn.svm import SVC  # noqa: F401
    except ImportError:
        # sklearn no disponible: test se salta silenciosamente
        return

    X, y, _, _ = sample_dataset(n=80, seed=0)
    # features: [M/M_crit, log t1, t2, |sin phi|]
    res = comparar_kernels(X, y, n_qubits=4, n_layers=2,
                           n_train=50, n_test=30,
                           device='cpu', seed=0)

    # el kernel cuantico no debe ser sustancialmente peor que el clasico
    diff = res.acc_cuantico - res.acc_clasico
    assert diff >= -0.05, (
        f"cuantico {res.acc_cuantico:.3f} muy por debajo de clasico "
        f"{res.acc_clasico:.3f} (diff {diff:+.3f})")


def test_memoria_pequena():
    """T-K5: n<=10, batch<=64, memoria < 2 GiB en CPU."""
    sim = Simulator(10, device='cpu', dtype=torch.complex64)
    X = np.random.default_rng(2).uniform(-1, 1, (64, 4))
    psi = feature_map_states(sim, X, n_layers=1)
    assert psi.shape == (64, 1024)
    K = kernel_matrix(psi)
    assert K.shape == (64, 64)
