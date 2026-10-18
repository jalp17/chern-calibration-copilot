"""
Estados iniciales para la evolución temporal de Haldane.

Cada punto k de la malla de Brillouin es un qubit (nivel A|B). Un estado
localizado en el espacio real en la posición r0 es, en la representación de
producto sobre k,

    |psi> = (1/sqrt(Nk)) sum_k exp(-i k . r0) |k, sublattice>

es decir, un producto sobre k de estados de un qubit preparables con una
puerta RZ (fase) sobre |0> (sublattice A) o |1> (sublattice B).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qsim_torch import Simulator, H, X, rotation_z
from .hamiltonian_td import honeycomb


def estado_producto_A(sim: Simulator) -> torch.Tensor:
    """Todos los k en |0> (sublattice A en todo punto k). Estado |0...0>."""
    return sim.zeros(1)


def estado_superposicion_uniforme(sim: Simulator) -> torch.Tensor:
    """Todos los k en |+> = (|0> + |1>)/sqrt(2). Hadamard en cada qubit."""
    psi = sim.zeros(1)
    Ht = H.to(sim.device).to(sim.dtype)
    for q in range(sim.n_qubits):
        psi = sim._apply_1q(psi, Ht, q)
    return psi


def estado_localizado(sim: Simulator, r0_index: int, net: dict | None = None,
                      sublattice: int = 0, N_mesh: int | None = None) -> torch.Tensor:
    """Estado localizado en el sitio real de índice r0_index.

    |psi> = producto sobre k de RZ(-k . r0) |sublattice>.

    r0_index: índice lineal de la celda (0..N_mesh^2-1); se mapea a una
    posición real r0 = (i, j) de la celda en la red honeycomb.
    sublattice: 0 = A (|0>), 1 = B (|1>).
    """
    net = net or honeycomb()
    nq = sim.n_qubits
    if N_mesh is None:
        N_mesh = int(round(math.sqrt(nq)))
    i = r0_index // N_mesh
    j = r0_index % N_mesh
    r0 = i * net["a1"] + j * net["a2"]

    N = N_mesh
    u = (np.arange(N) + 0.5) / N
    v = (np.arange(N) + 0.5) / N
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ks = (uu[..., None] * net["G1"] + vv[..., None] * net["G2"]).reshape(N * N, 2)

    psi = sim.zeros(1)
    if sublattice == 1:
        Xt = X.to(sim.device).to(sim.dtype)
        for q in range(nq):
            psi = sim._apply_1q(psi, Xt, q)
    for q in range(nq):
        k = ks[q]
        fase = -float(np.dot(k, r0))
        gz = rotation_z(torch.tensor([fase], dtype=torch.float64,
                                     device=sim.device), sim.dtype)
        psi = sim._apply_1q(psi, gz, q)
    return psi
