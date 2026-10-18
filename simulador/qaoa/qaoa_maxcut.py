"""Ansatz QAOA para MaxCut.

Evolution unitaria:
    U_C(gamma) = exp(-i gamma C)   C = sum_{(i,j)} (1 - Z_i Z_j)/2
    U_B(beta)  = exp(-i beta B)    B = sum_i X_i

Estado:
    |psi(gamma,beta)> = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1) U_C(gamma_1) |+...+>

Por que es eficiente:
  · C es diagonal en la base Z -> U_C(gamma) solo multiplica cada amplitud por
    exp(-i gamma * C_diag[idx]). Una multiplicacion elemental, sin matriz.
  · B = sum_i X_i -> exp(-i beta X) = RX(2 beta) (convencion estandar
    RX(t) = exp(-i t X / 2)). Por tanto U_B(beta) es un producto de RX(2 beta_q)
    sobre cada qubit: n puertas 1-qubit, sin entrelazamiento.

NOTA DE CONVENCION: la formula del briefing original decia
"exp(-i beta X) = RY(2 beta)". Eso es incorrecto fisicamente:
exp(-i beta X) = RX(2 beta), no RY(2 beta). Se usa RX aqui.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from qsim_torch import Simulator
from .hamiltonian_maxcut import (
    maxcut_eigenvalues,
    expectation_value,
    brute_force_maxcut,
    real_dtype,
)


def _rx(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """RX(theta) = exp(-i theta X / 2) = [[cos(t/2), -i sin(t/2)],
                                           [-i sin(t/2), cos(t/2)]].

    theta: (batch,) o escalar. Devuelve (batch, 2, 2).
    """
    half = theta / 2
    if half.dim() == 0:
        half = half.reshape(1)
    c = torch.cos(half)
    s = torch.sin(half)
    z = torch.zeros_like(half)
    return torch.stack([
        torch.stack([c, -1j * s], dim=-1),
        torch.stack([-1j * s, c], dim=-1),
    ], dim=-2).to(dtype)


def _initial_state(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Estado |+...+> = H^n |0...0>. Cada amplitud = 1/sqrt(2^n).

    OJO: 1/sqrt(2^n) NO es 1/2**(n//2) para n impar. Usar sqrt(2^n) directo.
    """
    amp = 1.0 / math.sqrt(1 << n)
    psi = torch.full((1, 1 << n), amp + 0j, dtype=dtype, device=device)
    return psi


def qaoa_state(sim: Simulator, edges: List[Tuple[int, int]],
                gamma: torch.Tensor, beta: torch.Tensor,
                p: int) -> torch.Tensor:
    """Construye el estado QAOA |psi(gamma,beta)>.

    gamma: (p,) angulos de coste.
    beta:  (p,) angulos de mezclador (un solo beta por capa, como es
            habitual: B = sum_i X_i tiene un unico paramtro por capa).
    """
    n = sim.n_qubits
    psi = _initial_state(n, str(sim.device), sim.dtype)

    c_diag = maxcut_eigenvalues(edges, n,
                                dtype=real_dtype(sim.dtype),
                                device=str(sim.device))

    for layer in range(p):
        phase = torch.exp(-1j * gamma[layer] * c_diag).to(sim.dtype)
        psi = psi * phase
        for q in range(n):
            psi = sim._apply_1q(psi, _rx(2.0 * beta[layer], sim.dtype), q)

    return psi


def qaoa_maxcut(sim: Simulator, edges: List[Tuple[int, int]],
                gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """<C(gamma,beta)> del Ansatz QAOA. Devuelve tensor (batch,) real.

    gamma: (p,) o (batch, p)
    beta:  (p,) o (batch, p)
    """
    n = sim.n_qubits
    p = gamma.shape[-1]
    b = gamma.shape[0] if gamma.dim() > 1 else 1
    gamma = gamma.reshape(b, p)
    beta = beta.reshape(b, p)

    c_diag = maxcut_eigenvalues(edges, n,
                                dtype=real_dtype(sim.dtype),
                                device=str(sim.device))
    psi = _initial_state(n, str(sim.device), sim.dtype).expand(b, 1 << n).clone()

    for layer in range(p):
        phase = torch.exp(-1j * gamma[:, layer].unsqueeze(1) * c_diag).to(sim.dtype)
        psi = psi * phase
        # un solo angulo de mezclador por capa (B = sum_i X_i, un solo beta)
        for q in range(n):
            psi = sim._apply_1q(psi, _rx(2.0 * beta[:, layer], sim.dtype), q)

    return expectation_value(psi, edges, n)


def cost_function(params: torch.Tensor, sim: Simulator,
                  edges: List[Tuple[int, int]], n: int, p: int) -> torch.Tensor:
    """Funcion de costo para el optimizador: -<C> (maximizar corte).

    params: (2p,) = [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}].
    gamma y beta comparten la misma dimension p (un angulo por capa).
    """
    gamma = params[:p].reshape(1, p)
    beta = params[p:].reshape(1, p)
    return -qaoa_maxcut(sim, edges, gamma, beta)[0]


def optimize_qaoa(sim: Simulator, edges: List[Tuple[int, int]],
                  p: int = 2, n_steps: int = 200,
                  lr: float = 0.05, seed: int = 0,
                  device: str = 'cpu') -> Tuple[torch.Tensor, float, float]:
    """Optimiza (gamma,beta) con autograd de PyTorch (Adam).

    Devuelve (params_finales, corte_maximo_encontrado, corte_optimo_fuerza_bruta).
    """
    n = sim.n_qubits
    torch.manual_seed(seed)
    params = torch.randn(2 * p, device=device, dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)

    best_cut = -1.0
    best_params = params.detach().clone()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = cost_function(params, sim, edges, n, p)
        loss.backward()
        if not torch.isfinite(params.grad).all():
            break
        opt.step()
        with torch.no_grad():
            current = -float(loss.item())
            if current > best_cut:
                best_cut = current
                best_params = params.detach().clone()

    return best_params, best_cut, brute_force_maxcut(edges, n)