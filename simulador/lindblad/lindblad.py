"""
Evolucion de Lindblad para sistemas quanticos abiertos.

Ecuacion maestra de Lindblad:

    d rho / dt = -i [H, rho] + sum_k ( L_k rho L_k† - 1/2 { L_k† L_k, rho })

Integracion por división de operadores:
  1. Evolucion unitaria exacta:  rho -> U rho U†,  U = expm(-i H dt)
  2. Disipador explicito de Euler:  rho -> rho + dt * D(rho)

Creado para el AMD Developer Hackathon ACT III (Chern Calibration Copilot).
"""

from __future__ import annotations

import torch

from .densidad import DensityMatrix, _COMPLEX


def _expand_op(op_1q: torch.Tensor, n_qubits: int, target_qubit: int = 0) -> torch.Tensor:
    """Expande un operador de un qubit al espacio de Hilbert completo de n qubits.

    El operador se coloca en `target_qubit` (0 = MSB, convencion big-endian)
    e identidad en todos los demas qubits.
    """
    I2 = torch.eye(2, dtype=op_1q.dtype, device=op_1q.device)
    mats = [I2] * n_qubits
    mats[target_qubit] = op_1q
    result = mats[0]
    for m in mats[1:]:
        result = torch.kron(result, m)
    return result


def _expand_jump_ops(jump_ops: list[torch.Tensor], n_qubits: int) -> list[torch.Tensor]:
    """Expande todos los operadores de salto al espacio de Hilbert completo."""
    expanded = []
    for L in jump_ops:
        L = L.to(dtype=_COMPLEX)
        if L.shape == (2, 2):
            expanded.append(_expand_op(L, n_qubits, 0))
        else:
            expanded.append(L.to(dtype=_COMPLEX))
    return expanded


def _disipador(rho: torch.Tensor, jump_ops: list[torch.Tensor]) -> torch.Tensor:
    """sum_k ( L_k rho L_k† - 1/2 { L_k† L_k, rho })."""
    dis = torch.zeros_like(rho)
    for L in jump_ops:
        L = L.to(dtype=rho.dtype, device=rho.device)
        Ld = L.conj().T
        LdL = Ld @ L
        dis = dis + L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
    return dis


def lindblad_step(rho: torch.Tensor, H: torch.Tensor,
                  jump_ops: list[torch.Tensor], dt: float) -> torch.Tensor:
    """Un paso de tiempo de la ecuacion de Lindblad por division de operadores.

    1. Evolucion unitaria exacta:  rho1 = U rho U†, U = expm(-i H dt)
    2. Disipador explicito de Euler:  rho2 = rho1 + dt * D(rho1)

    No modifica `rho` ni `H` originales.
    """
    rho = rho.to(dtype=_COMPLEX)
    H = H.to(dtype=_COMPLEX, device=rho.device)
    n_qubits = rho.shape[0].bit_length() - 1

    # Paso 1: evolucion unitaria exacta
    U = torch.linalg.matrix_exp(-1j * H * dt)
    rho1 = U @ rho @ U.conj().T

    # Paso 2: disipador explicito de Euler
    jump_ops_full = _expand_jump_ops(jump_ops, n_qubits)
    drho = _disipador(rho1, jump_ops_full)
    return rho1 + dt * drho


def lindblad_evolve(rho0: torch.Tensor, H: torch.Tensor,
                    jump_ops: list[torch.Tensor], t_final: float,
                    n_steps: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Evoluciona rho0 desde t=0 hasta t_final con n_steps pasos.

    Devuelve (rho_final, historial) donde historial incluye rho0 y el estado
    tras cada paso (longitud n_steps + 1).
    """
    if n_steps < 1:
        raise ValueError("n_steps debe ser >= 1")
    dt = float(t_final) / n_steps
    rho = rho0.clone().to(dtype=_COMPLEX)
    historial = [rho.clone()]
    for _ in range(n_steps):
        rho = lindblad_step(rho, H, jump_ops, dt)
        # Renormalizar traza para eliminar deriva numerica
        rho = rho / torch.trace(rho).real
        historial.append(rho.clone())
    return rho, historial


def lindblad_evolve_dm(rho0: DensityMatrix, H: torch.Tensor,
                       jump_ops: list[torch.Tensor], t_final: float,
                       n_steps: int) -> tuple[DensityMatrix, list[DensityMatrix]]:
    """Version que trabaja con DensityMatrix (comodo para tests)."""
    rho_final, historial = lindblad_evolve(rho0.rho, H, jump_ops, t_final, n_steps)
    return DensityMatrix(rho_final), [DensityMatrix(r) for r in historial]