"""
Canal despolarizante de un qubit.

    rho -> (1 - p) rho + p I / 2

Implementado de dos formas equivalentes:
  1. Operadores de Kraus: K0 = sqrt(1-p) I, K_i = sqrt(p/3) P_i (P in {X,Y,Z}).
  2. Superoperador: matriz 4x4 S tal que vec(E(rho)) = S vec(rho)
     (apilamiento por columnas), S = sum_k conj(K_k) (x) K_k.

Creado para el AMD Developer Hackathon ACT III (Chern Calibration Copilot).
"""

from __future__ import annotations

import torch

from .operadores_salto import I2, X, Y, Z

_COMPLEX = torch.complex128


def kraus_ops(p: float) -> list[torch.Tensor]:
    """Operadores de Kraus del canal despolarizante de un qubit."""
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p debe estar en [0, 1]")
    return [
        torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)).to(_COMPLEX) * I2,
        torch.sqrt(torch.tensor(p / 3.0, dtype=torch.float64)).to(_COMPLEX) * X,
        torch.sqrt(torch.tensor(p / 3.0, dtype=torch.float64)).to(_COMPLEX) * Y,
        torch.sqrt(torch.tensor(p / 3.0, dtype=torch.float64)).to(_COMPLEX) * Z,
    ]


def aplicar_canal(rho: torch.Tensor, p: float) -> torch.Tensor:
    """Aplica el canal despolarizante a una matriz de densidad 2x2."""
    from .densidad import DensityMatrix
    return DensityMatrix(rho).apply_channel(kraus_ops(p)).rho


def superoperator(p: float) -> torch.Tensor:
    """Superoperador 4x4 del canal despolarizante (base computacional,
    vec apila columnas): vec(E(rho)) = S vec(rho)."""
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p debe estar en [0, 1]")
    S = torch.zeros((4, 4), dtype=_COMPLEX)
    for K in kraus_ops(p):
        S = S + torch.kron(K.conj(), K)
    return S

__all__ = ["kraus_ops", "aplicar_canal", "superoperator"]
