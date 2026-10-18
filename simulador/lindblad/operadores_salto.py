"""
Operadores de salto (L_k) comunes para la ecuacion de Lindblad.

Cada funcion devuelve una tupla de matrices 2x2 (un unico operador de salto
o varios, segun el canal). Las matrices incluyen la fuerza de acoplamiento
sqrt(gamma) ya incorporada, tal como las espera lindblad_step.

Convenciones (base |0>, |1>):
    sigma- = |0><1| = [[0, 1], [0, 0]]    (relajacion / decayo)
    sigma+ = |1><0| = [[0, 0], [1, 0]]    (excitacion / ganancia)
    Z      = [[1, 0], [0, -1]]
    X      = [[0, 1], [1, 0]]
    Y      = [[0, -i], [i, 0]]

Creado para el AMD Developer Hackathon ACT III (Chern Calibration Copilot).
"""

from __future__ import annotations

import torch

_COMPLEX = torch.complex128

# --- Pauli de 1 qubit ------------------------------------------------------
I2 = torch.tensor([[1, 0], [0, 1]], dtype=_COMPLEX)
X = torch.tensor([[0, 1], [1, 0]], dtype=_COMPLEX)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=_COMPLEX)
Z = torch.tensor([[1, 0], [0, -1]], dtype=_COMPLEX)

SM = torch.tensor([[0, 1], [0, 0]], dtype=_COMPLEX)   # sigma- = |0><1|
SP = torch.tensor([[0, 0], [1, 0]], dtype=_COMPLEX)   # sigma+ = |1><0|


def _raiz(gamma: float) -> torch.Tensor:
    return torch.sqrt(torch.tensor(float(gamma), dtype=torch.float64)).to(_COMPLEX)


def decaimiento_amplitud(gamma: float) -> tuple[torch.Tensor, ...]:
    """Relajacion de amplitud (T1): energia se disipa al entorno.

    L = sqrt(gamma) * sigma-,人口 de |1> decae a |0> con tasa gamma.
    """
    return (_raiz(gamma) * SM,)


def perdida_fotonica(gamma: float) -> tuple[torch.Tensor, ...]:
    """Perdida de foton: sinonimo de decaimiento de amplitud."""
    return decaimiento_amplitud(gamma)


def desfasaje_puro(gamma: float) -> tuple[torch.Tensor, ...]:
    """Desfasaje puro (T2*): desfasa la coherencia sin cambiar poblaciones.

    L = sqrt(gamma) * Z. Las fuera-diagonales de rho decaen como exp(-2 gamma t).
    """
    return (_raiz(gamma) * Z,)


def despolarizante(gamma: float) -> tuple[torch.Tensor, ...]:
    """Despolarizante de un qubit: tres saltos, uno por Pauli no trivial.

    L_k = sqrt(gamma / 3) * P_k, k in {X, Y, Z}.
    """
    r = torch.sqrt(torch.tensor(float(gamma) / 3.0, dtype=torch.float64)).to(_COMPLEX)
    return (r * X, r * Y, r * Z)

__all__ = [
    "decaimiento_amplitud",
    "perdida_fotonica",
    "desfasaje_puro",
    "despolarizante",
    "I2", "X", "Y", "Z", "SM", "SP",
]
