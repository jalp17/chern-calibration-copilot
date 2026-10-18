"""
Modelo de ruido fisico para el codigo de superficie.

Aplica errores de bit-flip (X), phase-flip (Z) y ambos (Y) con probabilidad p
por qubit. Para d=3 el estado tiene 13 qubits, asi que el ruido es barato de
simular exactamente como un string de Pauli (o aplicandolo sobre el
statevector con el simulador base).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .codigo_superficie import SurfaceCode, pauli_string

Pauli = Tuple[Tuple[int, str], ...]


def muestrear_error(code: SurfaceCode, p: float, seed: Optional[int] = None,
                    rng: Optional[torch.Generator] = None) -> Pauli:
    """Muestra un error de Pauli aleatorio.

    Cada qubit recibe I con prob 1-p, y X/Y/Z con probabilidad p/3 cada uno.
    Devuelve el string de Pauli del error acumulado.
    """
    if rng is None:
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
    error: list = []
    for q in code.qubits:
        r = torch.rand(1, generator=rng).item()
        if r < p:
            r2 = torch.rand(1, generator=rng).item()
            if r2 < 1 / 3:
                error.append((q, 'X'))
            elif r2 < 2 / 3:
                error.append((q, 'Z'))
            else:
                error.append((q, 'Y'))
    return tuple(sorted(error))


def aplicar_error_estado(psi: torch.Tensor, sim, error: Pauli,
                         code: SurfaceCode) -> torch.Tensor:
    """Aplica un string de Pauli sobre un statevector (batch, 2^n).

    Usa _apply_1q del simulador para cada Pauli no identidad del error.
    """
    from qsim_torch import X as Xm, Y as Ym, Z as Zm
    gates = {'X': Xm, 'Y': Ym, 'Z': Zm}
    for q, p in error:
        gate = gates[p].to(psi.dtype).to(psi.device)
        psi = sim._apply_1q(psi, gate, q)
    return psi
