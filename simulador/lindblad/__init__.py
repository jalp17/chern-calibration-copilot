"""Paquete Lindblad: matriz de densidad y ecuacion maestra de sistema abierto.

Modulos:
  densidad           - DensityMatrix (estado mixto, operaciones basicas)
  lindblad           - lindblad_step, lindblad_evolve (integrador)
  operadores_salto   - L_k comunes (amplitud, desfasaje, despolarizante, fotonica)
  canal_depolarizante - canal despolarizante (Kraus + superoperador)

Uso minimo:
    from simulador.lindblad import DensityMatrix, lindblad_evolve, decaimiento_amplitud
    rho = DensityMatrix.from_statevector(psi)
    rho_final, _ = lindblad_evolve(rho.rho, H, jump_ops, t_final=1.0, n_steps=100)
"""

from __future__ import annotations

from .canal_depolarizante import aplicar_canal, kraus_ops, superoperator
from .densidad import DensityMatrix
from .lindblad import lindblad_evolve, lindblad_evolve_dm, lindblad_step
from .operadores_salto import (I2, SP, SM, X, Y, Z,
                               decaimiento_amplitud, desfasaje_puro,
                               despolarizante, perdida_fotonica)

__all__ = [
    "DensityMatrix",
    "lindblad_step",
    "lindblad_evolve",
    "lindblad_evolve_dm",
    "aplicar_canal",
    "kraus_ops",
    "superoperator",
    "decaimiento_amplitud",
    "perdida_fotonica",
    "desfasaje_puro",
    "despolarizante",
    "I2", "X", "Y", "Z", "SM", "SP",
]
