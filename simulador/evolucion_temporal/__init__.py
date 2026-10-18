"""Paquete de evolución temporal de Haldane (ecuación de Schrödinger td).

Módulos:
  hamiltonian_td  - H(k, t) paramétrico en el tiempo (M(t) cruzando M_c)
  estado_inicial  - estados localizados en sitio o borde
  evolucion_td   - integrador Trotter (puertas 1q) + exponenciación matriz
  diagnosticos    - transferencia de carga, polarización, Chern dinámico

Uso mínimo:
    from evolucion_temporal import HaldaneTD, EvolucionTD, estado_producto_A
    ham = HaldaneTD(M_start=-1.5, M_end=1.5, T=1.0, modo="rampa")
    sim = Simulator(n_qubits=4, device="cpu")
    evol = EvolucionTD(sim, ham, dt=0.01)
    psi, _ = evol.evolucionar(estado_producto_A(sim), t0=0.0, n_steps=50)
"""

from __future__ import annotations

from .diagnosticos import (chern_dinamico, norma, polarizacion_berry,
                           poblacion_sublattice, transferencia_carga)
from .evolucion_td import EvolucionTD
from .estado_inicial import (estado_localizado, estado_producto_A,
                             estado_superposicion_uniforme)
from .hamiltonian_td import HaldaneTD, d_vector, hamiltonian_2x2

__all__ = [
    "HaldaneTD",
    "d_vector",
    "hamiltonian_2x2",
    "EvolucionTD",
    "estado_producto_A",
    "estado_superposicion_uniforme",
    "estado_localizado",
    "norma",
    "poblacion_sublattice",
    "transferencia_carga",
    "polarizacion_berry",
    "chern_dinamico",
]
