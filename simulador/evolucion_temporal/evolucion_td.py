"""
Integrador de evolución temporal i d_t |psi(t)> = H(t) |psi(t)> para Haldane.

Estrategia: H = sum_k H(k), donde cada H(k) actúa sobre su propio qubit y
[H(k), H(k')] = 0. Por tanto un paso de Trotter de primer orden,

    U(dt) = prod_k exp(-i H(k) dt),

es EXACTO (error de Trotter = 0). Cada factor es una puerta de 1 qubit
aplicada con Simulator._apply_1q. No se requieren CNOT porque los sectores
k son independientes; se provee exponenciación de matriz completa para
validación de sistemas pequeños.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qsim_torch import Simulator
from .hamiltonian_td import HaldaneTD


class EvolucionTD:
    """Integrador Trotter para la evolución temporal de Haldane.

    Parameters
    ----------
    sim : Simulator
        Simulador statevector con n_qubits = N_mesh**2 (un qubit por punto k).
    ham : HaldaneTD
        Hamiltoniano dependiente del tiempo.
    dt : float
        Paso de tiempo.
    """

    def __init__(self, sim: Simulator, ham: HaldaneTD, dt: float):
        self.sim = sim
        self.ham = ham
        self.dt = float(dt)
        self.n_qubits = sim.n_qubits

    # ------------------------------------------------------------- paso

    def paso(self, psi: torch.Tensor, t: float) -> torch.Tensor:
        """Aplica un paso de Trotter de tamaño dt en el tiempo t.

        Como los H(k) conmutan (actúan en qubits distintos), el producto
        ordenado de unitarios de 1 qubit es exacto: error de Trotter = 0.
        """
        dt = self.dt
        sim = self.sim
        for q in range(self.n_qubits):
            U = self.ham.unidad_evolucion_k(self.k_points[q], t, dt)
            Ut = torch.tensor(U, dtype=sim.dtype, device=sim.device)
            psi = sim._apply_1q(psi, Ut, q)
        return psi

    # ------------------------------------------------------------- evolución

    def evolucionar(self, psi0: torch.Tensor, t0: float = 0.0,
                     t1: float | None = None, n_steps: int | None = None,
                     guardar_historial: bool = False):
        """Evoluciona psi0 de t0 durante n_steps pasos de tamaño dt.

        Si se dan t1 y n_steps es None, se infiere
        n_steps = round((t1 - t0) / dt).

        Devuelve (psi_final, historial) donde historial es lista de
        (t, psi) si guardar_histral, en cuyo caso incluye t0.
        """
        if n_steps is None:
            assert t1 is not None, "hay que proporcionar t1 o n_steps"
            n_steps = int(round((t1 - t0) / self.dt))
        psi = psi0.clone()
        t = t0
        historial = [(t0, psi.clone())] if guardar_historial else None
        for _ in range(n_steps):
            psi = self.paso(psi, t)
            t = t + self.dt
            if guardar_historial:
                historial.append((t, psi.clone()))
        return psi, historial

    def evolucionar_pasos(self, psi0: torch.Tensor, t0: float,
                           n_steps: int,
                           guardar_historial: bool = False):
        """Evoluciona exactamente n_steps pasos empezando en t0."""
        return self.evolucionar(psi0, t0=t0, n_steps=n_steps,
                                guardar_historial=guardar_historial)

    # ------------------------------------------------------------- puntos k

    @property
    def k_points(self) -> np.ndarray:
        """Malla de puntos k (n_qubits, 2), uno por qubit, orden row-major."""
        if not hasattr(self, "_k_points"):
            net = self.ham.net
            N = int(round(math.sqrt(self.n_qubits)))
            u = (np.arange(N) + 0.5) / N
            v = (np.arange(N) + 0.5) / N
            uu, vv = np.meshgrid(u, v, indexing="ij")
            self._k_points = (uu[..., None] * net["G1"] +
                              vv[..., None] * net["G2"]).reshape(N * N, 2)
        return self._k_points

    # ------------------------------------------------------------- exponenciación de matriz

    def hamiltoniano_completo(self, t: float) -> np.ndarray:
        """H_full(t) = sum_k H(k, t) ⊗ I_rest, matriz 2^n x 2^n.

        qubit 0 es el factor más significativo (MSB), coherente con
        qsim_torch (embed empieza por qubit 0 a la izquierda).
        """
        n = self.n_qubits
        I2 = np.eye(2, dtype=np.complex128)
        size = 1 << n
        H_full = np.zeros((size, size), dtype=np.complex128)
        for q in range(n):
            Hk = self.ham.H_k(self.k_points[q], t)           # (2, 2)
            ops = [I2] * n
            ops[q] = Hk
            term = np.array([[1]], dtype=np.complex128)
            for op in ops:
                term = np.kron(term, op)
            H_full = H_full + term
        return H_full

    def exponenciacion_matriz(self, psi0: torch.Tensor, t: float,
                              H_full: np.ndarray | None = None) -> torch.Tensor:
        """|psi(t)> = exp(-i H_full t) |psi0> por exponencial de matriz.

        Referencia independiente (scipy.linalg.expm) para validar el
        integrador Trotter. Sólo sistemas pequeños (2^n manejable).
        """
        from scipy.linalg import expm
        if H_full is None:
            H_full = self.hamiltoniano_completo(t)
        U = expm(-1j * H_full * t)
        psi_np = psi0[0].numpy()
        psi_t = U @ psi_np
        return torch.tensor(psi_t, dtype=self.sim.dtype,
                            device=self.sim.device).unsqueeze(0)
