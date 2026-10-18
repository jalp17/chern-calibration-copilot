"""
Hamiltoniano de Haldane dependiente del tiempo H(k, t) = d(k, t) . sigma.

dz(k, t) = M(t) - 2 t2 sin(phi) sum_i sin(k . b_i)
dx, dy   = t1 sum_i cos/sin(k . d_i)        (independientes de t)

M(t) es una rampa lineal o senoidal que cruza la masa crítica
M_c = 3 sqrt(3) t2 |sin(phi)|, disparando la transición topológica.

Cada punto k de la malla de Brillouin es un sistema de 2 niveles (el spinor
A|B), es decir, exactamente 1 qubit. Los distintos k actúan sobre qubits
distintos, luego [H(k), H(k')] = 0 y la evolución total se factoriza en
puertas de 1 qubit por k (Trotter de primer orden con error nulo).
"""

from __future__ import annotations

import math

import numpy as np

from haldane import honeycomb, critical_mass


def d_vector(net: dict, k: np.ndarray, M: float,
             t1: float = 1.0, t2: float = 0.20,
             phi: float = math.pi / 2) -> np.ndarray:
    """(dx, dy, dz) para k de forma (2,) o (..., 2). Devuelve (..., 3)."""
    k = np.asarray(k, dtype=np.float64)
    kd = k @ net["d"].T                      # (..., 3)
    kb = k @ net["b"].T                      # (..., 3)
    dx = t1 * np.sum(np.cos(kd), axis=-1)
    dy = t1 * np.sum(np.sin(kd), axis=-1)
    dz = M - 2.0 * t2 * np.sin(phi) * np.sum(np.sin(kb), axis=-1)
    return np.stack([dx, dy, dz], axis=-1)


def hamiltonian_2x2(net: dict, k: np.ndarray, M: float,
                    t1: float = 1.0, t2: float = 0.20,
                    phi: float = math.pi / 2) -> np.ndarray:
    """H(k, t) 2x2 en la base (A, B). k: (2,) -> (2, 2)."""
    dx, dy, dz = d_vector(net, k, M, t1, t2, phi)
    H = np.empty(dx.shape + (2, 2), dtype=np.complex128)
    H[..., 0, 0] = dz
    H[..., 0, 1] = dx - 1j * dy
    H[..., 1, 0] = dx + 1j * dy
    H[..., 1, 1] = -dz
    return H


class HaldaneTD:
    """H(k, t) de Haldane con M(t) paramétrico en el tiempo.

    Modos:
      - "rampa":    M lineal de M_start a M_end en el intervalo [0, T].
      - "seno":     M = M_start + (M_end-M_start)/2 * (1 - cos(pi t/T)).
      - "ciclico":  M = M_c sin(2 pi t / T)  (ciclo de bombeo de Thouless).
    """

    def __init__(self, net: dict | None = None, t1: float = 1.0,
                 t2: float = 0.20, phi: float = math.pi / 2,
                 M_start: float = -1.5, M_end: float = 1.5,
                 T: float = 1.0, modo: str = "rampa"):
        self.net = net or honeycomb()
        self.t1 = t1
        self.t2 = t2
        self.phi = phi
        self.M_start = M_start
        self.M_end = M_end
        self.T = float(T)
        self.modo = modo
        self.M_c = float(critical_mass(0.0, 1.0, t2, phi))

    def M(self, t: float) -> float:
        """Valor de la masa en el tiempo t."""
        if self.modo == "rampa":
            s = min(max(t / self.T, 0.0), 1.0)
            return self.M_start + (self.M_end - self.M_start) * s
        if self.modo == "seno":
            s = 0.5 * (1.0 - math.cos(math.pi * min(max(t / self.T, 0.0), 1.0)))
            return self.M_start + (self.M_end - self.M_start) * s
        if self.modo == "ciclico":
            return float(self.M_c * math.sin(2.0 * math.pi * t / self.T))
        raise ValueError(f"modo desconocido: {self.modo}")

    def cruza_transicion(self, t: float) -> bool:
        """True si |M(t)| está dentro de un pequeño epsilon de M_c."""
        return abs(abs(self.M(t)) - self.M_c) < 1e-3

    def H_k(self, k: np.ndarray, t: float) -> np.ndarray:
        """Hamiltoniano 2x2 en el punto k y tiempo t."""
        return hamiltonian_2x2(self.net, np.asarray(k, dtype=np.float64),
                               self.M(t), self.t1, self.t2, self.phi)

    def d_k(self, k: np.ndarray, t: float) -> np.ndarray:
        """Vector d(k, t) tal que H = d . sigma. Devuelve (3,) o (..., 3)."""
        return d_vector(self.net, np.asarray(k, dtype=np.float64),
                        self.M(t), self.t1, self.t2, self.phi)

    def unidad_evolucion_k(self, k: np.ndarray, t: float, dt: float) -> np.ndarray:
        """U(k, t, dt) = exp(-i H(k, t) dt), matriz 2x2 exacta.

        Usa exp(-i d.sigma dt) = cos(|d|dt) I - i sin(|d|dt) d_hat . sigma.
        """
        d = self.d_k(k, t)                                  # (3,)
        mag = float(np.linalg.norm(d))
        if mag < 1e-14:
            return np.eye(2, dtype=np.complex128)
        nhat = d / mag
        a = mag * dt                                        # ángulo / 2
        c = math.cos(a)
        s = math.sin(a)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        U = c * np.eye(2, dtype=np.complex128) - 1j * s * (
            nhat[0] * sigma_x + nhat[1] * sigma_y + nhat[2] * sigma_z)
        return U
