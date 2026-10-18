"""
Matriz de densidad para el simulador de circuitos cuánticos.

Representa un estado mixto rho (matriz dxd, d = 2^n) y ofrece las
operaciones básicas: evolución unitaria, canales de Kraus, observables,
pureza, entropia de von Neumann y trazada parcial.

Todo en PyTorch (complex128), CPU por defecto. Disenado para n <= 6 en
las pruebas locales (matriz 64x64 = 4 KiB) y escalable a n ~ 14 en GPU.

Creado para el AMD Developer Hackathon ACT III (Chern Calibration Copilot).
"""

from __future__ import annotations

import string

import torch

_COMPLEX = torch.complex128


def _n_qubits(rho: torch.Tensor) -> int:
    d = rho.shape[0]
    n = d.bit_length() - 1
    if (1 << n) != d or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho debe ser cuadrada de dimension 2^n, obtuvo {rho.shape}")
    return n


class DensityMatrix:
    """Envuelve una matriz de densidad rho (d x d), d = 2^n."""

    def __init__(self, rho: torch.Tensor) -> None:
        rho = rho.to(dtype=_COMPLEX)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho debe ser una matriz cuadrada 2D")
        self.rho = rho
        self.n = _n_qubits(rho)

    # --- constructores -----------------------------------------------------

    @classmethod
    def from_statevector(cls, psi: torch.Tensor) -> "DensityMatrix":
        """rho = |psi><psi| a partir de un vector de estado puro."""
        psi = psi.to(dtype=_COMPLEX).reshape(-1)
        return cls(torch.outer(psi, psi.conj()))

    @classmethod
    def from_probabilities(cls, probs: torch.Tensor) -> "DensityMatrix":
        """rho diagonal a partir de un vector de probabilidades."""
        probs = probs.to(dtype=torch.float64).reshape(-1)
        return cls(torch.diag(probs.to(dtype=_COMPLEX)))

    @classmethod
    def maximamente_mezclado(cls, n: int, device=None, dtype=_COMPLEX) -> "DensityMatrix":
        """rho = I / 2^n."""
        d = 1 << n
        return cls(torch.eye(d, device=device, dtype=dtype) / d)

    # --- operaciones -------------------------------------------------------

    def apply_unitary(self, U: torch.Tensor) -> "DensityMatrix":
        """rho -> U rho U†."""
        U = U.to(dtype=_COMPLEX, device=self.rho.device)
        self.rho = U @ self.rho @ U.conj().T
        return self

    def apply_channel(self, kraus_ops: list[torch.Tensor]) -> "DensityMatrix":
        """rho -> sum_k K_k rho K_k† (canal de Kraus, traza preservante si
        sum_k K_k† K_k = I)."""
        out = torch.zeros_like(self.rho)
        for K in kraus_ops:
            K = K.to(dtype=_COMPLEX, device=self.rho.device)
            out = out + K @ self.rho @ K.conj().T
        self.rho = out
        return self

    def expectation(self, observable: torch.Tensor) -> float:
        """Esperado Tr[rho O] (O hermitiano => resultado real)."""
        observable = observable.to(dtype=_COMPLEX, device=self.rho.device)
        return float(torch.trace(self.rho @ observable).real)

    def purity(self) -> float:
        """Pureza Tr[rho^2] en [1/d, 1]. 1 = estado puro."""
        return float(torch.trace(self.rho @ self.rho).real)

    def von_neumann_entropy(self) -> float:
        """S = -Tr[rho log rho] (log natural)."""
        w = torch.linalg.eigvalsh(self.rho)
        w = w.clamp(min=0.0)
        nz = w > 0
        return float((-w[nz] * torch.log(w[nz])).sum())

    def partial_trace(self, subsystem_indices: list[int]) -> "DensityMatrix":
        """Traza parcial sobre los subsistemas (qubits) indicados.

        Devuelve la matriz de densidad reducida sobre los qubits restantes.
        """
        n = self.n
        subs = set(subsystem_indices)
        if not subs.issubset(set(range(n))):
            raise ValueError("indices de subsistema fuera de rango")

        dim = 2
        tensor = self.rho.reshape((dim,) * n + (dim,) * n)

        labels = string.ascii_letters
        row_labels = [labels[i] for i in range(n)]
        col_labels = [labels[n + i] for i in range(n)]
        in_str = "".join(row_labels + col_labels)
        out_labels = [labels[i] for i in range(n) if i not in subs] + \
                     [labels[n + i] for i in range(n) if i not in subs]
        out_str = "".join(out_labels)

        reduced = torch.einsum(f"{in_str}->{out_str}", tensor)
        m = n - len(subs)
        rho_reduced = reduced.reshape(1 << m, 1 << m)
        return DensityMatrix(rho_reduced)

    # --- utilidades --------------------------------------------------------

    def trace(self) -> float:
        return float(torch.trace(self.rho).real)

    def eigenvalues(self) -> torch.Tensor:
        return torch.linalg.eigvalsh(self.rho)

    def is_positive(self, tol: float = 1e-9) -> bool:
        return bool(torch.all(self.eigenvalues() >= -tol))

    def __repr__(self) -> str:
        return f"DensityMatrix(n={self.n}, trace={self.trace():.6f}, purity={self.purity():.6f})"
