"""Operador de coste MaxCut: C = sum_{(i,j) en E} (1 - Z_i Z_j) / 2.

Para n pequeno (<=6 qubits) se construye la matriz 2^n x 2^n con kron de Pauli Z.
Tambien hay version vectorizada que aplica sobre un statevector sin construir
la matriz completa: C es diagonal en la base Z, asi que basta con un vector de
valores propios de longitud 2^n (un valor por configuraion de bits).
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    """float32/float64 que corresponde a un dtype complejo.

    torch.finfo(complex_dtype).dtype devuelve un STRING, por lo que se
    resuelve con getattr. Igual que en qsim_torch.py.
    """
    return getattr(torch, torch.finfo(complex_dtype).dtype)


def maxcut_matrix(edges: List[Tuple[int, int]], n: int) -> torch.Tensor:
    """Matriz 2^n x 2^n del operador C = sum (1 - Z_i Z_j)/2.

    Construida con kron de Pauli Z para n pequeno (<=6 qubits). Cada Z_i Z_j
    es el kron de n tensores 2x2 donde las posiciones i y j son Z y el resto
    identidad. Suma sobre aristas y escala por 1/2.

    Para uso en produccion usa maxcut_eigenvalues (diagonal, mas eficiente):
    C es diagonal en la base Z.
    """
    size = 1 << n
    I2 = torch.eye(2, dtype=torch.complex128)
    Z2 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    C = torch.zeros(size, size, dtype=torch.complex128)
    for (i, j) in edges:
        ops = [I2] * n
        ops[i] = Z2
        ops[j] = Z2
        ZiZj = ops[0]
        for k in range(1, n):
            ZiZj = torch.kron(ZiZj, ops[k])
        C = C + 0.5 * (torch.eye(size, dtype=torch.complex128) - ZiZj)
    return C


def maxcut_eigenvalues(edges: List[Tuple[int, int]], n: int,
                       dtype: torch.dtype = torch.float64,
                       device: str = 'cpu') -> torch.Tensor:
    """Valores propios de C (vector de longitud 2^n), version diagonal.

    C es diagonal en la base Z y sus valores propios son REALES (C es
    hermitiano). Devuelve un tensor real, no complejo: sirve tanto para
    construir la fase U_C(gamma) = exp(-i gamma C) como para calcular
    <C> = sum_i |psi_i|^2 * C_diag[i] sin mezclar tipos complejos.

    Cada configuraion de bits idx aporta a la arista (i,j):
        0.5 * (1 - s_i(idx) * s_j(idx))   con s_q = +1 si bit q = 0, -1 si 1
    Es decir: 1 si los bits i y j difieren (arista cortada), 0 si no.
    """
    size = 1 << n
    idx = torch.arange(size, device=device)
    diag = torch.zeros(size, dtype=dtype, device=device)
    for (i, j) in edges:
        bi = (idx >> (n - i - 1)) & 1
        bj = (idx >> (n - j - 1)) & 1
        si = 1.0 - 2.0 * bi.to(dtype)
        sj = 1.0 - 2.0 * bj.to(dtype)
        diag = diag + 0.5 * (1.0 - si * sj)
    return diag


def expectation_value(psi: torch.Tensor, edges: List[Tuple[int, int]],
                      n: int) -> torch.Tensor:
    """<C> = sum_i |psi_i|^2 * C_diag[i] = <psi| C |psi>.

    Vectorizada: una sola contraction (batch, 2^n) @ (2^n,) en lugar de
    construir la matriz 2^n x 2^n.
    """
    diag = maxcut_eigenvalues(edges, n,
                              dtype=real_dtype(psi.dtype), device=psi.device)
    probs = psi.abs() ** 2
    return (probs * diag).sum(dim=1)


def maxcut_value_from_bits(edges: List[Tuple[int, int]], assignment: int,
                           n: int) -> float:
    """Corte de una asignacion de bits concreta: numero de aristas cortadas.

    assignment: entero de n bits, bit q = 0 o 1 = lado del corte del qubit q.
    """
    cut = 0
    for (i, j) in edges:
        bi = (assignment >> (n - i - 1)) & 1
        bj = (assignment >> (n - j - 1)) & 1
        if bi != bj:
            cut += 1
    return float(cut)


def matrix_from_eigenvalues(edges: List[Tuple[int, int]], n: int) -> torch.Tensor:
    """Matriz diagonal construida a partir de maxcut_eigenvalues (referencia)."""
    diag = maxcut_eigenvalues(edges, n, dtype=torch.float64)
    return torch.diag(diag.to(torch.complex128))


def brute_force_maxcut(edges: List[Tuple[int, int]], n: int) -> float:
    """Corte maximo por fuerza bruta (n <= 6)."""
    best = 0.0
    for assignment in range(1 << n):
        best = max(best, maxcut_value_from_bits(edges, assignment, n))
    return best