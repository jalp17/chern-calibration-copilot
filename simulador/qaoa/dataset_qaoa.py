"""Dataset de grafos para QAOA MaxCut.

Grafos aleatorios con n pequeno (<=6 qubits) y corte optimo por fuerza bruta.
El tamano pequeno es para que la busqueda por fuerza bruta (2^n configuraciones)
sea factible en CPU sin exceder 2 GiB de memoria.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def random_graph(n: int, p_edge: float = 0.5, seed: int = 0) -> List[Tuple[int, int]]:
    """Grafo aleatorio no dirigido con n nodos.

    Cada arista (i, j) con i < j se incluye con probabilidad p_edge.
    El resultado es una lista de aristas ordenadas (i, j) con i < j.
    """
    if n < 2:
        return []
    rng = torch.Generator().manual_seed(seed)
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(torch.rand(1, generator=rng).item()) < p_edge:
                edges.append((i, j))
    return edges


def complete_graph(n: int) -> List[Tuple[int, int]]:
    """Grafo completo K_n: todas las aristas (i, j) con i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def empty_graph(n: int) -> List[Tuple[int, int]]:
    """Grafo sin aristas."""
    return []


def cycle_graph(n: int) -> List[Tuple[int, int]]:
    """Grafo ciclo C_n: aristas (0,1), (1,2), ..., (n-1, 0)."""
    if n < 3:
        return complete_graph(n) if n == 2 else []
    return [(i, (i + 1) % n) for i in range(n)]


def maxcut_tensor(edges: List[Tuple[int, int]], n: int) -> torch.Tensor:
    """Etiqueta del dataset: corte maximo por fuerza bruta, como tensor escalar."""
    from .hamiltonian_maxcut import brute_force_maxcut
    return torch.tensor(brute_force_maxcut(edges, n), dtype=torch.float64)


def dataset(n_samples: int = 20, n: int = 5, p_edge: float = 0.5,
            seed: int = 0) -> Tuple[List[List[Tuple[int, int]]],
                                    torch.Tensor]:
    """Dataset de (grafos, cortes optimos).

    Devuelve (grafos, cortes) donde grafos[i] es la lista de aristas del i-esimo
    grafo y cortes[i] es el corte maximo por fuerza bruta.
    """
    from .hamiltonian_maxcut import brute_force_maxcut
    grafos: List[List[Tuple[int, int]]] = []
    cortes: List[float] = []
    for k in range(n_samples):
        edges = random_graph(n, p_edge, seed=seed + k)
        cortes.append(brute_force_maxcut(edges, n))
        grafos.append(edges)
    return grafos, torch.tensor(cortes, dtype=torch.float64)