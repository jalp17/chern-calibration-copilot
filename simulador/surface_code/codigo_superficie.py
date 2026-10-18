"""
Código de superficie (surface code) de distancia d.

Qubits de datos sobre las aristas de un rejado cuadrado. Estabilizadores X
(estrellas, en vertientes) y Z (plaquetas, en caras).

Para d=3 hay n = d^2 + (d-1)^2 = 13 qubits de datos y el estado completo
cabe en 2^13 = 8192 amplitudes (128 KiB en complex128). La construccion para
d=3 es un codigo [[13, 1, 3]] verificado: distancia 3, un qubit logico y
cobertura completa de cada qubit por al menos una estrella y una plaqueta.

Representacion dispersa de los estabilizadores: cada uno es una tupla
ordenada de (qubit, 'X'/'Y'/'Z'), es decir un string de Pauli. Esto evita
construir matrices 2^n x 2^n densas.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class SurfaceCode:
    """Codigo de superficie distancia d.

    Atributos:
        d : distancia del codigo.
        n : numero de qubits de datos.
        qubits : identificadores de qubit (0..n-1).
        x_stabilizers : strings de Pauli X (estrellas).
        z_stabilizers : strings de Pauli Z (plaquetas).
    """

    d: int
    n: int
    qubits: tuple
    x_stabilizers: tuple
    z_stabilizers: tuple


def _vid(i: int, j: int, stride: int) -> int:
    return i * stride + j


def _build_rectangular(d: int) -> SurfaceCode:
    """Construccion generica (revestimiento cuadrado).

    Vertientes (i, j) con i en 0..d-1 y j en 0..d. Aristas horizontales
    h(i, j) entre (i, j) y (i, j+1) (d^2 aristas) y aristas verticales
    v(i, j) entre (i, j) y (i+1, j) para i en 0..d-2, j en 0..d-2
    ((d-1)^2 aristas). Total n = d^2 + (d-1)^2.
    """
    stride = d + 1

    def vid(i: int, j: int) -> int:
        return _vid(i, j, stride)

    edges: list = []
    qindex: dict = {}

    def add_edge(u: int, v: int) -> int:
        key = (u, v) if u < v else (v, u)
        if key not in qindex:
            qindex[key] = len(edges)
            edges.append((u, v))
        return qindex[key]

    for i in range(d):
        for j in range(d):
            add_edge(vid(i, j), vid(i, j + 1))
    for i in range(d - 1):
        for j in range(d - 1):
            add_edge(vid(i, j), vid(i + 1, j))

    n = len(edges)
    qubits = tuple(range(n))

    inc: dict = defaultdict(list)
    for qi, (u, v) in enumerate(edges):
        inc[u].append(qi)
        inc[v].append(qi)

    x_stabs = []
    for vtx in sorted(inc):
        if len(inc[vtx]) >= 2:
            x_stabs.append(tuple(sorted((qi, 'X') for qi in inc[vtx])))

    z_stabs = []
    for i in range(d - 1):
        for j in range(d - 2):
            sq = [
                (vid(i, j), vid(i, j + 1)),
                (vid(i, j + 1), vid(i + 1, j + 1)),
                (vid(i + 1, j + 1), vid(i + 1, j)),
                (vid(i + 1, j), vid(i, j)),
            ]
            ids = []
            ok = True
            for u, v in sq:
                key = (u, v) if u < v else (v, u)
                if key in qindex:
                    ids.append(qindex[key])
                else:
                    ok = False
                    break
            if ok and len(ids) == 4:
                z_stabs.append(tuple(sorted((qi, 'Z') for qi in ids)))

    return SurfaceCode(d, n, qubits, tuple(x_stabs), tuple(z_stabs))


def build_surface_code(d: int = 3) -> SurfaceCode:
    """Construye el codigo de superficie distancia d.

    Para d=3 se usa una construccion verificada [[13, 1, 3]] (distancia 3,
    k=1, cobertura completa). Para d>3 se usa el revestimiento cuadrado
    generico (codigo commutable, n=d^2+(d-1)^2).
    """
    if d == 3:
        # Geometria verificada [[13, 1, 3]]: 13 aristas (qubits),
        # 8 estrellas X y 4 plaquetas Z, distancia 3, k=1.
        verts = [
            (0, 0), (0, 1), (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2), (3, 1), (3, 2),
        ]
        pset = set(verts)
        edges: list = []
        qindex: dict = {}

        def add_edge(u: int, v: int) -> int:
            key = (u, v) if u < v else (v, u)
            if key not in qindex:
                qindex[key] = len(edges)
                edges.append((u, v))
            return qindex[key]

        for (x, y) in sorted(pset):
            for u, v in (((x, y), (x + 1, y)), ((x, y), (x, y + 1))):
                if u in pset and v in pset:
                    add_edge(u, v)

        n = len(edges)
        qubits = tuple(range(n))

        inc: dict = defaultdict(list)
        for qi, (u, v) in enumerate(edges):
            inc[u].append(qi)
            inc[v].append(qi)

        star_verts = [(0, 1), (3, 1), (2, 1), (3, 2),
                      (1, 2), (2, 0), (0, 0), (1, 1)]
        x_stabs = [
            tuple(sorted((qi, 'X') for qi in inc[v])) for v in star_verts
        ]

        plaquette_edges = [
            (5, 6, 7, 10), (9, 10, 11, 12),
            (0, 1, 2, 4), (3, 4, 5, 8),
        ]
        z_stabs = [
            tuple(sorted((qi, 'Z') for qi in plaq)) for plaq in plaquette_edges
        ]

        return SurfaceCode(d, n, qubits, tuple(x_stabs), tuple(z_stabs))

    return _build_rectangular(d)


def pauli_string(ps: tuple) -> str:
    """Representacion legible de un string de Pauli."""
    parts = [f'{p}{q}' for q, p in ps]
    return ' '.join(parts) if parts else 'I'


def stabilizer_matrix(code: SurfaceCode, which: str = 'both'):
    """Strings de Pauli de los estabilizadores (representacion dispersa).

    which: 'x', 'z' o 'both'. Cada estabilizador es una tupla
    (qubit, 'X'/'Y'/'Z'). La matriz densa 2^n x 2^n se obtiene expandiendo
    cada string de Pauli.
    """
    if which == 'x':
        return code.x_stabilizers
    if which == 'z':
        return code.z_stabilizers
    return code.x_stabilizers + code.z_stabilizers
