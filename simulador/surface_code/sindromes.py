"""
Calculo de sindromes de error para el codigo de superficie.

El sindrome de un error E sobre el estado del codigo (autovector +1 de todos
los estabilizadores) es el vector de eigenvalores de los estabilizadores:
    s_i = +1 si [E, S_i] = 0,  s_i = -1 si {E, S_i} = 0.
Un error de bit-flip (X) anticommuta con las plaquetas Z que lo contienen;
un error de phase-flip (Z) anticommuta con las estrellas X que lo contienen.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .codigo_superficie import SurfaceCode, pauli_string

Pauli = Tuple[Tuple[int, str], ...]


def _anticommute(pauli: Pauli, stabilizer: Pauli) -> bool:
    """Devuelve True si dos strings de Pauli anticommutan (comparten un numero
    impar de qubits sobre los cuales actuan con Paulis distintos e no-I)."""
    da = dict(pauli)
    db = dict(stabilizer)
    comunes = 0
    for q in da:
        if q in db and da[q] != db[q]:
            comunes += 1
    return comunes % 2 == 1


def sindrome(error: Pauli, code: SurfaceCode) -> Tuple[int, ...]:
    """Sindrome de un error sobre el codigo.

    Devuelve una tupla de +1/-1, un valor por estabilizador (primero los X,
    despues los Z). +1 = commuta (no detectado por ese estabilizador),
    -1 = anticommuta (detectado).
    """
    out = []
    for stab in code.x_stabilizers:
        out.append(-1 if _anticommute(error, stab) else 1)
    for stab in code.z_stabilizers:
        out.append(-1 if _anticommute(error, stab) else 1)
    return tuple(out)


def sindrome_con_error(code: SurfaceCode, error_qubits: Iterable[int],
                       tipo: str = 'X') -> Tuple[int, ...]:
    """Sindrome de un error que afecta a los qubits dados con el mismo Pauli."""
    error = tuple(sorted((q, tipo) for q in error_qubits))
    return sindrome(error, code)


def es_nulo(s: Sequence[int]) -> bool:
    """Sindrome nulo: todos los valores son +1 (no hay error detectable)."""
    return all(v == 1 for v in s)


def peso(pauli: Pauli) -> int:
    return len(pauli)


if __name__ == '__main__':
    from .codigo_superficie import build_surface_code
    code = build_surface_code(3)
    print('qubits:', code.n, 'X stars:', len(code.x_stabilizers),
          'Z plaquettes:', len(code.z_stabilizers))
    # Sindrome sin error
    print('sin error:', sindrome((), code), 'es_nulo:', es_nulo(sindrome((), code)))
    # Sindrome de un error de bit-flip en el qubit 0
    e = ((0, 'X'),)
    print('error X en q0:', sindrome(e, code), 'es_nulo:', es_nulo(sindrome(e, code)))
