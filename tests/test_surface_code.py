"""Tests para el codigo de superficie (Tarea 5 — surface code).

El codigo de distancia d=3 implementado es un codigo [[13, 1, 3]]: 13 qubits de
datos, 8 estabilizadores X (estrellas), 4 estabilizadores Z (plaquetas), un
qubit logico y distancia 3 (corrige un error, detecta dos).
"""

import itertools

from simulador.surface_code import (
    build_surface_code,
    sindrome,
    es_nulo,
    decodificar,
    tasa_error_logico,
)


def test_numero_estabilizadores():
    """Para d=3 hay 8 estrellas X y 4 plaquetas Z (12 estabilizadores, k=1)."""
    code = build_surface_code(3)
    assert code.n == 13, f'n esperada 13, obtenida {code.n}'
    assert len(code.x_stabilizers) == 8, \
        f'estrellas X esperadas 8, obtenidas {len(code.x_stabilizers)}'
    assert len(code.z_stabilizers) == 4, \
        f'plaquetas Z esperadas 4, obtenidas {len(code.z_stabilizers)}'


def test_sindrome_sin_error():
    """Sin errores, el sindrome es todos +1."""
    code = build_surface_code(3)
    assert es_nulo(sindrome((), code)), \
        'el sindrome sin errores debe ser todos +1'


def test_sindrome_error_unico():
    """Un error de bit-flip sobre un qubit de una plaqueta da sindrome no nulo."""
    code = build_surface_code(3)
    zqs = sorted({q for st in code.z_stabilizers for q, _ in st})
    assert zqs, 'debe haber al menos una plaqueta Z'
    q = zqs[0]
    assert not es_nulo(sindrome(((q, 'X'),), code)), \
        f'un bit-flip en el qubit {q} debe producir sindrome no nulo'


def test_decodador_corrige_error_unico():
    """El decodador corrige cualquier error de un solo qubit (peso 1)."""
    code = build_surface_code(3)
    for q in range(code.n):
        for p in ('X', 'Y', 'Z'):
            _, fallo = decodificar(((q, p),), code)
            assert not fallo, \
                f'el decodador no corrige el error {p} en el qubit {q}'


def test_tasa_error_logico_pequena_p():
    """Para p pequeno (0.01) la tasa logica P_L(p) es menor que p."""
    code = build_surface_code(3)
    pl = tasa_error_logico(code, 0.01, n_tiros=3000, seed=11)
    assert pl < 0.01, f'P_L(0.01)={pl:.4f} no es menor que 0.01'


def test_distancia_codigo():
    """d=3 corrige hasta 1 error y detecta hasta 2 (sin operadores logicos
    de peso 1 ni 2 fuera del grupo de estabilizadores)."""
    code = build_surface_code(3)
    from simulador.surface_code.decodador import (
        _pauli_bin, _gf2_rank, _in_span,
    )

    stabs = [_pauli_bin(s, code.n)
             for s in code.x_stabilizers + code.z_stabilizers]
    rank, basis = _gf2_rank(stabs)
    assert code.n - rank == 1, 'el codigo debe tener un qubit logico'

    # Ningun operador de peso 1 o 2 que conmuta con todos los estabilizadores
    # puede quedar fuera del grupo de estabilizadores (eso seria un operador
    # logico de peso <= 2, imposible en un codigo de distancia 3).
    for w in (1, 2):
        for qs in itertools.combinations(range(code.n), w):
            for ch in itertools.product('XYZ', repeat= w):
                e = tuple(sorted(zip(qs, ch)))
                s = sindrome(e, code)
                es_estabilizador = _in_span(_pauli_bin(e, code.n), basis)
                assert not (es_nulo(s) and not es_estabilizador), \
                    f'operador logico de peso {w}: {e}'

    # Ademas, todos los errores de peso 1 son corregidos por el decodador.
    for q in range(code.n):
        for p in ('X', 'Y', 'Z'):
            _, fallo = decodificar(((q, p),), code)
            assert not fallo, f'no se corrige error {p} q{q}'
