"""
Decodador para el codigo de superficie (distancia d).

Para d=3 se usa una busqueda exhaustiva sobre correcciones de peso <= 1
(identidad más un error de Pauli en un solo qubit): dada una sindrome, se
elige la correccion de peso minimo que reproduce esa sindrome. Luego se
comprueba si error x correccion es un operador logico (falla del decodador).
"""

from __future__ import annotations

from typing import Optional, Tuple

from .codigo_superficie import SurfaceCode, pauli_string
from .sindromes import Pauli, sindrome, peso

# ---------------------------------------------------------------------------
# Algebra de Pauli (binaria, fase ignorada) para el test de operador logico.
# ---------------------------------------------------------------------------

def _pauli_bin(pauli: Pauli, n: int) -> Tuple[int, ...]:
    x = [0] * n
    z = [0] * n
    for q, p in pauli:
        if p in ('X', 'Y'):
            x[q] = 1
        if p in ('Z', 'Y'):
            z[q] = 1
    return tuple(x + z)


def _symp(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    m = len(a) // 2
    s = 0
    for i in range(m):
        s ^= (a[i] & b[m + i]) ^ (a[m + i] & b[i])
    return s == 0


def _gf2_rank(rows):
    rows = [list(r) for r in rows if any(r)]
    if not rows:
        return 0, []
    nc = len(rows[0])
    basis = []
    r = 0
    for c in range(nc):
        piv = None
        for i in range(r, len(rows)):
            if rows[i][c]:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(r + 1, len(rows)):
            if rows[i][c]:
                rows[i] = [a ^ bb for a, bb in zip(rows[i], rows[r])]
        basis.append(tuple(rows[r]))
        r += 1
        if r == len(rows):
            break
    return r, basis


def _in_span(v: Tuple[int, ...], basis) -> bool:
    return _gf2_rank([list(v)] + [list(b) for b in basis])[0] == len(basis)


def _mul_pauli(a: Pauli, b: Pauli, n: int) -> Pauli:
    """Producto de dos strings de Pauli (fase ignorada)."""
    da = dict(a)
    db = dict(b)
    out: dict = {}
    for q in set(da) | set(db):
        pa = da.get(q, 'I')
        pb = db.get(q, 'I')
        if pa == 'I':
            res = pb
        elif pb == 'I':
            res = pa
        elif pa == pb:
            res = 'I'
        else:
            # dos Paulis distintos no identidad -> el tercero
            res = ({'X', 'Y', 'Z'} - {pa, pb}).pop()
        if res != 'I':
            out[q] = res
    return tuple(sorted(out.items()))


# ---------------------------------------------------------------------------
# Tabla de correcciones de peso <= 1.
# ---------------------------------------------------------------------------

def _correcciones(code: SurfaceCode):
    """Lista de (correccion, sindrome, peso) para peso 0 y 1."""
    n = code.n
    corr = []
    # peso 0: identidad
    corr.append(((), sindrome((), code), 0))
    for q in range(n):
        for p in ('X', 'Y', 'Z'):
            c = ((q, p),)
            corr.append((c, sindrome(c, code), 1))
    return corr


_CORRECTION_CACHE: dict = {}


def _tabla(code: SurfaceCode):
    key = code.d
    if key not in _CORRECTION_CACHE:
        _CORRECTION_CACHE[key] = _correcciones(code)
    return _CORRECTION_CACHE[key]


def decodificar_sindrome(syndrome: Tuple[int, ...], code: SurfaceCode,
                         max_weight: int = 1) -> Optional[Pauli]:
    """Devuelve la correccion de peso <= max_weight que reproduce la sindrome.

    Devuelve None si no hay correccion de ese peso que coincida (sindrome
    no compatible con errores de peso <= max_weight).
    """
    mejor: Optional[Pauli] = None
    mejor_peso = max_weight + 1
    for corr, s, w in _tabla(code):
        if w > max_weight:
            continue
        if s == syndrome and w < mejor_peso:
            mejor = corr
            mejor_peso = w
    return mejor


def es_error_logico(error: Pauli, correction: Pauli, code: SurfaceCode) -> bool:
    """True si error x correction es un operador logico no trivial.

    Como la correccion reproduce la sindrome del error, error x correction
    conmuta con todos los estabilizadores. Es un fallo logico si no pertenece
    al grupo de estabilizadores.
    """
    producto = _mul_pauli(error, correction, code.n)
    if not producto:
        return False
    n = code.n
    stabs = [_pauli_bin(s, n) for s in code.x_stabilizers + code.z_stabilizers]
    _, basis = _gf2_rank(stabs)
    # producto conmuta con todos los estabilizadores (por construccion);
    # es fallo si no esta en el grupo de estabilizadores.
    return not _in_span(_pauli_bin(producto, n), basis)


def decodificar(error: Pauli, code: SurfaceCode,
                max_weight: int = 1) -> Tuple[Optional[Pauli], bool]:
    """Decodifica un error.

    Devuelve (correccion, es_fallo_logico).

    * Si no hay correccion de peso <= max_weight que reproduzca el sindrome,
      el error se *detecta* pero no se corrige con peso bajo: no es un fallo
      logico por sí mismo (correccion=None, es_fallo_logico=False).
    * Si se encuentra una correccion, es_fallo_logico es True solo cuando
      error x correccion es un operador logico no trivial (el decodador
      introdujo un error logico).
    """
    syndrome = sindrome(error, code)
    correction = decodificar_sindrome(syndrome, code, max_weight=max_weight)
    if correction is None:
        return None, False
    return correction, es_error_logico(error, correction, code)
