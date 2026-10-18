"""
Tasa de error logico P_L(p) para el codigo de superficie.

Simula muchos tiros: para cada uno se genera un error aleatorio con
probabilidad p por qubit, se calcula el sindrome, se decodifica y se cuenta
si el decodador comete un error logico. P_L(p) = fracasos / tiros.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .codigo_superficie import SurfaceCode
from .decodador import decodificar
from .ruido_codigo import muestrear_error


def tasa_error_logico(code: SurfaceCode, p: float, n_tiros: int = 2000,
                      seed: int = 0, max_weight: int = 1) -> float:
    """Estima P_L(p) por Monte Carlo.

    Devuelve la fracción de tiros en que el decodador falla (error logico).
    """
    fallos = 0
    for tiro in range(n_tiros):
        error = muestrear_error(code, p, seed=seed + tiro)
        _, fallo = decodificar(error, code, max_weight=max_weight)
        if fallo:
            fallos += 1
    return fallos / n_tiros


def curva_tasa_error_logico(code: SurfaceCode, ps: List[float],
                            n_tiros: int = 1000, seed: int = 0,
                            max_weight: int = 1) -> Dict[float, float]:
    """Evalua P_L(p) para varios valores de p. Devuelve {p: P_L}."""
    return {p: tasa_error_logico(code, p, n_tiros=n_tiros, seed=seed,
                                 max_weight=max_weight) for p in ps}


def umbral_teorico() -> float:
    """Umbral teorico aproximado del codigo de superficie (ruido independiente).

    Para el codigo de superficie con ruido fenomenologico el umbral esta
    alrededor del 1 %; para ruido independiente por compuerta es menor.
    Se devuelve como referencia, no como resultado de la simulacion.
    """
    return 0.01
