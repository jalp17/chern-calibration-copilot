"""
Paquete de codigo de superficie (surface code) para el simulador.

Modulos:
  codigo_superficie - construccion del codigo y estabilizadores dispersos
  sindromes         - calculo de sindromes de error
  ruido_codigo      - modelo de ruido fisico (X, Y, Z por qubit)
  decodador         - decodador exhaustivo peso <= 1 (d=3) y test logico
  tasa_error_logico - P_L(p) por Monte Carlo

Uso minimo:
    from simulador.surface_code import build_surface_code, sindrome, decodificar, tasa_error_logico
    code = build_surface_code(3)
    s = sindrome(((0, 'X'),), code)
    correccion, fallo = decodificar(((0, 'X'),), code)
"""

from __future__ import annotations

from .codigo_superficie import (SurfaceCode, build_surface_code,
                                pauli_string, stabilizer_matrix)
from .sindromes import (sindrome, sindrome_con_error, es_nulo, peso)
from .ruido_codigo import (muestrear_error, aplicar_error_estado)
from .decodador import (decodificar, decodificar_sindrome, es_error_logico)
from .tasa_error_logico import (tasa_error_logico, curva_tasa_error_logico,
                                umbral_teorico)

__all__ = [
    'SurfaceCode',
    'build_surface_code',
    'pauli_string',
    'stabilizer_matrix',
    'sindrome',
    'sindrome_con_error',
    'es_nulo',
    'peso',
    'muestrear_error',
    'aplicar_error_estado',
    'decodificar',
    'decodificar_sindrome',
    'es_error_logico',
    'tasa_error_logico',
    'curva_tasa_error_logico',
    'umbral_teorico',
]
