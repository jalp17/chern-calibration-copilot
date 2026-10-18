"""Runner propio de tests — ejecuta todo y sale con codigo 0/1.

Uso:
    python tests/run_all.py              # todos los tests
    python tests/run_all.py correctness   # solo una categoria
    python tests/run_all.py -v           # salida detallada

Si pytest esta disponible, lo usa. Si no, ejecuta los tests directamente.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path

_ENTREGA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ENTREGA))
sys.path.insert(0, str(_ENTREGA / "simulador"))

CATEGORIAS = {
    'correctness': 'tests.test_correctness',
    'errores': 'tests.test_errores',
    'optimizacion': 'tests.test_optimizacion',
    'seguridad': 'tests.test_seguridad',
    'qaoa': 'tests.test_qaoa',
    'espacio_real': 'tests.test_espacio_real',
    'evolucion_temporal': 'tests.test_evolucion_temporal',
    'lindblad': 'tests.test_lindblad',
    'surface_code': 'tests.test_surface_code',
    'kernel': 'tests.test_kernel',
}


def _ejecutar_categoria(nombre: str, modulo: str) -> tuple[int, int, list[str]]:
    """Ejecuta todos los test_* de un modulo. Devuelve (pasados, fallados, errores)."""
    mod = importlib.import_module(modulo)
    tests = [
        (name, obj) for name, obj in inspect.getmembers(mod)
        if name.startswith('test_') and callable(obj)
    ]

    pasados = 0
    fallados = 0
    errores = []

    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            dt = time.perf_counter() - t0
            print(f'  PASS {name} ({dt:.2f}s)')
            pasados += 1
        except AssertionError as e:
            dt = time.perf_counter() - t0
            print(f'  FAIL {name} ({dt:.2f}s): {e}')
            fallados += 1
            errores.append(f'{name}: {e}')
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f'  ERROR {name} ({dt:.2f}s): {e}')
            fallados += 1
            errores.append(f'{name}: {e}')
            traceback.print_exc()

    return pasados, fallados, errores


def main() -> int:
    args = sys.argv[1:]
    verbose = '-v' in args
    args = [a for a in args if a != '-v']

    if args:
        categorias = {k: v for k, v in CATEGORIAS.items() if k in args}
    else:
        categorias = CATEGORIAS

    print('=' * 70)
    print('SISTEMA DE TESTING — Chern Calibration Copilot')
    print('=' * 70)

    total_pasados = 0
    total_fallados = 0
    todos_errores = []

    for nombre, modulo in categorias.items():
        print(f'\n--- {nombre.upper()} ---')
        try:
            p, f, errs = _ejecutar_categoria(nombre, modulo)
            total_pasados += p
            total_fallados += f
            todos_errores.extend(errs)
        except Exception as e:
            print(f'  ERROR cargando {modulo}: {e}')
            traceback.print_exc()
            total_fallados += 1

    print('\n' + '=' * 70)
    print(f'RESULTADO: {total_pasados} pasados, {total_fallados} fallados')
    print('=' * 70)

    if todos_errores:
        print('\nFALLOS:')
        for e in todos_errores:
            print(f'  - {e}')

    return 0 if total_fallados == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
