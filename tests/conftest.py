"""Configuracion comun para todos los tests.

Anyade entrega/ y entrega/simulador/ al path para que los imports
funcionen igual que en demo/app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ENTREGA = Path(__file__).resolve().parents[1]
_SIMULADOR = _ENTREGA / "simulador"

for p in (str(_ENTREGA), str(_SIMULADOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

torch.set_num_threads(2)
