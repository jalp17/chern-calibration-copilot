"""Categoria: LINDBLAD — matriz de densidad y ecuacion maestra de sistema abierto.

Pruebas:
  1. test_trace_conservado      - Tr(rho) = 1 en todo momento
  2. test_purity_unitaria       - pureza = 1 sin operadores de salto
  3. test_purity_decae          - pureza < 1 con disipacion
  4. test_positividad           - autovalores >= 0 siempre
  5. test_canal_despolarizante  - rho maximamente mezclado es punto fijo
  6. test_entropia_crece        - entropia de von Neumann no disminuye
"""

from __future__ import annotations

import torch

from simulador.lindblad import (DensityMatrix, decaimiento_amplitud,
                                desfasaje_puro, despolarizante,
                                lindblad_evolve, lindblad_step)
from simulador.lindblad.canal_depolarizante import aplicar_canal

torch.set_num_threads(2)

DT = 0.01
N_STEPS = 100
T_FINAL = DT * N_STEPS  # 1.0
TOL = 1e-6


def _hermitian_h(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((1 << n, 1 << n), generator=g, dtype=torch.float64)
    return (a + a.T) / 2.0


def test_trace_conservado():
    """T-L1: la traza de rho se conserva al longo de la evolucion."""
    n = 4
    d = 1 << n
    psi = torch.zeros(d, dtype=torch.complex128)
    psi[0] = 1.0
    rho0 = DensityMatrix.from_statevector(psi).rho
    H = _hermitian_h(n, seed=1)
    jump_ops = list(decaimiento_amplitud(0.1))
    _, historial = lindblad_evolve(rho0, H, jump_ops, T_FINAL, N_STEPS)
    for i, rho in enumerate(historial):
        tr = float(torch.trace(rho).real)
        assert abs(tr - 1.0) < 1e-8, f"paso {i}: trace = {tr}"


def test_purity_unitaria():
    """T-L2: evolucion unitaria (sin saltos) conserva pureza = 1."""
    n = 2
    d = 1 << n
    psi = torch.zeros(d, dtype=torch.complex128)
    psi[0] = 1.0
    rho = DensityMatrix.from_statevector(psi)
    H = _hermitian_h(n, seed=2)
    rho_final, _ = lindblad_evolve(rho.rho, H, [], T_FINAL, N_STEPS)
    dm = DensityMatrix(rho_final)
    assert abs(dm.purity() - 1.0) < 1e-8, f"purity unitaria = {dm.purity()}"


def test_purity_decae():
    """T-L3: disipacion pura (desfasaje) reduce la pureza por debajo de 1."""
    n = 1
    d = 2
    psi = torch.tensor([1.0, 1.0], dtype=torch.complex128) / (2.0 ** 0.5)
    rho = DensityMatrix.from_statevector(psi)
    H = torch.zeros((d, d), dtype=torch.complex128)
    jump_ops = list(desfasaje_puro(1.0))
    rho_final, _ = lindblad_evolve(rho.rho, H, jump_ops, T_FINAL, N_STEPS)
    dm = DensityMatrix(rho_final)
    assert dm.purity() < 1.0 - 1e-3, f"purity no decaio: {dm.purity()}"
    assert abs(dm.purity() - 0.5) < 0.05, f"purity final inesperada: {dm.purity()}"


def test_positividad():
    """T-L4: todos los autovalores de rho son >= 0 en cada paso."""
    n = 3
    d = 1 << n
    psi = torch.zeros(d, dtype=torch.complex128)
    psi[0] = 1.0
    rho0 = DensityMatrix.from_statevector(psi).rho
    H = _hermitian_h(n, seed=3)
    jump_ops = list(despolarizante(0.2))
    _, historial = lindblad_evolve(rho0, H, jump_ops, T_FINAL, N_STEPS)
    for i, rho in enumerate(historial):
        w = torch.linalg.eigvalsh(rho)
        assert bool(torch.all(w >= -1e-9)), f"paso {i}: autovalor negativo {w.min()}"


def test_canal_despolarizante():
    """T-L5: rho = I/2 es punto fijo del canal despolarizante."""
    d = 2
    rho_mix = torch.eye(d, dtype=torch.complex128) / d
    for p in (0.0, 0.1, 0.3, 1.0):
        rho_out = aplicar_canal(rho_mix, p)
        err = float((rho_out - rho_mix).abs().max())
        assert err < 1e-10, f"p={p}: I/2 no es punto fijo, err={err}"


def test_entropia_crece():
    """T-L6: con disipacion la entropia de von Neumann no disminuye."""
    n = 1
    d = 2
    psi = torch.tensor([1.0, 1.0], dtype=torch.complex128) / (2.0 ** 0.5)
    rho0 = DensityMatrix.from_statevector(psi)
    H = torch.zeros((d, d), dtype=torch.complex128)
    jump_ops = list(desfasaje_puro(2.0))
    _, historial = lindblad_evolve(rho0.rho, H, jump_ops, T_FINAL, N_STEPS)
    s0 = DensityMatrix(historial[0]).von_neumann_entropy()
    s1 = DensityMatrix(historial[-1]).von_neumann_entropy()
    assert s1 >= s0 - 1e-6, f"entropia disminuyo: {s0} -> {s1}"
    assert s1 > s0 + 1e-4, f"entropia no crecio: {s0} -> {s1}"
