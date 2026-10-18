"""Tests para la evolucion temporal de Haldane (ecuacion de Schrodinger td).

Valida el integrador Trotter de primer orden (exacto porque [H(k), H(k')] = 0)
y los diagnosticos de `simulador/evolucion_temporal/`.

Todas las pruebas corren en CPU con n <= 4 qubits (statevector <= 256
amplitudes, < 4 MiB) y sin acceso a red.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.linalg import expm

from qsim_torch import Simulator
from haldane import honeycomb
from evolucion_temporal import (
    HaldaneTD,
    EvolucionTD,
    estado_producto_A,
    estado_localizado,
    norma,
    transferencia_carga,
    chern_dinamico,
    polarizacion_berry,
)

D = torch.complex128


def _build(n_qubits: int = 4, dt: float = 0.01,
           M_start: float = -1.5, M_end: float = 1.5,
           T: float = 1.0, modo: str = "rampa"):
    net = honeycomb()
    sim = Simulator(n_qubits)
    ham = HaldaneTD(net=net, M_start=M_start, M_end=M_end, T=T, modo=modo)
    evol = EvolucionTD(sim, ham, dt=dt)
    return sim, ham, evol


def test_norma_conservada():
    """||psi|| = 1 tras N pasos de evolucion con H(t) variando (rampa)."""
    sim, _, evol = _build()
    psi0 = estado_producto_A(sim)
    psi, _ = evol.evolucionar(psi0, t0=0.0, n_steps=200)
    n = norma(psi).item()
    assert abs(n - 1.0) < 1e-9, f'norma={n:.6f} (esperado 1)'


def test_evolucion_unitaria_H_constante():
    """H constante: el paso de Trotter coincide con expm(-i H t) (error de maquina)."""
    net = honeycomb()
    sim = Simulator(4)
    ham = HaldaneTD(net=net, M_start=0.0, M_end=0.0, T=1.0, modo="rampa")  # M constante
    evol = EvolucionTD(sim, ham, dt=0.01)
    psi0 = estado_producto_A(sim)
    psi, _ = evol.evolucionar(psi0, t0=0.0, n_steps=10)  # t_total = 0.1
    H_full = evol.hamiltoniano_completo(0.0)
    U = expm(-1j * H_full * 0.1)
    psi_ref = torch.tensor(U @ psi0[0].numpy(), dtype=sim.dtype).unsqueeze(0)
    err = (psi - psi_ref).abs().max().item()
    assert err < 1e-10, f'Trotter vs expm: error {err:.2e}'


def test_paso_pequeno_lineal():
    """dt pequeno: evolucion ~= identidad, variacion lineal en primer orden en dt."""
    net = honeycomb()
    sim = Simulator(4)
    ham = HaldaneTD(net=net, M_start=0.0, M_end=0.0, T=1.0, modo="rampa")
    psi0 = estado_producto_A(sim)
    deltas = []
    for dt in (0.1, 0.05):
        evol = EvolucionTD(sim, ham, dt=dt)
        p1, _ = evol.evolucionar(psi0, t0=0.0, n_steps=1)
        deltas.append((p1 - psi0).abs().max().item())
    ratio = deltas[0] / deltas[1]
    assert 1.5 < ratio < 2.5, \
        f'escalado lineal en dt: ratio={ratio:.2f} (esperado ~2)'


def test_estado_localizado_h0():
    """H efectivamente nulo (dt = 0): el estado localizado no cambia."""
    net = honeycomb()
    sim = Simulator(4)
    ham = HaldaneTD(net=net, M_start=0.0, M_end=0.0, T=1.0, modo="rampa")
    evol = EvolucionTD(sim, ham, dt=0.0)
    psi0 = estado_localizado(sim, 0, net, sublattice=0, N_mesh=2)
    psi, _ = evol.evolucionar(psi0, t0=0.0, n_steps=10)
    assert (psi - psi0).abs().max().item() == 0.0, \
        'estado con dt = 0 deberia ser invariante'


def test_chern_dinamico_rampa():
    """Chern dinamico es real finito, acotado en (-1, 1) y es P_final - P_inicial."""
    sim, _, evol = _build()
    psi0 = estado_producto_A(sim)
    pf, hist = evol.evolucionar(psi0, t0=0.0, t1=1.0, n_steps=100,
                                guardar_historial=True)
    Q = chern_dinamico(hist, sim, evol.k_points)
    assert np.isfinite(Q), f'chern dinamico no finito: {Q}'
    assert -1.0 < Q < 1.0, f'chern dinamico fuera de (-1, 1): {Q}'
    Pi = polarizacion_berry(hist[0][1], sim, evol.k_points)
    Pf = polarizacion_berry(hist[-1][1], sim, evol.k_points)
    assert abs(Q - (Pf - Pi)) < 1e-12, \
        f'chern dinamico != P_final - P_inicial: {Q} vs {Pf - Pi}'


def test_transferencia_carga():
    """Transferencia de carga finita y acotada en [0, n_qubits]; dt = 0 la conserva."""
    net = honeycomb()
    sim = Simulator(4)
    ham = HaldaneTD(net=net, M_start=-1.5, M_end=1.5, T=1.0, modo="rampa")
    evol = EvolucionTD(sim, ham, dt=0.01)
    psi0 = estado_producto_A(sim)
    pf, _ = evol.evolucionar(psi0, t0=0.0, n_steps=50)
    tr = transferencia_carga(pf, sim).item()
    assert np.isfinite(tr), f'transferencia no finita: {tr}'
    assert 0.0 <= tr <= sim.n_qubits, \
        f'transferencia fuera de [0, {sim.n_qubits}]: {tr}'
    e0 = EvolucionTD(sim, ham, dt=0.0)
    p0, _ = e0.evolucionar(psi0, t0=0.0, n_steps=10)
    t0 = transferencia_carga(psi0, sim).item()
    t1 = transferencia_carga(p0, sim).item()
    assert abs(t1 - t0) < 1e-12, f'dt=0 no conserva transferencia: {t0} -> {t1}'
