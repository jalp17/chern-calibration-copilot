"""Categoria C: OPTIMIZACION (medidas, no afirmaciones).

Tests que MIDEN y REPORTAN. No fallan por lentitud, fallan si empeora
un umbral. Toda medicion se imprime para el reporte.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from qsim_torch import Simulator, ry, rotation_z, state_fidelity
from barrido_v2 import fhs, prepare, bz
from haldane import honeycomb, occupied_band, hamiltonian

D = torch.complex128


def _timeit(fn, warmup: int = 1, runs: int = 3) -> float:
    """Mejor tiempo medio (seg) de `runs` vueltas."""
    for _ in range(warmup):
        fn()
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# --- C1: speedup de apply_rz_all vs bucle de RY --------------------------

def test_speedup_rz_all_vs_bucle_ry():
    """T-C1: apply_rz_all debe ser >1.5x mas rapido que bucle de RY."""
    n = 6
    batch = 32
    sim = Simulator(n)
    g = torch.Generator().manual_seed(0)
    psi = torch.randn(batch, 1 << n, generator=g, dtype=D)
    psi = psi / psi.norm(dim=1, keepdim=True)
    theta = torch.randn(n, generator=g, dtype=torch.float64)

    def bucle_ry():
        out = psi.clone()
        for q in range(n):
            out = sim._apply_1q(out, ry(theta[q], D), q)
        return out

    def rz_all():
        return sim.apply_rz_all(psi, theta)

    t_bucle = _timeit(bucle_ry, warmup=2, runs=5)
    t_rz = _timeit(rz_all, warmup=2, runs=5)
    speedup = t_bucle / t_rz

    print(f'  C1  bucle RY: {t_bucle:.4f}s | apply_rz_all: {t_rz:.4f}s | speedup: {speedup:.2f}x')
    assert speedup > 1.5, f'speedup {speedup:.2f}x < 1.5x'


# --- C2: speedup de measure_z_all vs bucle de measure_z ------------------

def test_speedup_measure_z_all_vs_bucle():
    """T-C2: measure_z_all debe ser mas rapido que bucle de measure_z."""
    n = 8
    batch = 64
    sim = Simulator(n)
    g = torch.Generator().manual_seed(0)
    psi = torch.randn(batch, 1 << n, generator=g, dtype=D)
    psi = psi / psi.norm(dim=1, keepdim=True)

    def bucle():
        return torch.stack([sim.measure_z(psi, q) for q in range(n)], dim=1)

    def vectorial():
        return sim.measure_z_all(psi)

    t_bucle = _timeit(bucle, warmup=2, runs=5)
    t_vec = _timeit(vectorial, warmup=2, runs=5)
    speedup = t_bucle / t_vec

    print(f'  C2  bucle measure_z: {t_bucle:.4f}s | measure_z_all: {t_vec:.4f}s | speedup: {speedup:.2f}x')
    assert speedup > 1.0, f'speedup {speedup:.2f}x <= 1.0x'


# --- C3: memoria de un lote ----------------------------------------------

def test_memoria_lote_formula():
    """T-C3: memoria de un lote = nseeds*Nk^2*2^n*16 bytes (±20%)."""
    n = 4
    Nk = 4
    nseeds = 8
    B = nseeds * Nk * Nk

    teórico_bytes = B * (1 << n) * 16
    teórico_gb = teórico_bytes / 1024**3

    net = honeycomb()
    ks = bz(net, Nk)
    u = occupied_band(hamiltonian(net, ks, 0.0, 1.0, 0.20, math.pi / 2))
    phi = torch.tensor(u.reshape(Nk * Nk, 2), dtype=D)
    phi_rep = phi.repeat(nseeds, 1)

    sim = Simulator(n)
    with torch.no_grad():
        psi = prepare(sim, phi_rep, 2, None, 'indep')

    real_bytes = psi.numel() * psi.element_size()
    real_gb = real_bytes / 1024**3

    ratio = real_bytes / teórico_bytes
    print(f'  C3  teorico: {teórico_gb:.4f} GiB | real: {real_gb:.4f} GiB | ratio: {ratio:.3f}')
    assert 0.8 <= ratio <= 1.2, f'ratio {ratio:.3f} fuera de [0.8, 1.2]'


# --- C4: complex64 vs complex128 -----------------------------------------

def test_complex64_vs_complex128_precision():
    """T-C4: error relativo de complex64 vs complex128 < 1e-5."""
    n = 4
    g = torch.Generator().manual_seed(7)
    x64 = torch.randn(8, n, generator=g, dtype=torch.float64)
    w64 = torch.randn(4, n, generator=g, dtype=torch.float64)

    ref = Simulator(n, 'cpu', dtype=torch.complex128).variational_circuit(x64, w64)
    fast = Simulator(n, 'cpu', dtype=torch.complex64).variational_circuit(
        x64.float(), w64.float())

    err = float((ref - fast.double()).abs().max())
    scale = float(ref.abs().max())
    rel = err / scale

    print(f'  C4  complex64 vs complex128: abs {err:.3e} | rel {rel:.3e}')
    assert rel < 1e-5, f'error relativo {rel:.2e} >= 1e-5'


# --- C5: overhead de RSS sobre el lote teórico ----------------------------

def test_rss_overhead():
    """T-C5: overhead de RSS sobre el lote teorico (reportar ratio)."""
    import os
    import psutil

    n = 4
    Nk = 4
    nseeds = 8
    B = nseeds * Nk * Nk

    teórico_bytes = B * (1 << n) * 16

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss

    net = honeycomb()
    ks = bz(net, Nk)
    u = occupied_band(hamiltonian(net, ks, 0.0, 1.0, 0.20, math.pi / 2))
    phi = torch.tensor(u.reshape(Nk * Nk, 2), dtype=D)
    phi_rep = phi.repeat(nseeds, 1)

    sim = Simulator(n)
    with torch.no_grad():
        psi = prepare(sim, phi_rep, 2, None, 'indep')

    rss_after = process.memory_info().rss
    overhead_bytes = rss_after - rss_before
    ratio = overhead_bytes / teórico_bytes

    print(f'  C5  teorico: {teórico_bytes/1024**2:.2f} MiB | '
          f'RSS overhead: {overhead_bytes/1024**2:.2f} MiB | ratio: {ratio:.2f}')
    # El overhead razonable es < 10x el lote (torch, python, etc.)
    assert ratio < 10.0, f'overhead {ratio:.2f}x demasiado alto'
