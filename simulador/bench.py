"""
Benchmark del simulador statevector: CPU vs GPU.

El objetivo del hackathon es demostrar que la GPU de AMD es
load-bearing (cambia el resultado), no decorativa. Este script produce
esa evidencia.

Mide:
  1. Throughput de capas variationales segun nº de qubits
  2. Escalado de batch (ganancia real de GPU)
  3. Tiempo de gradiente (parameter-shift) y su coste
  4. Techo de memoria -> cuantos qubits caben en la VRAM

    python3 bench.py              # perfil por defecto
    python3 bench.py --sweep      # barrido completo (lento en CPU)
    python3 bench.py --sweep --device cuda
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from qsim_torch import LAYERS, Simulator
BYTES_PER_AMP = 16  # complex128


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()


def bench(fn, device: torch.device, warmup: int = 1, runs: int = 3) -> float:
    """Devuelve el mejor tiempo medio (seg) de `runs` iteraciones."""
    for _ in range(warmup):
        fn()
    sync(device)
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync(device)
        best = min(best, (time.perf_counter() - t0) / 1)
    return best


def make_workload(sim: Simulator, batch: int, layers: int = LAYERS):
    n = sim.n_qubits
    x = torch.randn(batch, n, dtype=torch.float64, device=sim.device)
    w = torch.randn(layers, n, dtype=torch.float64, device=sim.device,
                    requires_grad=True)

    def step():
        out = sim.variational_circuit(x, w)
        out.pow(2).mean().backward()
        w.grad = None

    return step


def mem_statevector(n: int) -> float:
    """GB que ocupa un unico statevector complex128 de n qubits."""
    return (1 << n) * BYTES_PER_AMP / 1024 ** 3


def fit_in_vram(total_gb: float, batch: int = 1, reserve: float = 0.65) -> int:
    """Mayor n cuyo lote de `batch` statevectors cabe en `reserve` de la VRAM."""
    n = 1
    while (1 << (n + 1)) * BYTES_PER_AMP * batch / 1024 ** 3 < total_gb * reserve:
        n += 1
    return n


def report_device(device: torch.device) -> None:
    print(f'device      : {device}')
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        print(f'GPU         : {p.name}')
        print(f'VRAM        : {p.total_memory / 1024 ** 3:.1f} GiB')
        print(f'ROCm/HIP    : {torch.version.hip}')
        print(f'capacidades : compute {p.major}.{p.minor}, multiprocessors {p.multi_processor_count}')
    else:
        print(f'torch       : {torch.__version__}')
        print(f'hilos       : {torch.get_num_threads()}')
    print()


def sweep(device: torch.device, ns: list[int], batches: list[int]) -> None:
    print(f'{"n":>4} {"batch":>6} {"capas/s":>10} {"amplitudes/s":>16} '
          f'{"GB est.":>9} {"s/capa":>10}')
    print('-' * 64)
    for n in ns:
        sim = Simulator(n, device=str(device))
        for b in batches:
            if mem_statevector(n) * b > 6.0:
                continue
            try:
                step = make_workload(sim, b)
                t = bench(step, device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                print(f'{n:>4} {b:>6} {"OOM/err":>10}  {str(exc)[:30]}')
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                continue
            layers = LAYERS * n
            amps = b * (1 << n)
            print(f'{n:>4} {b:>6} {layers / t:>10.1f} {amps / t:>16.3e} '
                  f'{mem_statevector(n) * b:>9.3f} {t / layers:>10.2e}')


def scaling_test(device: torch.device, n: int = 20) -> None:
    """Ganancia por batcheo: el motivo por el que la GPU gana de verdad."""
    print(f'\nEscalado de batch con n={n} qubits ({LAYERS} capas)')
    print(f'{"batch":>8} {"tiempo":>11} {"throughput":>14} {"speedup":>9}')
    print('-' * 46)
    sim = Simulator(n, device=str(device))
    base = None
    for b in (1, 8, 64, 256, 1024):
        if mem_statevector(n) * b > 4.0:
            print(f'{b:>8} {"excede RAM":>11}')
            continue
        try:
            step = make_workload(sim, b)
            t = bench(step, device, runs=2)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print(f'{b:>8} {"OOM":>11}')
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            continue
        per_sample = b / t
        if base is None:
            base = per_sample
        print(f'{b:>8} {t:>10.4f}s {per_sample:>14.1f} {per_sample / base:>8.2f}x')


def memory_report(device: torch.device) -> None:
    print('\nTecho de statevector (complex128, 16 B por amplitud)')
    print(f'{"n":>4} {"GB por estado":>14} {"GB @batch=256":>16}')
    print('-' * 38)
    for n in (20, 24, 26, 28, 30, 31, 32):
        per = mem_statevector(n)
        print(f'{n:>4} {per:>14.3f} {per * 256:>16.1f}')

    total = 48.0
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f'\nVRAM {total:.1f} GiB -> con 65% reservado:')
    for b in (1, 64, 256, 1024):
        print(f'  batch {b:>5}: hasta n = {fit_in_vram(total, b)} qubits')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sweep', action='store_true', help='barrido de qubits')
    ap.add_argument('--n', type=int, default=None, help='qubits para el perfil de batch')
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    report_device(device)

    gpu = device.type == 'cuda'

    if args.sweep:
        ns = [12, 14, 16] if not gpu else [16, 20, 24, 26, 28]
        batches = [1, 16, 64] if not gpu else [1, 256, 1024]
        sweep(device, ns, batches)
        scaling_test(device, n=16 if not gpu else 26)
    else:
        n = args.n or (18 if gpu else 10)
        scaling_test(device, n=n)

    memory_report(device)

    if gpu:
        print(f'\nPico de memoria reservado: '
              f'{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GiB')


if __name__ == '__main__':
    main()