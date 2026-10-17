# ============================================================
#  CELDA 2 — BENCHMARK: ¿es la GPU load-bearing?
#  Pegar en una NUEVA celda (las funciones de la celda 1
#  siguen definidas en el kernel).
# ============================================================

import torch, time, math

def _sync(d):
    if d.type == 'cuda': torch.cuda.synchronize()

def _t(fn, dev, warmup=1, runs=3):
    """Mejor tiempo medio de `runs` vueltas."""
    for _ in range(warmup): fn()
    _sync(dev)
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); _sync(dev)
        best = min(best, time.perf_counter() - t0)
    return best

def real_dtype(complex_dtype):
    """Dtype real(float32/float64) que corresponde a un complejo.

    OJO: torch.finfo(torch.complex128).dtype devuelve el STRING 'float64',
    no el dtype. Hay que resolverlo con getattr.
    """
    return getattr(torch, torch.finfo(complex_dtype).dtype)

def workload(sim, batch, layers):
    n = sim.n_qubits
    dt = real_dtype(sim.dtype)
    assert dt in (torch.float32, torch.float64), f'dtype inesperado: {dt}'
    x = torch.randn(batch, n, dtype=dt, device=sim.device)
    w = torch.randn(layers, n, dtype=dt, device=sim.device, requires_grad=True)
    def step():
        sim.variational_circuit(x, w).pow(2).mean().backward()
        w.grad = None
    return step

def gb_statevector(n, batch, itemsize=16):
    return (1 << n) * itemsize * batch / 1024**3

# --- A) CPU vs GPU en la MISMA máquina ------------------------------
# La maquina del notebook tiene la W7900D. Comparemos contra su
# propio CPU: si gana aqui, gana en cualquier sitio.
BUDGET = 12.0   # GiB

def sweep(n_list, batch_list, device):
    dev = torch.device(device)
    rows = []
    for n in n_list:
        sim = Simulator(n, str(dev))
        for b in batch_list:
            if gb_statevector(n, b) > BUDGET:
                continue
            try:
                t = _t(workload(sim, b, LAYERS), dev, runs=2)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
                    print(f'  n={n:>3} b={b:>5}  OOM: {str(e)[:40]}')
                    continue
                raise
            layers = LAYERS * n
            rows.append((n, b, t, b * (1 << n) / t, layers / t,
                         gb_statevector(n, b)))
    return rows

NS   = [16, 20, 22, 24, 26]
BATS = [1, 64, 256, 1024]

print('=' * 78)
print('A) CPU vs GPU  —  misma maquina, 4 capas, complex128, fwd+bwd')
print('=' * 78)

gpu_rows = sweep(NS, BATS, 'cuda') if torch.cuda.is_available() else []
print(f'{"n":>4} {"batch":>6} {"GPU s":>10} {"GPU amp/s":>13} '
      f'{"CPU s":>10} {"speedup":>9} {"GiB":>7}')
print('-' * 78)

# El CPU de la maquina del notebook es lento: solo se mide donde el
# trabajo es pequeno. Saltar el resto en vez de colgarse.
CPU_MAX_N, CPU_MAX_B = 22, 64

for n, b, tg, amp, lay, g in gpu_rows:
    tc = float('nan')
    if n <= CPU_MAX_N and b <= CPU_MAX_B:
        sim = Simulator(n, 'cpu')
        try:
            tc = _t(workload(sim, b, LAYERS), torch.device('cpu'),
                   warmup=0, runs=1)
        except RuntimeError:
            tc = float('nan')
    sp = tc / tg if tg and tc == tc else float('nan')
    note = '' if tc == tc else '   (CPU omitido: demasiado lento)'
    print(f'{n:>4} {b:>6} {tg:>10.4f} {amp:>13.3e} '
          f'{tc:>10.4f} {sp:>8.1f}x {g:>7.2f}{note}')

if gpu_rows:
    best = max(gpu_rows, key=lambda r: r[3])
    print(f'\n  mejor: n={best[0]} batch={best[1]} -> '
          f'{best[3]:.3e} amplitudes/s  ({best[5]:.2f} GiB)')

# --- B) complex128 vs complex64 --------------------------------------
# RDNA3 tiene FP64 a 1/64 de FP32. Si el simulador esta limitado
# por FP64, complex64 dara mucho mas. Esto decide el dtype del
# proyecto entero.
print()
print('=' * 78)
print('B) complex128 vs complex64  — ¿el FP64 de RDNA3 es el cuello?')
print('=' * 78)
print(f'{"n":>4} {"batch":>6} {"c128 s":>10} {"c64 s":>10} {"ratio":>8}')
print('-' * 78)

if torch.cuda.is_available():
    for n, b in [(20, 256), (22, 128), (24, 64)]:
        ts = {}
        for dt in (torch.complex128, torch.complex64):
            sim = Simulator(n, 'cuda', dtype=dt)
            try:
                ts[dt] = _t(workload(sim, b, LAYERS),
                            torch.device('cuda'), runs=3)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                ts[dt] = float('nan')
                torch.cuda.empty_cache()
        r = ts[torch.complex128] / ts[torch.complex64] \
            if ts[torch.complex64] == ts[torch.complex64] else float('nan')
        print(f'{n:>4} {b:>6} {ts[torch.complex128]:>10.4f} '
              f'{ts[torch.complex64]:>10.4f} {r:>7.1f}x')
    print('\n  ratio alto -> FP64 limita: usar complex64 (o mixtura).')

# --- C) precision vs velocidad: donde complex64 se rompe ------------
print()
print('=' * 78)
print('C) ¿Donde complex64 pierde precision?  (gate fidelity vs c128)')
print('=' * 78)
if torch.cuda.is_available():
    print(f'{"n":>4} {"max err":>12} {"rel err":>11}')
    print('-' * 30)
    # ENTRADAS IDENTICAS en ambos casos: si no, se comparan dos
    # circuitos distintos y el "error" mide ruido, no precision.
    # Se generan en float64 con semilla fija y se castean a float32.
    for n in [14, 18, 22, 24]:
        g = torch.Generator(device='cpu').manual_seed(1234 + n)
        x64 = torch.randn(4, n, generator=g, dtype=torch.float64)
        w64 = torch.randn(LAYERS, n, generator=g, dtype=torch.float64)

        ref = Simulator(n, 'cuda', dtype=torch.complex128).variational_circuit(
            x64.cuda(), w64.cuda())

        x32, w32 = x64.float().cuda(), w64.float().cuda()
        fast = Simulator(n, 'cuda', dtype=torch.complex64).variational_circuit(
            x32, w32).double()

        err = float((ref - fast).abs().max())
        scale = float(ref.abs().max())
        print(f'{n:>4} {err:>12.3e} {err / scale:>11.3e}')
    print('\n  si rel err < 1e-5 -> complex64 es aceptable para inferencia.')

if torch.cuda.is_available():
    print(f'\nPico VRAM reservado: {torch.cuda.max_memory_reserved()/1024**3:.2f} GiB')
    print(f'Pico VRAM asignado: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB')

# RECORDATORIO: pulsar "Turn-off Session" al terminar.