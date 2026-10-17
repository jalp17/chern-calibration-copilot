# ============================================================
#  CELDA 2 — BENCHMARK v2
#  Corrige: (1) contabilidad de memoria de autograd, (2) progreso
#  visible, (3) tope de tiempo global.
#
#  La v1 fallaba porque BUDGET solo miraba el forward. PyTorch
#  guarda 4*LAYERS*n activaciones por elemento para el backward:
#  n=16 batch=1024 pedia 256 GB en una tarjeta de 48 GB.
#
#  La v2 separa las DOS preguntas legitimas:
#    A) ¿hasta donde puedo SIMULAR?   -> forward solo (no_grad)
#    B) ¿hasta donde puedo ENTRENAR?  -> forward + backward
#  Son limites muy distintos y confundirlos fue el error.
# ============================================================

import torch, time, math, sys

GB = 1024**3
TIME_CAP = 20.0        # segundos por punto; si se supera, se abandona
TOTAL_CAP = 600.0      # tope global de la celda

T0 = time.perf_counter()

def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def _t(fn, warmup=1, runs=3):
    for _ in range(warmup): fn()
    _sync()
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); _sync()
        best = min(best, time.perf_counter() - t0)
    return best

def real_dtype(cdtype):
    # torch.finfo(cdtype).dtype devuelve el STRING 'float64', no el dtype
    return getattr(torch, torch.finfo(cdtype).dtype)

# --- contabilidad de memoria -----------------------------------------
def mem_fwd(n, b):  return b * (1 << n) * 16 * 2 / GB
def mem_bwd(n, b):  return 4 * LAYERS * n * b * (1 << n) * 16 / GB

VRAM = (torch.cuda.get_device_properties(0).total_memory / GB
        if torch.cuda.is_available() else 4.0)
SAFE = VRAM * 0.80          # 80% del total: deja aire al allocator

def _free():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# --- workloads -------------------------------------------------------
def _dummy(sim):
    return torch.zeros(LAYERS, sim.n_qubits,
                       dtype=real_dtype(sim.dtype), device=sim.device)

def wl_fwd(sim, batch):
    dt = real_dtype(sim.dtype)
    x = torch.randn(batch, sim.n_qubits, dtype=dt, device=sim.device)
    w = _dummy(sim)
    def step():
        with torch.no_grad():
            sim.variational_circuit(x, w)
    return step

def wl_train(sim, batch):
    dt = real_dtype(sim.dtype)
    x = torch.randn(batch, sim.n_qubits, dtype=dt, device=sim.device)
    w = torch.randn(LAYERS, sim.n_qubits, dtype=dt,
                    device=sim.device, requires_grad=True)
    def step():
        sim.variational_circuit(x, w).pow(2).mean().backward()
        w.grad = None
    return step

def _fmt(n, b, t, amp, g):
    return (f'  n={n:>3} b={b:>5}  {t:>8.3f}s  {amp:>10.3e} amp/s'
            f'  {g:>6.2f} GiB')

# ================================================================
print('=' * 72)
print('A) SIMULAR — forward solo (no_grad)')
print('=' * 72)
if not torch.cuda.is_available():
    print('sin GPU; salta a la nota final')
else:
    p = torch.cuda.get_device_properties(0)
    print(f'GPU {p.name} | {VRAM:.0f} GiB | {p.multi_processor_count} CUs | '
          f'ROCm {torch.version.hip}')
    print(f'presupuesto seguro: {SAFE:.0f} GiB, tope {TOTAL_CAP:.0f}s\n')
    print(f'{"":<34}{"tiempo":>9}{"amp/s":>13}{"GiB":>8}')
    print('-' * 72)
    stopped = False
    for n in (14, 18, 20, 22, 24, 26, 28):
        sim = Simulator(n, 'cuda')
        for b in (1, 8, 32, 128, 512, 2048):
            if stopped or time.perf_counter() - T0 > TOTAL_CAP:
                stopped = True; break
            g = mem_fwd(n, b)
            if g > SAFE:
                print(f'  n={n:>3} b={b:>5}  --  {g:.1f} GiB excede {SAFE:.0f} GiB')
                continue
            _free()
            try:
                t = _t(wl_fwd(sim, b), warmup=1, runs=3)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f'  n={n:>3} b={b:>5}  OOM: {str(e)[:44]}')
                _free(); continue
            amp = b * (1 << n) / t
            print(_fmt(n, b, t, amp, g), flush=True)
            if t > TIME_CAP:
                print(f'  n={n:>3} b={b:>5}  -- {t:.1f}s > tope {TIME_CAP:.0f}s, paro aqui')
                stopped = True; break

# ================================================================
print()
print('=' * 72)
print('B) ENTRENAR — forward + backward (autograd guarda 4*LAYERS*n tensores)')
print('=' * 72)
if not torch.cuda.is_available():
    print('sin GPU')
else:
    print(f'{"":<34}{"tiempo":>9}{"amp/s":>13}{"bwd GiB":>9}')
    print('-' * 72)
    stopped = False
    for n in (10, 12, 14, 16, 18, 20):
        sim = Simulator(n, 'cuda')
        for b in (1, 4, 16, 64, 256):
            if stopped or time.perf_counter() - T0 > TOTAL_CAP:
                stopped = True; break
            g = mem_bwd(n, b)
            if g > SAFE:
                print(f'  n={n:>3} b={b:>5}  --  {g:.1f} GiB excede {SAFE:.0f} GiB')
                continue
            _free()
            try:
                t = _t(wl_train(sim, b), warmup=1, runs=2)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f'  n={n:>3} b={b:>5}  OOM: {str(e)[:44]}')
                _free(); continue
            amp = b * (1 << n) / t
            print(_fmt(n, b, t, amp, g), flush=True)
            if t > TIME_CAP:
                print(f'  n={n:>3} b={b:>5}  -- {t:.1f}s > tope, paro')
                stopped = True; break

# ================================================================
print()
print('=' * 72)
print('C) complex128 vs complex64 — ¿el FP64 de RDNA3 es el cuello?')
print('=' * 72)
if not torch.cuda.is_available():
    print('sin GPU')
else:
    print(f'{"":<26}{"c128":>10}{"c64":>10}{"ratio":>9}')
    print('-' * 55)
    for n, b in ((16, 32), (20, 8)):
        ts = {}
        for dt in (torch.complex128, torch.complex64):
            sim = Simulator(n, 'cuda', dtype=dt)
            try:
                ts[dt] = _t(wl_fwd(sim, b), warmup=1, runs=3)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                ts[dt] = float('nan'); _free()
        r = (ts[torch.complex128] / ts[torch.complex64]
             if ts[torch.complex64] == ts[torch.complex64] else float('nan'))
        print(f'  n={n:>3} b={b:>4}       {ts[torch.complex128]:>9.4f}'
              f' {ts[torch.complex64]:>9.4f} {r:>8.1f}x', flush=True)
    print('\n  ratio alto -> FP64 limita: usar complex64 o mixtura.')

# ================================================================
print()
print('=' * 72)
print('D) ¿Donde complex64 pierde precision? (mismos datos, dos dtypes)')
print('=' * 72)
if not torch.cuda.is_available():
    print('sin GPU')
else:
    print(f'{"n":>4}{"max err":>14}{"rel err":>12}')
    print('-' * 32)
    for n in (12, 16, 20, 24):
        g = torch.Generator(device='cpu').manual_seed(1234 + n)
        x64 = torch.randn(4, n, generator=g, dtype=torch.float64).cuda()
        w64 = torch.randn(LAYERS, n, generator=g, dtype=torch.float64).cuda()
        ref = Simulator(n, 'cuda', dtype=torch.complex128).variational_circuit(x64, w64)
        fast = Simulator(n, 'cuda', dtype=torch.complex64).variational_circuit(
            x64.float(), w64.float()).double()
        err = float((ref - fast).abs().max())
        print(f'{n:>4}{err:>14.3e}{err / float(ref.abs().max()):>12.3e}', flush=True)
    print('\n  rel err < 1e-5 -> complex64 aceptable para inferencia.')

if torch.cuda.is_available():
    print(f'\nPico VRAM reservado: '
          f'{torch.cuda.max_memory_reserved()/GB:.2f} GiB')
    print(f'Pico VRAM asignado : '
          f'{torch.cuda.max_memory_allocated()/GB:.2f} GiB')
    print(f'\nCelda completa en {time.perf_counter()-T0:.1f}s')
print('\nRECORDATORIO: pulsar "Turn-off Session" al terminar.')