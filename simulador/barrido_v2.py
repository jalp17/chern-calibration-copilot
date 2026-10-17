"""
BARRIDO v2 — fragilidad del Chern ante error de puerta.

CORRECCIONES respecto a v1 (revisadas una a una):

 1. FHS corre en el DISPOSITIVO, no en CPU. En v1 estaba en numpy con
    `.cpu().numpy()`, o sea que la GPU solo hacia el circuito, que es la
    parte pequena. Aqui todo va en torch.
 2. Las SEMILLAS se batchean: una sola llamada con
    (Nk^2 x nseeds) estados en vez de un bucle serial. Esto es lo que
    hace que el lote no quepa en la CPU.
 3. Infidelity con DOS estimadores:
      - muestreada (ruido de muestreo, malo para p pequeno)
      - analitica E[F] = (1 - p/2)^(n_puertas), varianza CERO
 4. DOS MODELOS DE RUIDO, declarados y comparados:
      - 'indep'  : un Pauli por (puerta, k). Cada preparacion es una
                   ejecucion distinta. Es lo fisico en un dispositivo.
      - 'shared' : un Pauli por puerta, igual para todos los k.
                   Corresponde a un unitario fijo al final, que
                   CONSERVA EL CHERN EXACTAMENTE (verificado).
 5. Barras de error: error estandar de C y de la fraccion correcta.

Memoria: el lote es Nk^2 x nseeds estados de 2^n en complex128.

Uso:
  python3 barrido_v2.py --n 10 --Nk 16 --nseeds 256 --device cuda
  python3 barrido_v2.py --n 6 --Nk 8  --nseeds 64  --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from haldane import critical_mass, gap_at_dirac, honeycomb, hamiltonian, occupied_band
from qsim_torch import Simulator, ry, rotation_z, state_fidelity
from ruido import PAULI_1Q

PAULI_STACK = torch.stack([p for p in PAULI_1Q])          # (4,2,2)


# ---------------------------------------------------------------- FHS torch
def fhs(states: torch.Tensor):
    """Chern por FHS, en el dispositivo.

    states: (..., N1, N2, d). Los dos ultimos ejes de malla son los dos
    ultimos antes de las amplitudes.

        U1[i,j] = <u[i,j] | u[i+1,j]>
        U2[i,j] = <u[i,j] | u[i,j+1]>
        plaq    = U1[i,j] U2[i+1,j] U1[i,j+1]* U2[i,j]*

    Devuelve (C, min|overlap|). NaN si algun solapamiento se anula.
    """
    u = states
    U1 = (u.conj() * torch.roll(u, -1, dims=-3)).sum(-1)   # (..., N1, N2)
    U2 = (u.conj() * torch.roll(u, -1, dims=-2)).sum(-1)   # (..., N1, N2)

    n1, n2 = U1.abs(), U2.abs()
    mn = torch.minimum(n1, n2).amin(dim=(-2, -1))       # (lote,)

    a = U1 / n1
    U2s = torch.roll(U2, -1, dims=-2) / torch.roll(n2, -1, dims=-2)
    U1t = torch.roll(U1, -1, dims=-1)
    b = U1t.conj() / torch.roll(n1, -1, dims=-1)
    c = U2.conj() / n2
    plaq = a * U2s * b * c
    # SUMAR SOLO sobre los ejes de malla (-2, -1), NO sobre el lote.
    # Antes sumaba todo y devolvia -16 en vez de -1.
    C = torch.angle(plaq).sum(dim=(-2, -1)) / (2.0 * math.pi)
    C = torch.where(mn < 1e-12, torch.full_like(C, float('nan')), C)
    return C, mn


# ------------------------------------------------------- ruido batcheado
def sample_pauli_idx(B: int, n_gates: int, p: float, gen: torch.Generator,
                     device: torch.device) -> torch.Tensor:
    """Indices de Pauli (B, n_gates) en {0,1,2,3} con distribucion del
    canal depolarizante: I con 1-p, y X/Y/Z con p/3 cada uno."""
    if p <= 0:
        return torch.zeros(B, n_gates, dtype=torch.long, device=device)
    r = torch.rand(B, n_gates, generator=gen, device=device)
    idx = torch.zeros(B, n_gates, dtype=torch.long, device=device)
    idx = torch.where(r < p / 3, torch.ones_like(idx), idx)
    idx = torch.where((r >= p / 3) & (r < 2 * p / 3), torch.full_like(idx, 2), idx)
    idx = torch.where((r >= 2 * p / 3) & (r < p), torch.full_like(idx, 3), idx)
    return idx


def prepare(sim, phi_k, G, idx=None, noise='indep'):
    """Prepara |phi_k> en el qubit 0 con G pasos RY(theta/G) y un RZ(gamma).

    phi_k: (B, 2) estados objetivo.
    idx:   (B, G+1) indices de Pauli, o None.
    """
    B = phi_k.shape[0]
    theta = 2.0 * torch.atan2(phi_k[:, 1].abs(), phi_k[:, 0].abs()).real
    gamma = torch.angle(phi_k[:, 1]) - torch.angle(phi_k[:, 0])
    step = theta / G
    psi = sim.zeros(B)
    pg = PAULI_STACK.to(sim.device)

    def apply_noise(state, g):
        if idx is None:
            return state
        return sim._apply_1q(state, pg[idx[:, g]], 0)

    if noise == 'final':
        # Pauli DESPUES de todas las puertas: equivale a un unitario fijo
        # U, y <U psi_k | U psi_k'> = <psi_k | psi_k'> EXACTAMENTE.
        # Es el CONTROL: debe dar C = C_ideal a cualquier p.
        for g in range(G):
            psi = sim._apply_1q(psi, ry(step, sim.dtype), 0)
        psi = sim._apply_1q(psi, rotation_z(gamma, sim.dtype), 0)
        if idx is not None:
            psi = sim._apply_1q(psi, pg[idx[:, 0]], 0)
        return psi

    for g in range(G):
        psi = sim._apply_1q(psi, ry(step, sim.dtype), 0)
        psi = apply_noise(psi, g)
    psi = sim._apply_1q(psi, rotation_z(gamma, sim.dtype), 0)
    psi = apply_noise(psi, G)
    return psi


# ------------------------------------------------------------------- malla
def bz(net, N):
    u = (np.arange(N) + 0.5) / N
    uu, vv = np.meshgrid(u, u, indexing='ij')
    return (uu[..., None] * net['G1'] + vv[..., None] * net['G2'])


def n_noisy_gates(G: int, model: str) -> int:
    """Puertas que reciben ruido en cada modelo."""
    return 1 if model == 'final' else G + 1


def analytic_fidelity(p: float, n_gates: int) -> float:
    """COTA de fidelidad (1-p/2)^n -- NO es la fidelidad del estado real.

    Es la fidelidad media de un canal depolarizing sobre el estado
    MAXIMALLY MIXED (I/2), donde el depolarizing se comporta como un
    canal que multiplica la fidelidad por (1-p).

    Verificado por enumeracion exacta de los 4^n subconjuntos de Pauli
    (estado |0>): el valor real es ~3% MAS OPTIMISTA que esta cota para
    p<=0.1, y ~18% mas pesimista para p=0.3. La cota no sustituye a la
    fidelidad muestreada; se usa solo como linea de referencia.
    """
    return (1.0 - p / 2.0) ** n_gates


def run(a):
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    net = honeycomb()
    sim = Simulator(a.n, str(dev))
    N = a.Nk
    MC = critical_mass(0.0, 1.0, 0.20, math.pi / 2)

    ks = bz(net, N)
    u_ex = {nom: occupied_band(hamiltonian(net, ks, M, 1.0, 0.20, math.pi / 2))
            for nom, M in (('trivial', 1.6 * MC), ('topologica', 0.5 * MC))}

    B = N * N * a.nseeds
    mem = B * (1 << a.n) * 16 / 1024 ** 3
    print(f'dispositivo {dev}' + (f' | {torch.cuda.get_device_properties(0).name}'
          if dev.type == 'cuda' else ''))
    print(f'n={a.n}  G={a.G}  malla {N}x{N}  semillas={a.nseeds}  '
          f'puertas con ruido/modelo: indep={a.G+1}, final=1')
    print(f'M_crit={MC:.6f}  |d_z(K)|={gap_at_dirac(net,0,1,0.20,math.pi/2):.6f}')
    print(f'lote: {B:,} estados = {mem:.2f} GiB (complex128)')
    if dev.type == 'cpu' and mem > 3.0:
        print(f'  AVISO: {mem:.1f} GiB no cabe en la CPU de 4 GiB')
    print()

    gen = torch.Generator(device=dev).manual_seed(a.seed)
    out = {'n': a.n, 'G': a.G, 'Nk': N, 'nseeds': a.nseeds,
           'M_crit': MC, 'modelos': {}}

    for model in a.modelos:
        out['modelos'][model] = {}
        print('=' * 86)
        print(f'MODELO DE RUIDO: {model}')
        print('=' * 86)
        for nombre, u in u_ex.items():
            C_ex = int(round(float(fhs(torch.tensor(u, dtype=torch.complex128, device=dev))[0])))
            phi = torch.tensor(u.reshape(N * N, 2), dtype=torch.complex128, device=dev)
            phi_rep = phi.repeat(a.nseeds, 1)                 # (S*N^2, 2)

            with torch.no_grad():
                ideal = prepare(sim, phi_rep, a.G, None, model)
                Cid = fhs(ideal.reshape(a.nseeds, N, N, -1))
                C_ideal = float(Cid[0][0])
                mn0 = float(Cid[1][0])
                Fideal = float(state_fidelity(ideal, prepare(
                    sim, phi_rep, a.G, None, model)).mean())

            print(f'\n--- fase {nombre}: C exacto = {C_ex}, '
                  f'C preparacion ideal = {C_ideal:+.4f}, min|U| = {mn0:.2e}')
            print(f'{"p":>7} {"F analit":>10} {"F muestr":>10} {"C medio":>9} '
                  f'{"+/-":>7} {"% correcto":>11} {"+/-":>7}')
            print('-' * 68)

            rows = []
            for p in a.ps:
                if model == 'indep':
                    # Pauli INDEPENDIENTE por (puerta, k): destruye el bundle
                    idx = sample_pauli_idx(B, a.G + 1, p, gen, dev)
                else:
                    # 'shared' y 'final': Pauli FIJO por puerta, igual en toda
                    # la malla k de esa semilla. Es el modelo de un dispositivo
                    # real: las mismas puertas fisicas, distinto angulo por k.
                    per_seed = sample_pauli_idx(a.nseeds, a.G + 1, p, gen, dev)
                    idx = per_seed.repeat_interleave(N * N, dim=0)
                with torch.no_grad():
                    noisy = prepare(sim, phi_rep, a.G, idx, model)
                    Fm = float(state_fidelity(ideal, noisy).mean())
                # fhs() ya devuelve UN C POR SEMILLA: calcularlo todo de una
                # vez. El bucle por semilla hacia nseeds sincronizaciones
                # GPU->CPU, que es lo que hacia lento el barrido.
                Cs = fhs(noisy.reshape(a.nseeds, N, N, -1))[0].cpu().numpy()
                # FHS devuelve ENTEROS {-1,0,+1}: el bundle sigue siendo coherente.
                # C_mean NO es un numero de Chern; es el promedio de aciertos.
                entero = bool(np.all(np.abs(Cs - np.round(Cs)) < 1e-9))
                Cideal = float(np.round(Cid[0][0].item()))
                n_ok = int(np.sum(np.round(Cs) == int(np.sign(Cideal)) ))
                good = float(np.mean(np.abs(Cs - C_ex) < 1e-6))
                se_C = float(np.nanstd(Cs) / math.sqrt(a.nseeds))
                se_g = math.sqrt(max(good * (1 - good), 1e-12) / a.nseeds)
                Fa = analytic_fidelity(p, n_noisy_gates(a.G, model))
                print(f'{p:>7.4f} {Fa:>10.5f} {Fm:>10.5f} {np.nanmean(Cs):>+9.3f} '
                      f'{se_C:>7.3f} {100*good:>10.1f}% {100*se_g:>6.1f}', flush=True)
                rows.append({'p': p, 'F_analitica': Fa, 'F_muestreada': Fm,
                             'C_mean': float(np.nanmean(Cs)),
                             'C_se': se_C, 'frac': good, 'frac_se': se_g})
            out['modelos'][model][nombre] = {
                'C_exact': int(C_ex), 'C_ideal_prep': C_ideal, 'rows': rows}
            _free(dev)

    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'\n-> {a.out}')


def _free(dev):
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--G', type=int, default=4)
    ap.add_argument('--Nk', type=int, default=12)
    ap.add_argument('--nseeds', type=int, default=200)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--modelos', nargs='+', default=['shared', 'final', 'indep'])
    ap.add_argument('--out', default='barrido_v2.json')
    ap.add_argument('--ps', type=float, nargs='+',
                    default=[0.0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30])
    run(ap.parse_args())


if __name__ == '__main__':
    main()