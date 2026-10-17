"""
BARRIDO PRINCIPAL — fragilidad del numero de Chern ante error de puerta.

DISEÑO:
El circuito PREPARA el estado ocupado del modelo de Haldane, |phi(k)>.
El Chern se calcula con FHS sobre esa familia de estados. Si la
preparación es imperfecta, el Chern deja de ser un entero.

La profundidad G de la preparación (cuántas puertas se usan para
implementar la rotación que lleva |0> a |phi(k)>) y la tasa de error por
puerta p son los dos ejes. Hay un compromiso real:

    muchas puertas -> mejor preparación ideal
    muchas puertas -> mas ruido acumulado

VARIABLES
  n        qubits del registro (holgura de memoria; solo el 0 se usa)
  G        profundidad de la preparación, en puertas de 1 qubit
  p        probabilidad depolarizante POR PUERTA
  Nk       malla de la zona de Brillouin

Producto cartesiano = Nk^2 x nseeds x n_p, que es el lote.

Uso:
  python3 barrido_chern.py --n 6 --G 2 --Nk 16 --nseeds 32
  python3 barrido_chern.py --device cuda --n 12 --G 4 --Nk 24 --nseeds 64
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from haldane import (SQRT3, critical_mass, gap_at_dirac, honeycomb,
                      hamiltonian, occupied_band)
from qsim_torch import Simulator, ry, rotation_z, state_fidelity
from ruido import GateNoise

# RZ y RZ-hardware-efficient; se usan porque son las rotaciones de un
# Solo se usa RY + RZ: cualquier estado de 2 niveles se prepara con una
# secuencia de rotaciones.


def rotation_z(theta: torch.Tensor, dtype=torch.complex128) -> torch.Tensor:
    """RZ(theta) = diag(e^{-i t/2}, e^{+i t/2})."""
    h = theta / 2
    z = torch.zeros_like(h)
    return torch.stack([
        torch.stack([torch.exp(-1j * h), z], dim=-1),
        torch.stack([z, torch.exp(1j * h)], dim=-1),
    ], dim=-2).to(dtype)


def prepare(sim, phi_k, G, noise=None):
    """Prepara |phi_k> en el qubit 0 usando G puertas RY.

    |phi> = phi_0|0> + phi_1|1>  se escribe como RY(theta)|0> con

        theta = 2 * atan2(|phi_1|, |phi_0|)

    y se aplica RY(theta/G) G veces, que en el limite ideal es EXACTO pero
    reparte la preparacion en G pasos: mas profundidad, mas puertas
    donde acumular error.

    Devuelve (batch, 2^n).
    """
    theta = 2.0 * torch.atan2(phi_k[:, 1].abs(), phi_k[:, 0].abs()).real
    # fase RELATIVA: arg(phi_1) - arg(phi_0), no arg(phi_1).
    # Con arg(phi_1) solo el Chern da 0 en cuanto phi_0 tiene fase.
    gamma = torch.angle(phi_k[:, 1]) - torch.angle(phi_k[:, 0])
    psi = sim.zeros(phi_k.shape[0])
    for _ in range(G):
        # RY(theta/G) x G  reproduce RY(theta) exactamente
        psi = sim._apply_1q(psi, ry(theta / G, sim.dtype), 0)
        if noise is not None:
            psi = noise.apply_1q(psi, sim, 0)
    # RZ(gamma) aporta la FASE RELATIVA: sin ella el Chern sale 0.
    # (Sin esta puerta la fidelidad con el estado exacto se queda en ~0.2.)
    psi = sim._apply_1q(psi, rotation_z(gamma, sim.dtype), 0)
    if noise is not None:
        psi = noise.apply_1q(psi, sim, 0)
    return psi


def fhs(states):
    """states: (N,N,d) normalizados -> (C, min|U|)."""
    u = states
    U1 = np.einsum('...a,...a->...', u.conj(), np.roll(u, -1, axis=0))
    U2 = np.einsum('...a,...a->...', u.conj(), np.roll(u, -1, axis=1))
    n1, n2 = np.abs(U1), np.abs(U2)
    mn = float(min(n1.min(), n2.min()))
    if mn < 1e-12:
        return float('nan'), mn
    plaq = (U1 / n1) * (np.roll(U2, -1, axis=0) / n2) * \
           (np.conj(np.roll(U1, -1, axis=1)) / n1) * (np.conj(U2) / n2)
    return float(np.angle(plaq).sum() / (2 * np.pi)), mn


def bz(net, N):
    u = (np.arange(N) + 0.5) / N
    uu, vv = np.meshgrid(u, u, indexing='ij')
    return (uu[..., None] * net['G1'] + vv[..., None] * net['G2'])


def run(a):
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    net = honeycomb()
    sim = Simulator(a.n, str(dev))
    N = a.Nk
    MC = critical_mass(0.0, 1.0, 0.20, math.pi / 2)
    Ms = [('trivial', 1.6 * MC), ('topologica', 0.5 * MC)]

    print(f'dispositivo {dev}' + (f' | {torch.cuda.get_device_properties(0).name}'
          if dev.type == 'cuda' else ''))
    print(f'n={a.n}  profundidad G={a.G}  malla {N}x{N}  semillas={a.nseeds}  '
          f'puertas con ruido={a.G + 1}')
    print(f'M_crit = {MC:.6f}   |d_z(K)| = {gap_at_dirac(net,0,1,0.20,math.pi/2):.6f}')

    out = {'n': a.n, 'G': a.G, 'Nk': N, 'nseeds': a.nseeds,
           'M_crit': MC, 'fases': {}}

    for nombre, M in Ms:
        ks = bz(net, N)
        H = hamiltonian(net, ks, M, 1.0, 0.20, math.pi / 2)
        u_exact = occupied_band(H)                    # (N,N,2)
        C_exact = round(fhs(u_exact)[0])
        print(f'\n=== fase {nombre}: M={M:.4f}  C exacto = {C_exact} ===')

        phi = torch.tensor(u_exact.reshape(N * N, 2),
                           dtype=torch.complex128, device=dev)
        with torch.no_grad():
            ideal = prepare(sim, phi, a.G)
        C_ideal, m0 = fhs(ideal.reshape(N, N, -1).cpu().numpy())
        # fidelidad de la preparacion ideal frente al estado de 2 niveles
        # objetivo exacto embebido (big-endian: qubit 0 es el MSB)
        tgt = torch.zeros(N * N, 1 << a.n, dtype=torch.complex128, device=dev)
        tgt[:, 0] = phi[:, 0]
        tgt[:, 1 << (a.n - 1)] = phi[:, 1]
        # FIDELIDAD REAL |psi|phi>|^2 — no sum|psi|^2|phi|^2 (colisión)
        F_prep = float(state_fidelity(ideal, tgt).mean())
        print(f'  C de la preparacion ideal (G={a.G}) = {C_ideal:+.6f}   '
              f'F vs estado exacto = {F_prep:.6f}')

        print(f'\n  {"p/puerta":>10} {"1-F":>9} {"C medio":>10} {"C std":>8} '
              f'{"% correcto":>11}')
        print('  ' + '-' * 54)
        rows = []
        for p in a.ps:
            Fs, Cs = [], []
            with torch.no_grad():
                for s in range(a.nseeds):
                    nz = GateNoise(p, p * a.ratio21, seed=a.seed + 7919 * s)
                    out_psi = prepare(sim, phi, a.G, nz)
                    # infidelity respecto al estado EXACTO de la banda:
                    # es la que decide si el Chern significa algo
                    Fs.append(float(state_fidelity(tgt, out_psi).mean()))
                    Cn, _ = fhs(out_psi.reshape(N, N, -1).cpu().numpy())
                    Cs.append(Cn if Cn == Cn else float('nan'))
            Fs, Cs = np.array(Fs), np.array(Cs)
            good = float(np.mean(np.abs(Cs - C_exact) < 1e-6))
            print(f'  {p:>10.4f} {1-Fs.mean():>9.4f} {np.nanmean(Cs):>+10.3f} '
                  f'{np.nanstd(Cs):>8.3f} {100*good:>10.0f}%', flush=True)
            rows.append({'p': p, 'infidelity': float(1 - Fs.mean()),
                         'C_mean': float(np.nanmean(Cs)),
                         'C_std': float(np.nanstd(Cs)),
                         'frac_correcto': good})
        out['fases'][nombre] = {'M': M, 'C_exact': int(C_exact),
                                'C_ideal_prep': C_ideal, 'rows': rows}
        _free(dev)

    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'\n-> {a.out}')


def _free(dev):
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--G', type=int, default=1)
    ap.add_argument('--Nk', type=int, default=12)
    ap.add_argument('--nseeds', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ratio21', type=float, default=3.0)
    ap.add_argument('--out', default='barrido.json')
    ap.add_argument('--ps', type=float, nargs='+',
                    default=[0.0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40])
    run(ap.parse_args())


if __name__ == '__main__':
    main()