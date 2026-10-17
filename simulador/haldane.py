"""
Modelo de Haldane + numero de Chern por la formula FHS.

Generador de ground truth para la tarea de clasificacion del hackathon,
y base fisica reutilizable para la Demo D de la exposicion.

La red, el Hamiltoniano y FHS son los mismos que usa
`exposicion-hall-cuantica/05-DEMOS/demo_d_numero_chern.py`, ya validado
contra el umbral analitico 3*sqrt(3)*t2*|sin(phi)|. Este modulo es la
version sin prints, importable y vectorizada en lote.

    from haldane import sample_dataset, chern_exact
    X, y, chern = sample_dataset(n=4000, seed=0)

Hamiltoniano de Haldane en la base de subredes A|B:

    H(k) = [[ dz,   dx - i*dy ],
            [ dx+i*dy,  -dz  ]]

    dx = t1 * sum_i cos(k . d_i)
    dy = t1 * sum_i sin(k . d_i)
    dz = M  - 2*t2*sin(phi) * sum_i sin(k . b_i)

con d_i los tres vecinos primeros y b_i los tres segundos.

Numero de Chern por FHS (Fukui-Hatsugai-Suzuki, PRL 93, 256404 (2004)):
enlaces U_mu(k) = <u(k)|u(k+hat mu)> / |.|, y

    C = (1/2pi) * sum_k arg[ U1(k) U2(k+1) U1(k+2)^* U2(k)^* ]

que da un ENTERO en un hueco abierto.

Referencias:
  F. D. M. Haldane, Phys. Rev. Lett. 61, 2015 (1988)
  Fukui, Hatsugai, Suzuki, Phys. Rev. Lett. 93, 256404 (2004)
"""

from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)


# --------------------------------------------------------------------------
# Red
# --------------------------------------------------------------------------

def honeycomb() -> dict:
    """Vectores de la red hexagonal (a escala de salto mas proximo = 1)."""
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, SQRT3 / 2.0])
    delta = np.array([0.5, SQRT3 / 6.0])                    # desplazamiento A -> B
    d = np.array([delta, delta - a1, delta - a2])           # 3 vecinos primeros
    b = np.array([d[1] - d[2], d[2] - d[0], d[0] - d[1]])   # 3 vecinos segundos
    G1 = np.array([2.0 * np.pi, 2.0 * np.pi / SQRT3])
    G2 = np.array([0.0, 4.0 * np.pi / SQRT3])
    return {"a1": a1, "a2": a2, "d": d, "b": b, "G1": G1, "G2": G2}


def _components(net: dict, k: np.ndarray, M: float, t1: float,
                t2: float, phi: float):
    """(dx, dy, dz) para k de forma (..., 2)."""
    kd = k @ net["d"].T                      # (..., 3)
    kb = k @ net["b"].T
    dx = t1 * np.sum(np.cos(kd), axis=-1)
    dy = t1 * np.sum(np.sin(kd), axis=-1)
    dz = M - 2.0 * t2 * np.sin(phi) * np.sum(np.sin(kb), axis=-1)
    return dx, dy, dz


def hamiltonian(net: dict, ks: np.ndarray, M: float, t1: float,
                t2: float, phi: float) -> np.ndarray:
    """Pila de H(k) 2x2 para puntos k de forma (..., 2)."""
    dx, dy, dz = _components(net, ks, M, t1, t2, phi)
    H = np.empty(np.broadcast(dx, dy, dz).shape + (2, 2), dtype=np.complex128)
    H[..., 0, 0] = dz
    H[..., 0, 1] = dx - 1j * dy
    H[..., 1, 0] = dx + 1j * dy
    H[..., 1, 1] = -dz
    return H


def occupied_band(H: np.ndarray) -> np.ndarray:
    """Autovector de menor energia de cada H(k). H de forma (..., 2, 2)."""
    w, v = np.linalg.eigh(H)
    return v[..., :, 0]


def dirac_point(net: dict, t2: float = 0.20,
                phi: float = np.pi / 2, n: int = 400) -> np.ndarray:
    """Punto de Dirac K: donde dx = dy = 0.

    Se busca numericamente en una malla gruesa sobre el paralelogramo de la
    zona de Brillouin. Hardcodear K es frágil: depende de la convencion
    exacta de a1, a2. (La Demo D de la exposicion hace lo mismo.)
    """
    G1, G2 = net["G1"], net["G2"]
    u = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    ks = (uu[..., None] * G1 + vv[..., None] * G2).reshape(-1, 2)
    dx, dy, _ = _components(net, ks, 0.0, 1.0, t2, phi)
    return ks[int(np.argmin(dx ** 2 + dy ** 2))]


def gap_at_dirac(net: dict, M: float, t1: float, t2: float,
                 phi: float) -> float:
    """|d_z(K)| en el punto de Dirac, que da el umbral de cierre de hueco."""
    K = dirac_point(net, t2, phi)
    _, _, dz = _components(net, K[None, :], M, t1, t2, phi)
    return float(abs(dz[0]))


def critical_mass(M: float, t1: float, t2: float, phi: float) -> float:
    """M critico: |M| = 3*sqrt(3)*t2*|sin(phi)|."""
    return 3.0 * SQRT3 * t2 * abs(np.sin(phi))


# --------------------------------------------------------------------------
# Chern por FHS
# --------------------------------------------------------------------------

def _bz_mesh(net: dict, N: int) -> np.ndarray:
    """Malla NxN sobre el paralelogramo de la zona de Brillouin. (N, N, 2)"""
    G1, G2 = net["G1"], net["G2"]
    u = (np.arange(N) + 0.5) / N
    v = (np.arange(N) + 0.5) / N
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return (uu[..., None] * G1 + vv[..., None] * G2).reshape(N * N, 2)


def chern_fhs(net: dict, N: int = 48, M: float = 0.0, t1: float = 1.0,
              t2: float = 0.20, phi: float = np.pi / 2) -> float:
    """Numero de Chern por FHS. Devuelve el entero mas cercano."""
    ks = _bz_mesh(net, N).reshape(N, N, 2)
    H = hamiltonian(net, ks, M, t1, t2, phi)
    u = occupied_band(H)                       # (N, N, 2)

    # enlaces
    U1 = np.einsum("...a,...a->...", u.conj(), np.roll(u, -1, axis=0))
    U2 = np.einsum("...a,...a->...", u.conj(), np.roll(u, -1, axis=1))
    U1 /= np.abs(U1)
    U2 /= np.abs(U2)

    # producto alrededor de cada plaqueta
    plaq = (U1
            * np.roll(U2, -1, axis=0)
            * np.conj(np.roll(U1, -1, axis=1))
            * np.conj(U2))
    C = np.angle(plaq).sum() / (2.0 * np.pi)
    return int(np.round(C))          # -0.0 -> 0


def chern_exact(M: float = 0.0, t1: float = 1.0, t2: float = 0.20,
                phi: float = np.pi / 2, N: int = 48) -> int:
    """C con los valores por defecto. Azucar para llamadas comodo."""
    return int(chern_fhs(honeycomb(), N, M, t1, t2, phi))


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def sample_dataset(n: int = 4000, seed: int = 0, t2_range: tuple = (0.05, 0.40),
                   phi_range: tuple = (np.pi / 3, 2 * np.pi / 3),
                   t1_range: tuple = (0.4, 2.2), N: int = 48,
                   margin: float = 0.12, net: dict | None = None,
                   verbose: bool = True):
    """Muestrea (M, t1, t2, phi) y calcula su Chern exacto con FHS.

    Para parametros fijos, C = 0 si |M| > 3*sqrt(3)*t2*|sin(phi)|. Si solo se
    dieran (M, t1) la frontera seria |X0| = 1: un umbral trivial y la tarea
    no diria nada. Por eso se VARÍAN t2 y phi: la frontera pasa a ser la
    superficie |M| = 3*sqrt(3)*t2*|sin(phi)| en 4D, que no la resuelve un
    clasificador lineal de una variable.

    Se EXCLUYE una banda de anchura 2*margin*M_crit alrededor del umbral:
    ahi el hueco casi se cierra y el Chern es numericamente fragil, asi
    que sin ese margen el dataset traeria puntos mal etiquetados.

    Devuelve:
        X:     (n, 4) float64 -> [M/M_crit, log(t1), t2, sin(phi)]
        y:     (n,)   int64   -> 0 trivial, 1 topologico
        chern: (n,)   int64   -> el Chern crudo (0 o -1)
        theta: (n, 4) float64 -> (M, t1, t2, phi) crudos
    """
    net = net or honeycomb()
    rng = np.random.default_rng(seed)

    X, y, chern, theta = [], [], [], []
    tries = 0
    while len(X) < n and tries < 400 * n:
        tries += 1
        t2 = rng.uniform(*t2_range)
        phi = rng.uniform(*phi_range)
        t1 = rng.uniform(*t1_range)
        M_crit = critical_mass(0.0, 1.0, t2, phi)

        # magnitud M fuera de la banda excluida, en uno de los dos sides
        if rng.random() < 0.5:
            mag = rng.uniform(M_crit * (1.0 + margin),
                              M_crit * (1.0 + 3.0 * margin))
        else:
            mag = rng.uniform(0.0, M_crit * (1.0 - margin))
        M = mag if rng.random() < 0.5 else -mag

        C = chern_fhs(net, N, M, t1, t2, phi)
        chern.append(C)
        y.append(0 if C == 0 else 1)
        X.append([M / M_crit, np.log(t1), t2, abs(np.sin(phi))])
        theta.append([M, t1, t2, phi])

    X = np.asarray(X[:n], dtype=np.float64)
    y = np.asarray(y[:n], dtype=np.int64)
    chern = np.asarray(chern[:n], dtype=np.int64)
    theta = np.asarray(theta[:n], dtype=np.float64)

    if verbose:
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        print(f'dataset: {len(y)} muestras | triviales {n0} | topologicas {n1}')
        print(f'  features: [M/M_crit, log t1, t2, |sin phi|]')
        print(f'  t2   en {t2_range},  phi en ({phi_range[0]:.3f}, {phi_range[1]:.3f})')

    return X, y, chern, theta


if __name__ == '__main__':
    net = honeycomb()

    print('=' * 68)
    print('VALIDACION — debe reproducir la Demo D de la exposicion')
    print('=' * 68)
    t2, phi = 0.20, np.pi / 2
    M_crit = critical_mass(0.0, 1.0, t2, phi)
    M_num = gap_at_dirac(net, 0.0, 1.0, t2, phi)
    print(f'  M_crit analitico 3*sqrt(3)*t2*sin(phi) = {M_crit:.6f}')
    print(f'  |d_z(K)| numerico                       = {M_num:.6f}')
    print(f'  error relativo = {abs(M_num / M_crit - 1):.2e}')

    for M in (2.0, 1.0, 0.0, -1.0, -2.0):
        C = chern_fhs(net, 48, M, 1.0, t2, phi)
        print(f'  M = {M:+.1f} -> C = {C:>2} '
              f'({"trivial" if C == 0 else "topologica"})')

    print()
    print('=' * 68)
    print('TRANSICION — debe ser ABRUPTA en |M| = M_crit')
    print('=' * 68)
    for r in (0.999, 1.0, 1.001):
        print(f'  |M|/M_crit = {r:<6} -> C = {chern_fhs(net, 64, r*M_crit, 1.0, t2, phi)}')

    print()
    print('=' * 68)
    print('CONVERGENCIA DE LA MALLA')
    print('=' * 68)
    for N in (16, 32, 48, 64):
        print(f'  N = {N:>3} -> C(M=1,0,-1) = '
              f'{[chern_fhs(net, N, M, 1.0, t2, phi) for M in (1.0, 0.0, -1.0)]}')

    print()
    print('=' * 68)
    print('DATASET')
    print('=' * 68)
    X, y, chern, theta = sample_dataset(n=800, seed=0)
    for i, name in enumerate(['M/M_crit', 'log t1', 't2', '|sin phi|']):
        print(f'  {name:>10}: [{X[:, i].min():+.3f}, {X[:, i].max():+.3f}]')