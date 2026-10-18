# dataset_real_space.py

import numpy as np
from haldane import critical_mass
try:
        from .hamiltonian_real_space import hamiltonian_real_space
        from .chern_real_space import chern_bott
    except ImportError:
        from hamiltonian_real_space import hamiltonian_real_space
        from chern_real_space import chern_bott


def dataset_real_space(n_samples, L=3, seed=0, t2_range=(0.05, 0.40),
                       phi_range=(np.pi/3, 2*np.pi/3),
                       t1_range=(0.4, 2.2), margin=0.12,
                       verbose=True):
    """Generate dataset of (M, t1, t2, phi) with exact Chern labels.

    The Chern is computed via the Bott index on the real-space
    Hamiltonian, which is exact for finite lattices.

    Returns:
        X:     (n_samples, 4) float64 -> [M/M_crit, log(t1), t2, |sin(phi)|]
        y:     (n_samples,) int64 -> 0 trivial, 1 topological
        chern: (n_samples,) int64 -> raw Chern (0 or -1)
        theta: (n_samples, 4) float64 -> (M, t1, t2, phi) raw
    """
    rng = np.random.default_rng(seed)
    n_occ = L * L  # half-filling: L^2 occupied states

    X, y, chern, theta = [], [], [], []
    tries = 0
    while len(X) < n_samples and tries < 400 * n_samples:
        tries += 1
        t2 = rng.uniform(*t2_range)
        phi = rng.uniform(*phi_range)
        t1 = rng.uniform(*t1_range)
        M_crit = critical_mass(0.0, 1.0, t2, phi)

        if rng.random() < 0.5:
            mag = rng.uniform(M_crit * (1.0 + margin),
                              M_crit * (1.0 + 3.0 * margin))
        else:
            mag = rng.uniform(0.0, M_crit * (1.0 - margin))
        M = mag if rng.random() < 0.5 else -mag

        H = hamiltonian_real_space(L, M, t1, t2, phi)
        C_int, C_real, _ = chern_bott(H, n_occ)
        chern.append(C_int)
        y.append(0 if C_int == 0 else 1)
        X.append([M / M_crit, np.log(t1), t2, abs(np.sin(phi))])
        theta.append([M, t1, t2, phi])

    X = np.asarray(X[:n_samples], dtype=np.float64)
    y = np.asarray(y[:n_samples], dtype=np.int64)
    chern = np.asarray(chern[:n_samples], dtype=np.int64)
    theta = np.asarray(theta[:n_samples], dtype=np.float64)

    if verbose:
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        print(f'dataset real-space: {len(y)} samples | trivial {n0} | topological {n1}')
        print(f'  features: [M/M_crit, log t1, t2, |sin phi|]')
        print(f'  L={L}, n_occ={n_occ}, dim={2*L*L}')

    return X, y, chern, theta