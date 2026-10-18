# hamiltonian_real_space.py

import numpy as np
from haldane import honeycomb


def hamiltonian_real_space(L, M, t1, t2, phi):
    """Build Haldane Hamiltonian in real space for an L x L periodic lattice.

    Returns a (2*L*L, 2*L*L) complex128 matrix.
    The basis is: for cell index c in row-major order (0..L*L-1),
    orbital 0 = sublattice A, orbital 1 = sublattice B.
    So the full index is c*2 + sublattice.
    """
    net = honeycomb()
    a1 = net['a1']
    a2 = net['a2']
    d_vecs = net['d']   # 3 nearest-neighbor vectors (A->B); d_vecs[0] = delta
    b_vecs = net['b']   # 3 next-nearest-neighbor vectors
    delta = d_vecs[0]

    N = L * L
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=np.complex128)

    # On-site: +M on A, -M on B
    for c in range(N):
        H[2*c, 2*c] += M
        H[2*c+1, 2*c+1] += -M

    # Nearest-neighbor hopping: -t1 between A and its 3 neighbors B
    for c in range(L * L):
        i = c // L
        j = c % L
        for dv in d_vecs:
            ni = i + int(round(dv[0]))
            nj = j + int(round(dv[1]))
            # periodic boundary conditions
            ni = ni % L
            nj = nj % L
            nc = ni * L + nj
            H[2*c, 2*nc+1] += -t1
            H[2*nc+1, 2*c] += -t1

    # Next-nearest-neighbor hopping: -i * t2 * exp(i*phi*s) for same sublattice
    # sign s = +1 for counterclockwise, -1 for clockwise
    for c in range(L * L):
        i = c // L
        j = c % L
        # A -> A hoppings (3 vectors: b[1]-b[0] etc, use b_vecs)
        for k, bv in enumerate(b_vecs):
            ni = i + int(round(bv[0]))
            nj = j + int(round(bv[1]))
            ni = ni % L
            nj = nj % L
            nc = ni * L + nj
            # sign: +1 if k in (0,1,2) pattern for CCW, -1 for CW
            # Standard Haldane: phase = +phi for CCW, -phi for CW
            # Using the convention from haldane.py: b = [d1-d2, d2-d0, d0-d1]
            # For A->A: phase = +phi for b[0], -phi for b[1], +phi for b[2]
            # (this is the standard convention giving Chern = -1 for M < Mc)
            if k == 0:
                phase = phi
            elif k == 1:
                phase = -phi
            else:
                phase = phi
            hop = t2 * np.exp(1j * phase)
            H[2*c, 2*nc] += hop
            H[2*nc, 2*c] += np.conj(hop)
            # B -> B hoppings (same vectors, opposite phase)
            hop_b = t2 * np.exp(-1j * phase)
            H[2*c+1, 2*nc+1] += hop_b
            H[2*nc+1, 2*c+1] += np.conj(hop_b)

    return H


def lattice_vectors(L):
    """Return (positions, a1, a2) for an L x L honeycomb lattice.

    positions: (2*L*L, 2) array of real-space site positions.
    """
    net = honeycomb()
    a1 = net['a1']
    a2 = net['a2']
    delta = net['d'][0]

    pos = []
    for i in range(L):
        for j in range(L):
            r = i * a1 + j * a2
            pos.append(r)           # sublattice A
            pos.append(r + delta)   # sublattice B
    return np.array(pos), a1, a2