# chern_real_space.py

import numpy as np


def projector_occupied(H, n_occ):
    """Projector onto the n_occ lowest-energy eigenvectors of H.

    P = sum_{i=0}^{n_occ-1} |u_i><u_i|
    """
    evals, evecs = np.linalg.eigh(H)
    U = evecs[:, :n_occ]
    P = U @ U.conj().T
    return P, evals


def chern_fhs_real_space(H, n_occ, L, M, net=None, N=24):
    """Numero de Chern del subespacio ocupado por FHS.

    Para un sistema periodico de tamano LxL, aplica la formula FHS
    sobre la malla de Brillouin discretizada (tamano N para convergencia).
    Equivalente a haldane.chern_fhs pero construyendo H(k) desde el
    Hamiltoniano en espacio real.

    Returns (C_int, C_raw).
    """
    if net is None:
        from haldane import honeycomb
        net = honeycomb()

    from haldane import occupied_band

    # Malla de zona de Brillouin (tamano N para convergencia, independiente de L)
    G1, G2 = net['G1'], net['G2']
    u = (np.arange(N) + 0.5) / N
    v = (np.arange(N) + 0.5) / N
    uu, vv = np.meshgrid(u, v, indexing='ij')
    ks = (uu[..., None] * G1 + vv[..., None] * G2).reshape(N * N, 2)

    # Hamiltoniano en k para cada punto de la malla
    from haldane import _components
    dx, dy, dz = _components(net, ks, M, 1.0, 0.20, np.pi / 2)
    Hk = np.empty((N, N, 2, 2), dtype=np.complex128)
    Hk[..., 0, 0] = dz.reshape(N, N)
    Hk[..., 0, 1] = (dx - 1j * dy).reshape(N, N)
    Hk[..., 1, 0] = (dx + 1j * dy).reshape(N, N)
    Hk[..., 1, 1] = (-dz).reshape(N, N)

    u_vecs = occupied_band(Hk)  # (N, N, 2)

    # Enlaces U(1)
    U1 = np.einsum('...a,...a->...', u_vecs.conj(), np.roll(u_vecs, -1, axis=0))
    U2 = np.einsum('...a,...a->...', u_vecs.conj(), np.roll(u_vecs, -1, axis=1))
    U1 /= np.abs(U1)
    U2 /= np.abs(U2)

    plaq = (U1
            * np.roll(U2, -1, axis=0)
            * np.conj(np.roll(U1, -1, axis=1))
            * np.conj(U2))
    C = np.angle(plaq).sum() / (2.0 * np.pi)

    return int(np.round(C)), C


def chern_bott(H, n_occ):
    """Chern number via Bott index (alternative method).

    For small lattices the Bott index may not converge to an exact integer;
    use chern_fhs_real_space for reliable results.
    """
    dim = H.shape[0]
    L = int(round(np.sqrt(dim / 2.0)))

    from haldane import honeycomb
    net = honeycomb()
    G1 = net['G1']
    G2 = net['G2']

    pos = []
    for i in range(L):
        for j in range(L):
            r = i * net['a1'] + j * net['a2']
            pos.append(r)
            pos.append(r + net['d'][0])
    pos = np.array(pos)

    nx = (pos @ G1 / (2.0 * np.pi)) % L
    ny = (pos @ G2 / (2.0 * np.pi)) % L

    P, evals = projector_occupied(H, n_occ)

    Ex = np.diag(np.exp(1j * 2.0 * np.pi * nx / L))
    Ey = np.diag(np.exp(1j * 2.0 * np.pi * ny / L))

    Ux = P @ Ex @ P
    Uy = P @ Ey @ P

    M = Uy.conj().T @ Ux @ Uy @ Ux.conj().T

    evals_M, evecs_M = np.linalg.eig(M)
    log_M = evecs_M @ np.diag(np.log(evals_M)) @ np.linalg.inv(evecs_M)

    trace_log = np.trace(log_M)
    C = trace_log / (2.0j * np.pi)

    return int(np.round(C.real)), C.real, evals


def chern_bott_batch(H_list, n_occ):
    """Chern numbers for a batch of Hamiltonians."""
    results = []
    for H in H_list:
        C_int, C_real, evals = chern_bott(H, n_occ)
        results.append((C_int, C_real))
    return results