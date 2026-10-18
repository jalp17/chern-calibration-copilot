"""Tests para Tarea 1: Chern en espacio real (red finita de Haldane).

Verifica:
  - Hermiticidad del Hamiltoniano en espacio real
  - Numero de Chern en fase topologica (M < Mc) y trivial (M > Mc)
  - Trace del proyector sobre estados ocupados = L^2
  - Batch de multiples parametros devuelve Chern entero
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parents[1]  # entrega/
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / 'simulador'))
sys.path.insert(0, str(_HERE / 'simulador' / 'espacio_real'))

import numpy as np
from haldane import critical_mass, chern_fhs, honeycomb
from hamiltonian_real_space import hamiltonian_real_space
from chern_real_space import chern_fhs_real_space, projector_occupied


L = 3
n_occ = L * L
t2, phi = 0.20, np.pi / 2
M_crit = critical_mass(0.0, 1.0, t2, phi)


def test_hamiltonian_hermitiano():
    """T-1-1: H = H^dag para L=2,3,4"""
    for L_test in [2, 3, 4]:
        H = hamiltonian_real_space(L_test, 0.5, 1.0, t2, phi)
        err = np.max(np.abs(H - H.conj().T))
        assert err < 1e-12, f"H no hermito para L={L_test}: error={err}"


def test_chern_espacio_real_topologico():
    """T-1-2: Chern = -1 en fase topologica (M < Mc)"""
    for M in [0.0, 0.5, 1.0]:
        H = hamiltonian_real_space(L, M, 1.0, t2, phi)
        C_int, C_real = chern_fhs_real_space(H, n_occ, L, M)
        assert C_int == -1, f"Esperaba C=-1 para M={M}, obtuvo C={C_int}"


def test_chern_espacio_real_trivial():
    """T-1-3: Chern = 0 en fase trivial (M > Mc)"""
    for M in [1.5, 2.0, 3.0]:
        H = hamiltonian_real_space(L, M, 1.0, t2, phi)
        C_int, C_real = chern_fhs_real_space(H, n_occ, L, M)
        assert C_int == 0, f"Esperaba C=0 para M={M}, obtuvo C={C_int}"


def test_proyector_trace_equals_N_occ():
    """T-1-4: trace del proyector sobre ocupadas = L^2"""
    H = hamiltonian_real_space(L, 0.5, 1.0, t2, phi)
    P, _ = projector_occupied(H, n_occ)
    tr = np.trace(P)
    assert abs(tr.real - n_occ) < 1e-10, f"trace(P)={tr.real}, esperado={n_occ}"


def test_batch_varios_parametros():
    """T-1-5: multiples (M, t2, phi) devuelven Chern entero"""
    parametros = [(0.0, 0.20, np.pi/2), (0.5, 0.30, np.pi/3),
                  (1.5, 0.20, np.pi/2), (2.0, 0.40, np.pi/4)]
    for M, t2_val, phi_val in parametros:
        H = hamiltonian_real_space(L, M, 1.0, t2_val, phi_val)
        C_int, C_real = chern_fhs_real_space(H, n_occ, L, M)
        assert abs(C_real - C_int) < 0.1, f"Chern no cercano a entero: {C_real}"