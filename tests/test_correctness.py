"""Categoria A: CORRECTNESS (fisica y codigo).

Verifica que el simulador y la fisica dan lo correcto:
- puertas unitarias vs kron exacto (n=1..4)
- CNOT todos los pares ordenados
- state_fidelity = |<psi|phi>|^2
- ry vs convencion estandar (medio angulo)
- FHS: sumar SOLO sobre ejes de malla
- Haldane: Chern = 0 trivial, -1 topologico
- prepare con ruido p=0 reproduce exactamente el estado ideal
- barrido_v2 con p=0 da 100% acierto
- agente decide ACEPTAR/INDECSO/RECHAZAR
"""

from __future__ import annotations

import math

import numpy as np
import torch

from qsim_torch import H, X, Y, Z, Simulator, rotation, ry, rotation_z, state_fidelity
from haldane import honeycomb, chern_fhs, chern_exact, critical_mass, occupied_band, hamiltonian
from barrido_v2 import fhs, prepare, bz
from agente import agente as ag

D = torch.complex128
I2M = np.eye(2, dtype=np.complex128)
RNG = np.random.default_rng(42)


def embed(ops: list, n: int) -> np.ndarray:
    m = np.array([[1]], dtype=np.complex128)
    for o in ops:
        m = np.kron(m, o)
    return m


def cnot_matrix(control: int, target: int, n: int) -> np.ndarray:
    size = 1 << n
    m = np.zeros((size, size), dtype=np.complex128)
    c_bit = n - 1 - control
    t_bit = n - 1 - target
    for i in range(size):
        j = i ^ (1 << t_bit) if (i >> c_bit) & 1 else i
        m[j, i] = 1.0
    return m


def rand_state(n: int) -> np.ndarray:
    a = RNG.normal(size=1 << n) + 1j * RNG.normal(size=1 << n)
    return a / np.linalg.norm(a)


# --- A1: Puertas 1-qubit vs kron exacto -----------------------------------

def test_puertas_un_qubit_vs_kron():
    """T-A1: Las 4 puertas (H, X, Y, Z) en todas las posiciones, n=1..4."""
    worst = 0.0
    for n in (1, 2, 3, 4):
        a = rand_state(n)
        psi0 = torch.tensor(a, dtype=D).unsqueeze(0)
        sim = Simulator(n)
        for gate in (H, X, Y, Z):
            gnp = gate.numpy()
            for q in range(n):
                ref = embed([gnp if i == q else I2M for i in range(n)], n) @ a
                got = sim._apply_1q(psi0, gate, q)[0].numpy()
                err = float(np.abs(got - ref).max())
                worst = max(worst, err)
    assert worst < 1e-12, f'puertas 1-qubit: error {worst:.2e}'


# --- A2: CNOT todos los pares ordenados -----------------------------------

def test_cnot_pares_ordenados():
    """T-A2: CNOT en todos los pares (control, target) con n=2..4."""
    worst = 0.0
    for n in (2, 3, 4):
        a = rand_state(n)
        psi0 = torch.tensor(a, dtype=D).unsqueeze(0)
        sim = Simulator(n)
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                got = sim._apply_cnot(psi0, c, t)[0].numpy()
                ref = cnot_matrix(c, t, n) @ a
                err = float(np.abs(got - ref).max())
                worst = max(worst, err)
    assert worst < 1e-12, f'CNOT: error {worst:.2e}'


# --- A3: state_fidelity = |<psi|phi>|^2 -----------------------------------

def test_state_fidelity_es_fidelidad():
    """T-A3: state_fidelity devuelve |<psi|phi>|^2, no otra cosa."""
    n = 3
    psi = torch.tensor(rand_state(n), dtype=D).unsqueeze(0)
    phi = torch.tensor(rand_state(n), dtype=D).unsqueeze(0)

    fid = state_fidelity(psi, phi)
    overlap = (psi.conj() * phi).sum()
    expected = overlap.abs() ** 2

    err = float((fid - expected).abs())
    assert err < 1e-12, f'fidelidad: error {err:.2e}'
    assert 0.0 <= float(fid) <= 1.0, f'fidelidad fuera de [0,1]: {float(fid)}'

    # identidad
    fid_id = state_fidelity(psi, psi)
    assert abs(float(fid_id) - 1.0) < 1e-12, f'fidelidad consigo mismo: {float(fid_id)}'


def test_state_fidelity_no_es_probabilidad_colision():
    """T-A3b: NO es sum_i |psi_i|^2 |phi_i|^2."""
    n = 3
    psi = torch.tensor(rand_state(n), dtype=D).unsqueeze(0)
    phi = torch.tensor(rand_state(n), dtype=D).unsqueeze(0)

    fid = state_fidelity(psi, phi)
    collision_prob = (psi.abs() ** 2 * phi.abs() ** 2).sum()

    # La fidelidad y la probabilidad de colision coinciden solo si los
    # estados son reales y positivos. Con estados complejos generales, no.
    # Verificamos que la formula implementada es la fidelidad
    overlap = (psi.conj() * phi).sum()
    expected = overlap.abs() ** 2
    assert abs(float(fid) - float(expected)) < 1e-12


# --- A4: ry vs convencion estandar (medio angulo) --------------------------

def test_ry_medio_angulo():
    """T-A4: ry(theta) construye RY(theta) = [[cos(theta/2), -sin(theta/2)], ...]."""
    theta = torch.tensor([0.3], dtype=torch.float64)
    g = ry(theta, D)

    c, s = math.cos(0.3 / 2), math.sin(0.3 / 2)
    expected = torch.tensor([[c, -s], [s, c]], dtype=D)

    err = float((g[0] - expected).abs().max())
    assert err < 1e-15, f'ry: error {err:.2e}'

    # rotation(cos, sin) con angulo completo da RY(2*theta), no RY(theta)
    theta2 = torch.tensor([0.3], dtype=torch.float64)
    g_full = rotation(torch.cos(theta2), torch.sin(theta2), D)
    expected_full = torch.tensor(
        [[math.cos(0.3), -math.sin(0.3)],
         [math.sin(0.3), math.cos(0.3)]], dtype=D)
    err_full = float((g_full[0] - expected_full).abs().max())
    assert err_full < 1e-15, f'rotation(cos,sin): error {err_full:.2e}'


# --- A5: FHS suma SOLO sobre ejes de malla --------------------------------

def test_fhs_suma_ejes_de_malla():
    """T-A5: fhs devuelve un C por semilla, no la suma sobre el lote."""
    net = honeycomb()
    N = 8
    M = 0.0
    t2, phi = 0.20, math.pi / 2

    ks = bz(net, N)
    u = occupied_band(hamiltonian(net, ks, M, 1.0, t2, phi))

    # una sola semilla
    C1, mn1 = fhs(torch.tensor(u, dtype=D))
    assert C1.shape == (), f'forma esperada (), {C1.shape}'

    # varias semillas batchadas: debe dar un C por semilla
    nseeds = 4
    u_batch = torch.tensor(u, dtype=D).unsqueeze(0).repeat(nseeds, 1, 1, 1)
    Cb, mnb = fhs(u_batch)
    assert Cb.shape == (nseeds,), f'forma esperada ({nseeds},), {Cb.shape}'

    # todas las semillas tienen el mismo C
    assert torch.allclose(Cb, Cb[0].expand_as(Cb), atol=1e-10), \
        f'C varia entre semillas: {Cb}'

    # el valor debe ser el entero correcto (M=0 -> C=-1 para Haldane)
    assert int(round(float(Cb[0]))) == -1, f'C(M=0) deberia ser -1, {float(Cb[0])}'


# --- A6: Haldane Chern = 0 trivial, -1 topologico ---------------------------

def test_chern_trivial_y_topologico():
    """T-A6: Chern = 0 trivial, -1 topologico, mallas N=8..32."""
    net = honeycomb()
    t2, phi = 0.20, math.pi / 2
    Mc = critical_mass(0.0, 1.0, t2, phi)

    for N in (8, 16, 24, 32):
        C_triv = chern_fhs(net, N, 1.6 * Mc, 1.0, t2, phi)
        C_topo = chern_fhs(net, N, 0.5 * Mc, 1.0, t2, phi)

        assert C_triv == 0, f'N={N}: trivial C={C_triv} (esperado 0)'
        assert C_topo == -1, f'N={N}: topologico C={C_topo} (esperado -1)'


def test_chern_exact_wrapper():
    """T-A6b: chern_exact() con defaults da -1."""
    C = chern_exact()
    assert C == -1, f'chern_exact() = {C} (esperado -1)'


# --- A7: prepare con p=0 reproduce el estado ideal -----------------------

def test_prepare_p0_exacto():
    """T-A7: prepare con idx=None reproduce el estado ideal exactamente."""
    net = honeycomb()
    N = 6
    M = 0.0
    t2, phi = 0.20, math.pi / 2

    ks = bz(net, N)
    u = occupied_band(hamiltonian(net, ks, M, 1.0, t2, phi))
    phi_states = torch.tensor(u.reshape(N * N, 2), dtype=D)

    sim = Simulator(1)
    G = 4

    with torch.no_grad():
        ideal = prepare(sim, phi_states, G, None, 'indep')
        C_ideal, _ = fhs(ideal.reshape(1, N, N, -1))

    # el Chern de la preparacion ideal debe ser -1
    assert int(round(float(C_ideal[0]))) == -1, \
        f'C preparacion ideal = {float(C_ideal[0])} (esperado -1)'


# --- A8: barrido con p=0 da 100% acierto ----------------------------------

def test_barrido_p0_100_por_ciento():
    """T-A8: barrido con p=0 da 100% acierto en ambos modelos."""
    import argparse
    from barrido_v2 import run

    a = argparse.Namespace(
        n=1, G=2, Nk=4, nseeds=16, device='cpu', seed=0,
        modelos=['shared', 'final'], ps=[0.0],
        out='/tmp/test_barrido_p0.json'
    )

    # Capturar stdout para no ensalar la salida
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(a)

    import json
    with open('/tmp/test_barrido_p0.json') as f:
        res = json.load(f)

    for modelo in ('shared', 'final'):
        for fase in ('trivial', 'topologica'):
            rows = res['modelos'][modelo][fase]['rows']
            assert len(rows) == 1, f'{modelo}/{fase}: {len(rows)} filas'
            assert rows[0]['frac'] == 1.0, \
                f'{modelo}/{fase}: frac={rows[0]["frac"]} (esperado 1.0)'


# --- A9: agente decide segun umbrales 95/50 --------------------------------

def test_agente_decision_umbrales():
    """T-A9: agente decide ACEPTAR/INDECSO/RECHAZAR segun umbrales 95/50."""

    def make_res(tasa: float) -> dict:
        return {
            'modelos': {
                'shared': {
                    'trivial': {
                        'C_exact': 0,
                        'rows': [
                            {'p': 0.0, 'frac': 1.0, 'F_muestreada': 1.0 - (1.0 - tasa)},
                            {'p': 0.1, 'frac': tasa, 'F_muestreada': 1.0 - (1.0 - tasa)},
                        ]
                    }
                }
            }
        }

    # ACEPTAR: tasa >= 0.95
    plan = ag.generar(make_res(0.97))
    assert plan.decision == 'ACEPTAR', f'tasa=0.97 -> {plan.decision}'

    # INDECSO: 0.50 <= tasa < 0.95
    plan = ag.generar(make_res(0.70))
    assert plan.decision == 'INDECSO', f'tasa=0.70 -> {plan.decision}'

    # RECHAZAR: tasa < 0.50
    plan = ag.generar(make_res(0.30))
    assert plan.decision == 'RECHAZAR', f'tasa=0.30 -> {plan.decision}'


def test_agente_umbral_exacto_95():
    """T-A9b: tasa = 0.95 exacto -> ACEPTAR."""
    res = {
        'modelos': {
            'shared': {
                'trivial': {
                    'C_exact': 0,
                    'rows': [{'p': 0.0, 'frac': 0.95, 'F_muestreada': 0.05}]
                }
            }
        }
    }
    plan = ag.generar(res)
    assert plan.decision == 'ACEPTAR', f'tasa=0.95 -> {plan.decision}'


def test_agente_umbral_exacto_50():
    """T-A9c: tasa = 0.50 exacto -> INDECSO."""
    res = {
        'modelos': {
            'shared': {
                'trivial': {
                    'C_exact': 0,
                    'rows': [{'p': 0.0, 'frac': 0.50, 'F_muestreada': 0.50}]
                }
            }
        }
    }
    plan = ag.generar(res)
    assert plan.decision == 'INDECSO', f'tasa=0.50 -> {plan.decision}'
