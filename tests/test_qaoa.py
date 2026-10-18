"""Tests para QAOA MaxCut."""

import math
import torch

from qsim_torch import Simulator
from qaoa import (
    qaoa_maxcut, optimize_qaoa, qaoa_state,
    complete_graph, empty_graph, cycle_graph, random_graph,
    brute_force_maxcut, expectation_value,
    maxcut_matrix, matrix_from_eigenvalues,
)

D = torch.complex128


def test_qaoa_grafo_vacio():
    """Grafo sin aristas -> corte = 0."""
    sim = Simulator(4)
    edges = empty_graph(4)
    gamma = torch.tensor([0.1, 0.2], dtype=torch.float64)
    beta = torch.tensor([0.3, 0.4], dtype=torch.float64)
    c = qaoa_maxcut(sim, edges, gamma, beta)
    assert c.shape == (1,), f'forma esperada (1,), {c.shape}'
    assert abs(float(c[0])) < 1e-6, f'grafo vacio: <C>={float(c[0])} (esperado 0)'


def test_qaoa_grafo_K2():
    """K2 (una arista): el optimizador alcanza el corte maximo = 1."""
    sim = Simulator(2)
    edges = complete_graph(2)
    params, cut, opt = optimize_qaoa(sim, edges, p=1, n_steps=300, seed=0)
    assert opt == 1.0, f'K2 optimo = {opt} (esperado 1)'
    assert cut >= opt - 1e-6, f'K2 no alcanza optimo: {cut} < {opt}'


def test_qaoa_p_cero():
    """p=0 -> estado |+...+> da corte = |E|/2 (cada arista contribuye 1/2)."""
    sim = Simulator(4)
    edges = complete_graph(4)  # 6 aristas
    gamma = torch.zeros(0, dtype=torch.float64)
    beta = torch.zeros(0, dtype=torch.float64)
    c = qaoa_maxcut(sim, edges, gamma, beta)
    expected = len(edges) / 2.0
    assert abs(float(c[0]) - expected) < 1e-6, \
        f'p=0: <C>={float(c[0])} (esperado |E|/2 = {expected})'


def test_qaoa_gradientes_finitos():
    """d<C>/d gamma y d<C>/d beta no son NaN y coinciden con diferencias finitas."""
    sim = Simulator(5)
    edges = cycle_graph(5)
    gamma = torch.tensor([0.5, 0.6], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    c = qaoa_maxcut(sim, edges, gamma, beta)
    c.backward()
    assert torch.isfinite(gamma.grad).all(), 'gradiente gamma no finito'
    assert torch.isfinite(beta.grad).all(), 'gradiente beta no finito'

    # diferencias finitas
    def f(params):
        g = params[:2].reshape(1, 2)
        b = params[2:].reshape(1, 2)
        return float(qaoa_maxcut(sim, edges, g, b)[0])

    params0 = torch.tensor([0.5, 0.6, 0.3, 0.4], dtype=torch.float64)
    eps = 1e-6
    fd = []
    for i in range(4):
        xp = params0.clone(); xp[i] += eps
        xm = params0.clone(); xm[i] -= eps
        fd.append((f(xp) - f(xm)) / (2 * eps))
    fd = torch.tensor(fd).double()
    assert torch.allclose(gamma.grad.double(), fd[:2], atol=1e-4), \
        f'gradiente gamma no coincide: {gamma.grad} vs {fd[:2]}'
    assert torch.allclose(beta.grad.double(), fd[2:], atol=1e-4), \
        f'gradiente beta no coincide: {beta.grad} vs {fd[2:]}'


def test_qaoa_maxcut_matrix_kron_vs_diagonal():
    """maxcut_matrix (kron de Pauli Z) coincide con la version diagonal."""
    edges = complete_graph(4)
    M = maxcut_matrix(edges, 4)
    Mref = matrix_from_eigenvalues(edges, 4)
    err = (M - Mref).abs().max().item()
    assert err < 1e-12, f'kron vs diagonal: error {err:.2e}'
    herm = (M - M.conj().T).abs().max().item()
    assert herm < 1e-12, f'C no es hermitiano: {herm:.2e}'


def test_qaoa_optimiza_y_no_supera_optimo():
    """El corte QAOA optimizado nunca supera el optimo por fuerza bruta."""
    for n, p, edges in [(4, 2, complete_graph(4)),
                        (5, 2, cycle_graph(5)),
                        (6, 3, complete_graph(6))]:
        sim = Simulator(n)
        params, cut, opt = optimize_qaoa(sim, edges, p=p, n_steps=200, seed=0)
        assert cut <= opt + 1e-6, \
            f'{n}q p={p}: corte QAOA {cut} supera optimo {opt}'


def test_qaoa_initial_state_normalized():
    """El estado inicial |+...+> esta normalizado para n par e impar."""
    for n in (2, 3, 4, 5, 6):
        sim = Simulator(n)
        edges = complete_graph(n)
        gamma = torch.zeros(0, dtype=torch.float64)
        beta = torch.zeros(0, dtype=torch.float64)
        psi = qaoa_state(sim, edges, gamma, beta, 0)
        norm = (psi.abs() ** 2).sum().item()
        assert abs(norm - 1.0) < 1e-9, f'n={n}: norma={norm} (esperado 1)'


def test_qaoa_dataset():
    """El dataset produce grafos y cortes optimos correctos."""
    from qaoa import dataset
    grafos, cortes = dataset(n_samples=10, n=5, p_edge=0.5, seed=0)
    assert len(grafos) == 10
    assert cortes.shape == (10,)
    for k, edges in enumerate(grafos):
        opt = brute_force_maxcut(edges, 5)
        assert abs(float(cortes[k]) - opt) < 1e-9, \
            f'muestra {k}: etiqueta {cortes[k]} != optimo {opt}'