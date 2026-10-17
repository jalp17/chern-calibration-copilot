"""Categoria B: ERRORES Y CASOS LIMITE.

Verifica que el sistema maneja correctamente:
- Simulator(0)
- batch=0, Nk=0, Nk=1
- p=1.0 (ruido total)
- p negativo o >1
- fhs con estados nulos
- state_fidelity con tensores de distinta forma
- LLM sin clave, con clave invalida, con endpoint caido
- agente.generar({})
- corpus vacio o ausente
"""

from __future__ import annotations

import math
import os
from unittest.mock import patch

import numpy as np
import torch

from qsim_torch import Simulator, state_fidelity
from barrido_v2 import fhs, prepare, bz, sample_pauli_idx
from haldane import honeycomb, occupied_band, hamiltonian
from agente import agente as ag
from agente import llm

D = torch.complex128


# --- B1: Simulator(0) -------------------------------------------------------

def test_simulator_n0():
    """T-B1: Simulator(0) — 1<<0 = 1, estado de 1 amplitud."""
    sim = Simulator(0)
    psi = sim.zeros(1)
    assert psi.shape == (1, 1), f'forma esperada (1,1), {psi.shape}'
    assert float(psi[0, 0].abs()) == 1.0


# --- B2: batch=0 -----------------------------------------------------------

def test_batch_0():
    """T-B2: batch=0 no debe crashear."""
    sim = Simulator(2)
    psi = sim.zeros(0)
    assert psi.shape == (0, 4), f'forma esperada (0,4), {psi.shape}'

    # apply_rz_all con batch=0
    theta = torch.tensor([0.1, 0.2], dtype=torch.float64)
    out = sim.apply_rz_all(psi, theta)
    assert out.shape == (0, 4), f'apply_rz_all: {out.shape}'


# --- B3: Nk=0 y Nk=1 -------------------------------------------------------

def test_nk_0():
    """T-B3: Nk=0 da malla vacia."""
    net = honeycomb()
    ks = bz(net, 0)
    assert ks.shape == (0, 0, 2), f'forma esperada (0,0,2), {ks.shape}'


def test_nk_1():
    """T-B3b: Nk=1 da un solo punto k."""
    net = honeycomb()
    ks = bz(net, 1)
    assert ks.shape == (1, 1, 2), f'forma esperada (1,1,2), {ks.shape}'


# --- B4: p=1.0 (ruido total) ----------------------------------------------

def test_p_1_no_crashea():
    """T-B4: p=1.0 (ruido total) no debe crashea."""
    sim = Simulator(1)
    B = 4
    phi = torch.tensor([[1.0, 0.0]], dtype=D).repeat(B, 1)
    gen = torch.Generator().manual_seed(0)

    idx = sample_pauli_idx(B, 3, 1.0, gen, torch.device('cpu'))
    assert idx.shape == (B, 3), f'forma esperada ({B},3), {idx.shape}'
    # con p=1.0, nunca debe salir I (0)
    assert (idx != 0).all(), 'con p=1.0 no deberia salir Identity'

    # prepare con p=1.0
    with torch.no_grad():
        psi = prepare(sim, phi, 2, idx, 'indep')
    assert psi.shape == (B, 2), f'prepare: {psi.shape}'
    assert torch.isfinite(psi).all(), 'prepare con p=1.0 da NaN'


# --- B5: p negativo o >1 ---------------------------------------------------

def test_p_negativo():
    """T-B5: p negativo — sample_pauli_idx devuelve zeros (sin ruido)."""
    gen = torch.Generator().manual_seed(0)
    idx = sample_pauli_idx(4, 3, -0.5, gen, torch.device('cpu'))
    assert (idx == 0).all(), 'p<0 deberia dar todo Identity'


def test_p_mayor_1():
    """T-B5b: p>1 — no valida, pero no crashea (comportamiento indefinido)."""
    gen = torch.Generator().manual_seed(0)
    # p=1.5: r < 0.5 siempre, así que siempre X. No crashea.
    idx = sample_pauli_idx(4, 3, 1.5, gen, torch.device('cpu'))
    assert idx.shape == (4, 3), f'forma esperada (4,3), {idx.shape}'


# --- B6: fhs con estados nulos --------------------------------------------

def test_fhs_estados_nulos():
    """T-B6: fhs con estados nulos — NaN controlado."""
    N = 4
    # estado nulo
    u_null = torch.zeros(1, N, N, 2, dtype=D)
    C, mn = fhs(u_null)
    assert torch.isnan(C[0]).all(), f'fhs con estado nulo deberia dar NaN, {C}'
    assert float(mn[0]) == 0.0, f'min|overlap| deberia ser 0, {float(mn[0])}'


# --- B7: state_fidelity con tensores de distinta forma -------------------

def test_state_fidelity_distinta_forma():
    """T-B7: state_fidelity con formas incompatibles debe fallar."""
    psi = torch.randn(2, 4, dtype=D)
    phi = torch.randn(2, 8, dtype=D)
    try:
        state_fidelity(psi, phi)
        assert False, 'deberia haber fallado con formas incompatibles'
    except (RuntimeError, ValueError):
        pass  # esperado


# --- B8: LLM sin clave, con clave invalida, con endpoint caido -----------

def test_llm_sin_clave():
    """T-B8: LLM sin clave — fallback a local."""
    with patch.dict(os.environ, {}, clear=True):
        r = llm.preguntar('system', 'user')
    assert r.via == 'local', f'sin clave: via={r.via}'
    assert r.error is not None


def test_llm_clave_invalida():
    """T-B8b: LLM con clave invalida — fallback a error."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'rc-invalid-key-for-testing',
        'OPENAI_BASE_URL': 'http://localhost:1',
        'LLM_MODEL': 'test-model'
    }, clear=True):
        r = llm.preguntar('system', 'user', timeout=2)
    assert r.via == 'error', f'clave invalida: via={r.via}'
    assert r.error is not None


def test_llm_endpoint_caido():
    """T-B8c: LLM con endpoint caido — fallback a error."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'rc-' + 'a' * 48,
        'OPENAI_BASE_URL': 'http://localhost:9999',
        'LLM_MODEL': 'test-model'
    }, clear=True):
        r = llm.preguntar('system', 'user', timeout=2)
    assert r.via == 'error', f'endpoint caido: via={r.via}'
    assert r.error is not None


# --- B9: agente.generar({}) -----------------------------------------------

def test_agente_json_vacio():
    """T-B9: agente.generar({}) — JSON vacio."""
    plan = ag.generar({})
    assert plan.decision == 'SIN DATOS', f'vacio: {plan.decision}'
    assert plan.aviso != '', 'deberia tener aviso'


# --- B10: corpus vacio o ausente ------------------------------------------

def test_agente_corpus_vacio():
    """T-B10: agente con corpus vacio — plan local sin fuentes."""
    res = {
        'modelos': {
            'shared': {
                'trivial': {
                    'C_exact': 0,
                    'rows': [{'p': 0.0, 'frac': 0.3, 'F_muestreada': 0.7}]
                }
            }
        }
    }
    plan = ag.generar(res)
    assert plan.decision == 'RECHAZAR', f'corpus vacio: {plan.decision}'
    # las fuentes deben existir (aunque el corpus este vacio, el plan local
    # siempre cita las secciones del corpus)
    assert len(plan.fuentes) > 0, 'deberia tener fuentes del corpus'
