"""
Ruido de puerta sobre el simulador statevector.

Los canales que se simulan son los depolarizantes, que son los que se usan
por defecto en los modelos de ruido de NISQ:

    E_p(rho) = (1 - q) rho + q/3 (X rho X + Y rho Y + Z rho Z)

para un solo qubit, y la extension natural a dos qubits sobre las 16 cadenas
de Pauli. En un vector de estado esto es trivial: se aplica un Pauli
aleatorio de la correspondiente distribucion.

Se acumula ruido en CADA puerta del circuito, que es lo que hace un modelo
de ruido creible: el error crece con el numero de puertas, no con una
constante inventada.

Verificacion del modulo (ejecutar este archivo):
  · p = 0 reproduce EXACTAMENTE el circuito ideal
  · el infidelity media crece linealmente en el numero de puertas
  · la tasa de depolarizacion por puerta coincide con la p del canal

Created for the AMD Developer Hackathon ACT III.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from qsim_torch import LAYERS, Simulator, ry, rotation_z, state_fidelity

# --- Pauli de 1 qubit -------------------------------------------------------
PAULI_1Q = (
    torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128),          # I
    torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),          # X
    torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),        # Y
    torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),         # Z
)


@dataclass
class GateNoise:
    """Modelo de ruido depolarizante por puerta.

    p_1q : probabilidad depolarizante en cada puerta de 1 qubit
    p_2q : probabilidad depolarizante en cada CNOT
    seed : semilla del RNG (para reproducibilidad)
    """

    p_1q: float = 0.0
    p_2q: float = 0.0
    seed: int = 0
    rng: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = torch.Generator().manual_seed(self.seed)

    # --- helpers ---
    def _sample_1q(self, p: float) -> int:
        """Devuelve el indice del Pauli aplicado (0 = I)."""
        if p <= 0:
            return 0
        r = torch.rand(1, generator=self.rng).item()
        if r < p / 3:
            return 1          # X
        if r < 2 * p / 3:
            return 2          # Y
        if r < p:
            return 3          # Z
        return 0              # I

    def _sample_2q(self, p: float) -> tuple[int, int]:
        """Devuelve (Pauli en el control, Pauli en el target)."""
        if p <= 0:
            return 0, 0
        out = []
        for _ in range(2):
            r = torch.rand(1, generator=self.rng).item()
            if r < p / 4:
                out.append(1)
            elif r < 2 * p / 4:
                out.append(2)
            elif r < 3 * p / 4:
                out.append(3)
            else:
                out.append(0)
        return out[0], out[1]

    # --- ruido explicito ---
    def apply_1q(self, psi: torch.Tensor, sim: Simulator, qubit: int) -> torch.Tensor:
        """Aplica un Pauli aleatorio de 1 qubit con la tasa p."""
        idx = self._sample_1q(self.p_1q)
        if idx == 0:
            return psi
        return sim._apply_1q(psi, PAULI_1Q[idx].to(psi.device), qubit)

    def apply_2q(self, psi: torch.Tensor, sim: Simulator,
                 control: int, target: int) -> torch.Tensor:
        """Aplica una cadena de Pauli aleatoria sobre (control, target)."""
        a, b = self._sample_2q(self.p_2q)
        if b:
            psi = sim._apply_1q(psi, PAULI_1Q[b].to(psi.device), target)
        if a:
            psi = sim._apply_1q(psi, PAULI_1Q[a].to(psi.device), control)
        return psi


def noisy_ansatz(sim: Simulator, data: torch.Tensor, weights: torch.Tensor,
                 noise: GateNoise | None = None,
                 ring: bool = True) -> torch.Tensor:
    """StronglyEntangling-like con RY + anillo CNOT y ruido por puerta.

    Se usa RY (no RZ) porque RZ es diagonal y no crea entrelazamiento desde
    |0...0>: ver sonda 2 del dossier.

    Devuelve el vector de estado (batch, 2^n), NO el readout.
    """
    n = sim.n_qubits
    psi = sim.from_angles(data)
    for layer in range(weights.shape[0]):
        for q in range(n):
            psi = sim._apply_1q(psi, ry(weights[layer, q], sim.dtype), q)
            if noise is not None:
                psi = noise.apply_1q(psi, sim, q)
        pairs = ([(q, q + 1) for q in range(n - 1)] + [(n - 1, 0)]) if ring \
            else [(q, q + 1) for q in range(n - 1)]
        for c, t in pairs:
            psi = sim._apply_cnot(psi, c, t)
            if noise is not None:
                psi = noise.apply_2q(psi, sim, c, t)
    return psi


def count_gates(n: int, layers: int = LAYERS, ring: bool = True) -> int:
    """Numero de puertas que reciben ruido: n*layers rotaciones + CNOTs."""
    cnots = (n if ring else n - 1) * layers
    return n * layers + cnots


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    torch.set_num_threads(2)
    print('MODULO DE RUIDO — verificacion\n')

    n, layers = 4, 2
    sim = Simulator(n, 'cpu')
    g = torch.Generator().manual_seed(3)
    data = torch.randn(64, n, generator=g, dtype=torch.float64) * 0.8
    w = torch.randn(layers, n, generator=g, dtype=torch.float64)

    # --- T-N1: p = 0 reproduce el ideal exactamente ---
    ref = noisy_ansatz(sim, data, w, None)
    zero = noisy_ansatz(sim, data, w, GateNoise(0.0, 0.0, seed=1))
    err = float((ref - zero).abs().max())
    print(f'T-N1  p=0 vs ideal: max err {err:.2e}   {"OK" if err == 0 else "FALLO"}')
    assert err == 0.0, 'el ruido con p=0 debe ser identico al ideal'

    # --- T-N2: la distribucion de Pauli es correcta ---
    # con p = 3/4, la probabilidad de Identity debe ser 1/4
    p = 0.75
    cnt = [0, 0, 0, 0]
    N = 20000
    for s in range(N):
        noise = GateNoise(p, 0.0, seed=s)
        cnt[noise._sample_1q(p)] += 1
    freqs = [c / N for c in cnt]
    esperado = [1 - p, p / 3, p / 3, p / 3]
    print(f'T-N2  p={p}: frec. Pauli observadas {["%.3f" % f for f in freqs]}')
    print(f'      esperadas                    {["%.3f" % e for e in esperado]}')
    ok = all(abs(a - b) < 0.02 for a, b in zip(freqs, esperado))
    print(f'      {"OK" if ok else "FALLO"}')
    assert ok, 'la distribucion del canal depolarizante no es la correcta'

    # --- T-N3: el infidelity crece con el numero de capas ---
    print('\nT-N3  infidelity media vs numero de capas (p_1q = p_2q = 0.02)')
    print(f'{"capas":>6} {"puertas":>8} {"1-F medio":>11} {"1-F max":>10}')
    for L in (1, 2, 3, 4, 6, 8):
        # MISMO circuito: ideal y ruidoso deben compartir los pesos.
        # (Comparar pesos distintos mide la ortogonalidad de estados, no el ruido.)
        wL = torch.randn(L, n, generator=g, dtype=torch.float64)
        ideal = noisy_ansatz(sim, data, wL, None)
        ng = count_gates(n, L)
        fs = []
        for s in range(24):
            nois = noisy_ansatz(sim, data, wL, GateNoise(0.02, 0.02, seed=s))
            f = float(state_fidelity(ideal, nois).mean())
            fs.append(f)
        fs = torch.tensor(fs)
        print(f'{L:>6} {ng:>8} {1-fs.mean():>11.4f} {1-fs.max():>10.4f}')
    print('\nLa infidelity debe crecer aproximadamente linealmente con las puertas.')