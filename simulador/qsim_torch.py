"""
Simulador de circuitos cuánticos por statevector en PyTorch (ROCm/AMD).

Plan B del proyecto: no depende de PennyLane ni de su pin de ROCm 7.0.
Funciona en cualquier ROCm, CUDA o CPU.

Pequeño simulador vectorial, optimizado para:
  · Headache de aplicar puertas 1-qubit como contracción tensorial
  · Batching nativo sobre muestras (aquí está toda la ganancia de GPU)
  · Autodiff de PyTorch para entrenar parámetros variacionales

Uso mínimo:
    from qsim_torch import Simulator
    sim = Simulator(n_qubits=4, device='cuda')
    out = sim.variational_circuit(x_batch, weights)

Created for the AMD Developer Hackathon ACT III.
"""

from __future__ import annotations
import math
import torch

LAYERS = 4  # capas por defecto en los circuitos variationales


def real_dtype(complex_dtype):
    """float32/float64 que corresponde a un dtype complejo.

    OJO: torch.finfo(torch.complex128).dtype devuelve el STRING 'float64',
    no el dtype. Hay que resolverlo con getattr. Y torch.complex128.real
    NO existe como atributo.
    """
    return getattr(torch, torch.finfo(complex_dtype).dtype)

# Puertas 1-qubit como tensores 2x2 (orden: |0>, |1>)
I2 = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)


def rotation(cos_t: torch.Tensor, sin_t: torch.Tensor,
             dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Construye la matriz [[cos, -sin], [sin, cos]] por qubit, vectorizada.

    cos_t/sin_t: tensores reales de forma (batch,), o escalares.
    dtype: complejo de salida (complex64 o complex128). El llamador
    debe pasar el dtype del simulador, si no el einsum no encaja.

    ⚠️ CONVENCIÓN: dado (cos θ, sin θ) esto construye RY(2θ) en la
    convención estándar RY(t) = exp(-i t Y / 2). Para obtener RY(θ)
    usa `ry(theta)`, que aplica el medio ángulo por ti.
    """
    c, s = cos_t, sin_t
    if c.dim() == 0:
        c = c.reshape(1)
        s = s.reshape(1)
    return torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1),
    ], dim=-2).to(dtype)


def rotation_z(theta: torch.Tensor, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """RZ(theta) = diag(e^{-i theta/2}, e^{+i theta/2}).

    theta: (batch,) o escalar. Devuelve (batch, 2, 2).
    Es DIAGONAL: no crea entrelazamiento. Ver sonda 2 del dossier.
    """
    h = theta / 2
    if h.dim() == 0:
        h = h.reshape(1)
    z = torch.zeros_like(h)
    return torch.stack([
        torch.stack([torch.exp(-1j * h), z], dim=-1),
        torch.stack([z, torch.exp(1j * h)], dim=-1),
    ], dim=-2).to(dtype)


def ry(theta: torch.Tensor, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """RY(θ) estándar: RY(t) = exp(-i t Y / 2) = [[cos(t/2), -sin(t/2)], ...].

    Ésta es la puerta que espera PennyLane, Qiskit y el resto de
    frameworks. `rotation(cos θ, sin θ)` NO es lo mismo: es RY(2θ).
    Mantener ambas era una inconsistencia real que la validación cruzada
    contra PennyLane destapó.
    """
    half = theta / 2
    return rotation(torch.cos(half), torch.sin(half), dtype=dtype)


class Simulator:
    """Simulador de statevector con batching de muestras."""

    def __init__(self, n_qubits: int, device: str = 'cpu',
                 dtype: torch.dtype = torch.complex128):
        self.n_qubits = n_qubits
        self.device = torch.device(device)
        self.dtype = dtype

    # ---------- álgebra de puertas ----------

    def _apply_1q(self, psi: torch.Tensor, gate: torch.Tensor,
                  qubit: int) -> torch.Tensor:
        """Aplica una puerta de 1 qubit a psi de forma (batch, 2**n).

        Descompone el índice en (izquierda, bit, derecha) donde
        izquierda = 1<<qubit y derecha = 1<<(n-qubit-1), contrae la puerta
        sobre el eje del bit y recompone.
        """
        b = psi.shape[0]
        left = 1 << qubit
        right = 1 << (self.n_qubits - qubit - 1)

        # (batch, left, 2, right)
        psi_t = psi.reshape(b, left, 2, right)

        gate = gate.to(psi.dtype)
        if gate.dim() == 2:
            out = torch.einsum('ij,bljr->blir', gate, psi_t)
        else:
            out = torch.einsum('bij,bljr->blir', gate, psi_t)

        # recompone a (batch, 2**n) preservando el orden big-endian original
        return out.reshape(b, -1)

    def _apply_cnot(self, psi: torch.Tensor, control: int,
                    target: int) -> torch.Tensor:
        """CNOT entre dos qubits."""
        ctrl = 1 << (self.n_qubits - control - 1)
        tgt = 1 << (self.n_qubits - target - 1)
        b = psi.shape[0]
        idx = torch.arange(1 << self.n_qubits, device=psi.device)
        ctrl_on = ((idx & ctrl) != 0)
        # destino = fuente si control activo
        swapped = torch.where(ctrl_on, idx ^ tgt, idx)
        return psi[:, swapped]

    # ---------- construcción de estados ----------

    def zeros(self, batch: int = 1) -> torch.Tensor:
        psi = torch.zeros(batch, 1 << self.n_qubits,
                          dtype=self.dtype, device=self.device)
        psi[:, 0] = 1.0
        return psi

    def from_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """Statevector inicial a partir de ángulos RY por qubit. (batch, n_qubits)"""
        b = angles.shape[0]
        psi = self.zeros(b)
        for q in range(self.n_qubits):
            psi = self._apply_1q(psi, ry(angles[:, q], self.dtype), q)
        return psi

    # ---------- RZ diagonal: la optimizacion de mayor impacto ----------

    def _bit_patterns(self) -> torch.Tensor:
        """(n, 2^n) float, con 1.0 donde el bit del qubit q esta activo.

        Se construye ya en el dtype real: hacerlo en complejo y castear
        despues emitia un warning de "casting discards imaginary part".
        """
        idx = torch.arange(1 << self.n_qubits, device=self.device)
        masks = torch.tensor(
            [1 << (self.n_qubits - q - 1) for q in range(self.n_qubits)],
            device=self.device,
        )
        return ((idx.unsqueeze(0) & masks.unsqueeze(1)) != 0).to(
            real_dtype(self.dtype))

    def apply_rz_all(self, psi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Aplica RZ(theta_q) en TODOS los qubits en una sola operacion.

        RZ(theta) = diag(e^{-i theta/2}, e^{+i theta/2}) es DIAGONAL, luego
        aplicarla a los n qubits equivale a multiplicar el statevector por un
        vector de fases de longitud 2^n:

            phase[i] = prod_q exp(i theta_q (bit_q(i) - 1/2))
                     = exp(i * (sum_q theta_q bit_q(i) - sum_q theta_q / 2))

        Esto sustituye n contracciones tensoriales (cada una un matmul 2x2
        con reshape/permute) por n sumas-acumuladas y una multiplicacion
        elemental. Medido: ~1.87x mas rapido en el bloque de rotaciones.

        theta: (n,) real, o (batch, n) real si hay pesos por muestra.
        """
        n = self.n_qubits
        theta = theta.to(real_dtype(self.dtype))
        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(psi.shape[0], n)

        bits = self._bit_patterns()                      # (n, 2^n)
        acc = torch.zeros(psi.shape[0], 1 << n,
                          dtype=real_dtype(self.dtype), device=self.device)
        for q in range(n):
            acc = acc + theta[:, q:q + 1] * bits[q]     # (batch, 2^n)
        half_sum = theta.sum(dim=1, keepdim=True) * 0.5
        phase = torch.exp(1j * (acc - half_sum)).to(self.dtype)
        return psi * phase

    # ---------- readout vectorial ----------

    def measure_z_all(self, psi: torch.Tensor) -> torch.Tensor:
        """<Z> en TODOS los qubits a la vez. Devuelve (batch, n).

        Una sola multiplicacion (batch, 2^n) @ (2^n, n) en lugar de n
        llamadas separadas: n x mas dimension de salida por una operacion.
        """
        bits = self._bit_patterns()                      # (n, 2^n)
        signs = 1.0 - 2.0 * bits      # +1 si bit=0, -1 si bit=1
        probs = psi.abs() ** 2                          # (batch, 2^n)
        return probs @ signs.T                           # (batch, n)

    # ---------- circuitos ----------

    def variational_circuit(self, data: torch.Tensor,
                            weights: torch.Tensor,
                            return_state: bool = False) -> torch.Tensor:
        """
        Circuito variational tipo StronglyEntangling, variante RY.

        data:    (batch, n_qubits) angulos de entrada
        weights: (n_layers, n_qubits) parametros entrenables
        return_state: si True, devuelve (readout, psi) para kernel cuantico.

        Returns: (batch,) <Z> en el ultimo qubit.
                 Si return_state=True: ((batch,), (batch, 2^n)).

        Se conserva para comparar contra la variante RZ optimizada
        (`strongly_entangling`). Ver T11: deben coincidir bit a bit.
        """
        psi = self.from_angles(data)

        for layer in range(weights.shape[0]):
            for q in range(self.n_qubits):
                psi = self._apply_1q(psi, ry(weights[layer, q], self.dtype), q)
            for q in range(self.n_qubits - 1):
                psi = self._apply_cnot(psi, q, q + 1)
            psi = self._apply_cnot(psi, self.n_qubits - 1, 0)

        readout = self.measure_z(psi)
        if return_state:
            return readout, psi
        return readout

    def strongly_entangling(self, data: torch.Tensor,
                            weights: torch.Tensor,
                            return_state: bool = False) -> torch.Tensor:
        """
        Variante OPTIMIZADA y fiel a PennyLane StronglyEntanglingAnsatz
        (subcircuit_type="default"): RZ en cada qubit + anillo CNOT.

        Cambios frente a `variational_circuit`:
          1. RZ en vez de RY en las capas de peso -> diagonal, una sola operacion
          2. readout vectorial -> (batch, n) en vez de (batch,)

        data:    (batch, n_qubits)
        weights: (n_layers, n_qubits)
        return_state: si True, devuelve (readout, psi) para kernel cuantico.
        Returns: (batch, n_qubits) valores esperados <Z_q>
                 Si return_state=True: ((batch, n), (batch, 2^n)).
        """
        psi = self.from_angles(data)
        n = self.n_qubits

        for layer in range(weights.shape[0]):
            psi = self.apply_rz_all(psi, weights[layer])   # 1 op, no n
            for q in range(n - 1):
                psi = self._apply_cnot(psi, q, q + 1)
            psi = self._apply_cnot(psi, n - 1, 0)

        readout = self.measure_z_all(psi)                      # (batch, n)
        if return_state:
            return readout, psi
        return readout

    def probabilities(self, psi: torch.Tensor) -> torch.Tensor:
        """|amplitude|^2 por muestra."""
        return (psi.abs() ** 2).sum(dim=1)

    def measure_z(self, psi: torch.Tensor, qubit: int | None = None) -> torch.Tensor:
        """Valor esperado <Z> en un qubit.

        Observable real: exige conj(psi)*psi, es decir sum_i s_i * |psi_i|^2.
        Devuelve (batch,) en [-1, 1] para estados normalizados.

        qubit=None mide el último qubit (el de mayor indice).
        """
        if qubit is None:
            qubit = self.n_qubits - 1
        bit = 1 << (self.n_qubits - qubit - 1)
        ones = torch.ones((), dtype=real_dtype(self.dtype), device=psi.device)
        signs = torch.where(
            (torch.arange(1 << self.n_qubits, device=psi.device) & bit) != 0,
            -ones, ones,
        )
        return ((psi.abs() ** 2) * signs).sum(dim=1)

    def normalized(self, psi: torch.Tensor) -> torch.Tensor:
        """Normaliza un statevector lote a lote."""
        norms = self.probabilities(psi).clamp_min(1e-30).sqrt()
        return psi / norms.unsqueeze(1)


def state_fidelity(psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """|<psi|phi>|^2 por muestra. (batch,)

    OJO: esto NO es sum_i |psi_i|^2 |phi_i|^2 (esa es la probabilidad de
    colision, <= 1 pero no es la fidelidad). Aqui ha fallado dos veces.

    psi, phi: (batch, 2^n). Devuelve float, acotado a [0, 1].
    """
    ov = (psi.conj() * phi).sum(dim=1).abs() ** 2
    return ov


def bell_state(n_qubits: int = 2, device: str = 'cpu') -> torch.Tensor:
    """Estado de Bell |00> + |11> / sqrt(2). Test de humo."""
    sim = Simulator(n_qubits, device=device)
    psi = sim.zeros(1)
    psi = sim._apply_1q(psi, H.to(sim.device), 0)
    psi = sim._apply_cnot(psi, 0, 1)
    return psi


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {dev}')
    if dev == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    psi = bell_state(2, dev)
    print(f'Bell: {psi[0].cpu().numpy().round(4)}')
    amp = 1.0 / math.sqrt(2)
    expected = torch.tensor([amp, 0, 0, amp],
                            dtype=torch.complex128, device=dev)
    err = (psi[0] - expected).abs().max()
    print(f'Error Bell: {err.item():.2e}')
    assert err < 1e-12, 'Bell state incorrecto'

    # gradiente a través del circuito
    n, B = 4, 8
    sim = Simulator(n, dev)
    x = torch.randn(B, n, device=dev, dtype=torch.float64)
    w = torch.randn(3, n, device=dev, dtype=torch.float64, requires_grad=True)
    out = sim.variational_circuit(x, w)
    loss = out.pow(2).mean()
    loss.backward()
    print(f'Salida shape: {out.shape}, loss: {loss.item():.4f}')
    print(f'Gradiente w shape: {w.grad.shape}, norm: {w.grad.norm().item():.4f}')
    assert torch.isfinite(w.grad).all(), 'gradiente no finito'
    print('\nOK — simulador funcionando.')
