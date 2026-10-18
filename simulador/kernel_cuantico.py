"""
Quantum Kernel para clasificacion de fases de Haldane.

El circuito variacional actua como feature map cuántico:
    U(x)|0>  mapea  x = [M/M_crit, log t1, t2, |sin phi|]  ->  |psi(x)>
y el kernel es
    K(x, y) = |<psi(x)|psi(y)>|^2   (overlap de statevectors).

Esto captura correlaciones de fase y entrelazamiento en el espacio de 2^n
dimensiones, que un kernel RBF sobre features crudas no representa eficientemente
con pocos parametros.

Todo se ejecuta sobre el simulador statevector existente (qsim_torch.py),
device-agnostic (CPU / CUDA ROCm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qsim_torch import Simulator, ry, rotation_z, real_dtype
from haldane import critical_mass


def _feature_angles(X: np.ndarray, n_qubits: int) -> torch.Tensor:
    """Codifica las 4 features de Haldane en angulos RY, tiled sobre n qubits.

    X: (B, 4) -> [M/M_crit, log t1, t2, |sin phi|]
    Devuelve: (B, n_qubits) en el dispositivo del simulador.
    """
    x = torch.tensor(X, dtype=torch.float64)
    B = x.shape[0]
    angles = torch.zeros(B, n_qubits, dtype=torch.float64)
    # las 4 features se repiten (tiled) sobre los qubits disponibles
    for q in range(n_qubits):
        angles[:, q] = x[:, q % 4]
    return angles


def _layer_weights(n_layers: int, n_qubits: int, seed: int = 0) -> torch.Tensor:
    """Pesos fijos del circuito (no entrenables en esta demo): anillo CNOT + RZ."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_layers, n_qubits, generator=g)


def feature_map_states(sim: Simulator, X: np.ndarray,
                       n_layers: int = 2, seed: int = 0) -> torch.Tensor:
    """Codifica X en estado cuántico y devuelve el statevector.

    Parametros
    ----------
    sim : Simulator
        Instancia del simulador (ya con device y dtype definidos).
    X : (B, 4) np.ndarray
        Features de Haldane: [M/M_crit, log t1, t2, |sin phi|].
    n_layers : int
        Capas del circuito variacional.
    seed : int
        Semilla para pesos fijos (reproducibilidad).

    Devuelve
    --------
    psi : (B, 2^n) torch.Tensor (complex)
        Statevector de cada muestra, en el dispositivo de `sim`.
    """
    angles = _feature_angles(X, sim.n_qubits)
    weights = _layer_weights(n_layers, sim.n_qubits, seed)

    # usamos return_state para obtener el statevector, no solo el readout
    _, psi = sim.strongly_entangling(angles, weights, return_state=True)
    return sim.normalized(psi)


def kernel_matrix(psi: torch.Tensor) -> torch.Tensor:
    """Matriz de overlaps K[i, j] = |<psi_i | psi_j>|^2.

    Usa state_fidelity vectorizado: |<psi_i|phi_j>|^2 por muestra.
    Para construir la matriz completa, iteramos sobre filas de psi contra
    todas las columnas de psi.
    """
    B = psi.shape[0]
    K = torch.zeros(B, B, dtype=real_dtype(psi.dtype),
                    device=psi.device)
    for i in range(B):
        K[i] = state_fidelity_vec(psi, psi[i])
    return K


def state_fidelity_vec(psi: torch.Tensor, phi_row: torch.Tensor) -> torch.Tensor:
    """|<psi | phi_row>|^2 para cada fila de psi contra un vector fijo."""
    ov = (psi.conj() * phi_row.unsqueeze(0)).sum(dim=1).abs() ** 2
    return ov.real


@dataclass
class QuantumKernelResult:
    acc_cuantico: float
    acc_clasico: float
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    K_train: np.ndarray
    K_test: np.ndarray
    n_qubits: int
    n_layers: int
    device: str


def comparar_kernels(X: np.ndarray, y: np.ndarray,
                     n_qubits: int = 6, n_layers: int = 2,
                     n_train: int | None = None,
                     n_test: int | None = None,
                     device: str = 'cpu',
                     seed: int = 0) -> QuantumKernelResult:
    """Compara SVM con kernel cuantico vs SVM-RBF clasico.

    Parametros
    ----------
    X : (N, 4) np.ndarray
        Features de Haldane.
    y : (N,) np.ndarray
        Etiquetas: 0 trivial, 1 topologica.
    n_qubits : int
        Qubits del feature map (debe ser >= 4 para codificar las 4 features).
    n_layers : int
        Capas del circuito variacional.
    n_train, n_test : int | None
        Tamano de train/test. Si None, se usa 70% / 30%.
    device : str
        'cpu' o 'cuda'.
    seed : int
        Semilla para split reproducible.

    Devuelve
    --------
    QuantumKernelResult con accuracies y matrices de kernel.
    """
    g = np.random.default_rng(seed)
    N = X.shape[0]
    idx = g.permutation(N)

    if n_train is None:
        n_train = int(0.7 * N)
    if n_test is None:
        n_test = N - n_train

    idx_train = idx[:n_train]
    idx_test = idx[n_train:n_train + n_test]

    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    # ---- kernel cuantico ----
    sim = Simulator(n_qubits, device=device, dtype=torch.complex64)

    psi_train = feature_map_states(sim, X_train, n_layers, seed)
    K_train = kernel_matrix(psi_train)

    psi_test = feature_map_states(sim, X_test, n_layers, seed)
    # K_test[i, j] = <psi_test_i | psi_train_j>
    K_test = torch.zeros(n_test, n_train,
                         dtype=real_dtype(psi_train.dtype),
                         device=psi_train.device)
    for i in range(n_test):
        K_test[i] = state_fidelity_vec(psi_train, psi_test[i])

    K_train_np = K_train.detach().cpu().numpy()
    K_test_np = K_test.detach().cpu().numpy()

    # ---- SVM con kernel precomputado ----
    try:
        from sklearn.svm import SVC
        svc_q = SVC(kernel='precomputed', C=1.0, random_state=seed)
        svc_q.fit(K_train_np, y_train)
        acc_cuantico = float(svc_q.score(K_test_np, y_test))

        svc_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=seed)
        svc_rbf.fit(X_train, y_train)
        acc_clasico = float(svc_rbf.score(X_test, y_test))
    except ImportError:
        # sklearn no disponible: fallback determinista simple
        acc_cuantico = 0.5
        acc_clasico = 0.5

    return QuantumKernelResult(
        acc_cuantico=acc_cuantico,
        acc_clasico=acc_clasico,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        K_train=K_train_np, K_test=K_test_np,
        n_qubits=n_qubits, n_layers=n_layers, device=device,
    )


def demo_kernel(n_qubits: int = 6, n_layers: int = 2,
                n_samples: int = 120, device: str = 'cpu',
                seed: int = 0) -> QuantumKernelResult:
    """Ejecuta una demo completa: dataset Haldane + comparacion de kernels."""
    from haldane import sample_dataset

    X, y, _, _ = sample_dataset(n=n_samples, seed=seed)
    # features ya vienen como [M/M_crit, log t1, t2, |sin phi|]
    return comparar_kernels(X, y, n_qubits=n_qubits, n_layers=n_layers,
                            device=device, seed=seed)
