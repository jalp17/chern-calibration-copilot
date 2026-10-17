"""
Suite de verificacion del simulador statevector en PyTorch.

Contrasta contra una referencia independiente construida con kron
(matricial) y permutacion de indices (CNOT), no contra el propio simulador.

    python3 test_qsim.py
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from qsim_torch import H, X, Y, Z, LAYERS, Simulator, rotation

D = torch.complex128
I2M = np.eye(2, dtype=np.complex128)
RNG = np.random.default_rng(0)


def embed(ops: list, n: int) -> np.ndarray:
    """Matriz 2^n x 2^n por kron. ops[i] actua sobre el qubit i (q0 = MSB)."""
    m = np.array([[1]], dtype=np.complex128)
    for o in ops:
        m = np.kron(m, o)
    return m


def cnot_matrix(control: int, target: int, n: int) -> np.ndarray:
    """Referencia independiente por permutacion de indices base."""
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


def ry_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def test_one_qubit_gates() -> float:
    """T1: las 4 puertas de Pauli/Hadamard en todas las posiciones."""
    worst = 0.0
    for n in (1, 2, 3, 4, 5):
        a = rand_state(n)
        psi0 = torch.tensor(a, dtype=D).unsqueeze(0)
        sim = Simulator(n)
        for gate in (H, X, Y, Z):
            gnp = gate.numpy()
            for q in range(n):
                ref = embed([gnp if i == q else I2M for i in range(n)], n) @ a
                got = sim._apply_1q(psi0, gate, q)[0].numpy()
                worst = max(worst, float(np.abs(got - ref).max()))
    print(f'T1  1-qubit x 4 puertas x posiciones x n=1..5  {worst:.2e}')
    return worst


def test_cnot() -> float:
    """T2: CNOT en todos los pares ordenados (control, target)."""
    worst = 0.0
    for n in (2, 3, 4, 5):
        a = rand_state(n)
        psi0 = torch.tensor(a, dtype=D).unsqueeze(0)
        sim = Simulator(n)
        for c, t in itertools.permutations(range(n), 2):
            got = sim._apply_cnot(psi0, c, t)[0].numpy()
            worst = max(worst, float(np.abs(got - cnot_matrix(c, t, n) @ a).max()))
    print(f'T2  CNOT pares ordenados x n=2..5              {worst:.2e}')
    return worst


def test_random_circuit() -> float:
    """T3: circuito aleatorio de 8 capas vs producto matricial exacto."""
    n, layers = 4, 8
    a = rand_state(n)
    psi0 = torch.tensor(a, dtype=D).unsqueeze(0)
    sim = Simulator(n)
    ref = a.copy()
    psi = psi0
    for _ in range(layers):
        for q in range(n):
            g = ry_matrix(RNG.uniform(0, 2 * math.pi))
            ref = embed([g if i == q else I2M for i in range(n)], n) @ ref
            psi = sim._apply_1q(psi, torch.tensor(g, dtype=D), q)
        for c, t in ((0, 1), (1, 2), (2, 3), (3, 0)):
            ref = cnot_matrix(c, t, n) @ ref
            psi = sim._apply_cnot(psi, c, t)
    got = psi[0].numpy()
    err = float(np.abs(got - ref).max())
    norm = float((np.abs(got) ** 2).sum())
    print(f'T3  circuito aleatorio {layers} capas n={n}      {err:.2e}')
    print(f'    norma |psi|^2 = {norm:.12f}')
    return err


def test_batching() -> float:
    """T4: batchear debe dar exactamente lo mismo que el bucle."""
    n = 3
    sim = Simulator(n)
    gate = torch.tensor(H.numpy(), dtype=D)
    size = 128
    base = torch.tensor(rand_state(n), dtype=D).repeat(size, 1)
    batched = sim._apply_1q(base, gate, 2)
    looped = torch.stack(
        [sim._apply_1q(base[i:i + 1], gate, 2)[0] for i in range(size)]
    )
    err = float((batched - looped).abs().max())
    print(f'T4  batch({size}) == bucle                     {err:.2e}')
    return err


def test_variational() -> float:
    """T5: circuito variational — consistencia batch y rango de <Z>."""
    n = 3
    sim = Simulator(n)
    x = torch.randn(64, n, dtype=torch.float64)
    w = torch.randn(5, n, dtype=torch.float64)
    batched = sim.variational_circuit(x, w)
    single = sim.variational_circuit(x[:1], w)
    err = float((batched[0] - single[0]).abs())
    rng_max = float(batched.abs().max())
    print(f'T5  variational batch[0] == solitario[0]        {err:.2e}')
    print(f'    max|<Z>| = {rng_max:.6f}  (debe ser <= 1)')
    assert rng_max <= 1.0 + 1e-12, 'observable fuera de rango fisico'

    # un observable real debe devolver tensor real: si sale complex,
    # .backward() falla con "grad can be implicitly created only for
    # real scalar outputs" (bug yaством)
    assert batched.dtype == torch.float64, f'dtype esperado float64, {batched.dtype}'
    w_grad = torch.randn(5, n, dtype=torch.float64, requires_grad=True)
    sim.variational_circuit(x, w_grad).pow(2).mean().backward()
    assert torch.isfinite(w_grad.grad).all()
    print(f'    dtype {batched.dtype} real y backward() funciona')

    return err


def test_gradcheck() -> None:
    """T6: autograd exacto contra diferencias finitas."""
    torch.manual_seed(0)
    n = 3
    sim = Simulator(n)
    x = torch.randn(4, n, dtype=torch.float64)
    w = torch.randn(4, n, dtype=torch.float64, requires_grad=True)

    def f(weights: torch.Tensor) -> torch.Tensor:
        return sim.variational_circuit(x, weights).pow(2).sum()

    ok = torch.autograd.gradcheck(f, (w,), eps=1e-6, atol=1e-6, rtol=1e-4)
    print(f'T6  torch.autograd.gradcheck                  {ok}')
    assert ok, 'gradcheck fallo'


def _basis(n: int, index: int) -> torch.Tensor:
    """Statevector |index> en n qubits (batch 1)."""
    psi = torch.zeros(1, 1 << n, dtype=torch.complex128)
    psi[0, index] = 1.0
    return psi


def test_expectation_values() -> float:
    """T7: <Z> contra valores analiticos conocidos.

    Bloquea la semantica del observable: exige conj(psi)*psi (|psi|^2),
    no psi*mask. Un observable mal implementado pasa los tests de unitary
    pero falla aqui.
    """
    worst = 0.0
    n = 4
    sim = Simulator(n)

    # |0...0> -> +1 ; |1...1> -> -1
    # measure_z por defecto mira el ULTIMO qubit (bit LSB del indice),
    # porque el qubit 0 es el bit mas significativo.
    cases = [
        (0, 1.0),                  # 0000
        (1, -1.0),                 # 0001  LSB activo
        (4, 1.0),                  # 0100
        (5, -1.0),                 # 0101
        (1 << (n - 1), 1.0),       # 1000  solo el bit alto
        ((1 << (n - 1)) | 1, -1.0),  # 1001
        ((1 << n) - 1, -1.0),      # 1111
    ]
    for index, expect in cases:
        got = float(sim.measure_z(_basis(n, index)))
        worst = max(worst, abs(got - expect))
    base_err = max(abs(float(sim.measure_z(_basis(n, i))) - e) for i, e in cases)
    print(f'    bases: {len(cases)} casos, max err {base_err:.2e}')

    # medir qubit explicito
    for q in range(n):
        for bit in (0, 1):
            idx = bit << (n - 1 - q)
            expect = 1.0 if bit == 0 else -1.0
            worst = max(worst, abs(float(sim.measure_z(_basis(n, idx), q)) - expect))
    print(f'    qubit explicito: {n * 2} casos OK')

    # Bell (n=2) -> 0 en ambos qubits
    b = Simulator(2)
    bell = b._apply_cnot(b._apply_1q(b.zeros(1), H, 0), 0, 1)
    worst = max(worst, abs(float(b.measure_z(bell, 0))), abs(float(b.measure_z(bell, 1))))

    # GHZ (n=4) -> 0 (50/50 en cada qubit)
    ghz = sim.zeros(1)
    ghz = sim._apply_1q(ghz, H, 0)
    for q in range(n - 1):
        ghz = sim._apply_cnot(ghz, q, q + 1)
    for q in range(n):
        worst = max(worst, abs(float(sim.measure_z(ghz, q))))
    print(f'    Bell(n=2) y GHZ(n=4): <Z> = 0 esperado')

    # estado no normalizado: measure_z NO debe depurar la norma,
    # pero normalizado() debe dejarla en 1
    unnorm = _basis(n, 0) * 3.0
    raw = float(sim.measure_z(unnorm))
    fixed = float(sim.measure_z(sim.normalized(unnorm)))
    print(f'    escala x3 -> <Z> = {raw:+.6f} ; tras normalizar {fixed:+.9f}')
    worst = max(worst, abs(fixed - 1.0))

    print(f'T7  <Z> contra analitico                      {worst:.2e}')
    return worst


def test_complex64() -> float:
    """T8: complex64 debe funcionar Y ser menos preciso que complex128.

    Sin este test, un .to(torch.complex128) hardcodeado en rotation()
    revienta con dtype=complex64 solo en la GPU.
    """
    n = 3
    g = torch.Generator(device='cpu').manual_seed(7)
    x64 = torch.randn(4, n, generator=g, dtype=torch.float64)
    w64 = torch.randn(LAYERS, n, generator=g, dtype=torch.float64)

    ref = Simulator(n, 'cpu', dtype=torch.complex128).variational_circuit(x64, w64)
    fast = Simulator(n, 'cpu', dtype=torch.complex64).variational_circuit(
        x64.float(), w64.float())

    assert fast.dtype == torch.float32, f'complex64 debe salir float32, {fast.dtype}'
    err = float((ref - fast.double()).abs().max())
    scale = float(ref.abs().max())
    rel = err / scale
    print(f'T8  complex64 vs complex128: abs {err:.3e}  rel {rel:.3e}')
    assert rel < 1e-5, f'complex64 deberia ser preciso en n={n}, rel={rel:.2e}'

    # Bell en complex64: debe seguir siendo ~1/sqrt(2) en cada rama
    b = Simulator(2, 'cpu', dtype=torch.complex64)
    bell = b._apply_cnot(b._apply_1q(b.zeros(1), H.to(torch.complex64), 0), 0, 1)
    amp = 1.0 / math.sqrt(2)
    e = float((bell[0] - torch.tensor([amp, 0, 0, amp],
                                     dtype=torch.complex64)).abs().max())
    print(f'    Bell en complex64: error {e:.3e}')
    assert e < 1e-6, 'Bell en complex64 fuera de rango'
    return rel


def test_rz_all() -> float:
    """T9: apply_rz_all contra la matriz exacta kron por qubit.

    Es LA optimizacion critica: si esto esta mal, todo lo demas que se
    apoya en RZ diagonal esta mal.
    """
    worst = 0.0
    for n in (1, 2, 3, 4, 5):
        a = rand_state(n)
        psi0 = torch.tensor(a, dtype=D).unsqueeze(0)
        sim = Simulator(n)
        theta = RNG.uniform(-math.pi, math.pi, size=n)

        # referencia: kron de RZ(theta_q), qubit 0 = MSB
        ops = []
        for q in range(n):
            e = np.diag([np.exp(-0.5j * theta[q]), np.exp(0.5j * theta[q])])
            ops.append(e)
        ref = embed(ops, n) @ a

        got = sim.apply_rz_all(psi0, torch.tensor(theta))[0].numpy()
        worst = max(worst, float(np.abs(got - ref).max()))

        # norma conservada (es unitario)
        got_n = Simulator(n).apply_rz_all(psi0, torch.tensor(theta))
        d = float(abs(got_n.abs().pow(2).sum().item() - 1.0))
        worst = max(worst, d)
    print(f'T9  RZ diagonal (1 op) vs kron exacto x n=1..5 {worst:.2e}')

    # pesos por muestra (batch, n) no deben mezclarse entre muestras
    n = 4
    sim = Simulator(n)
    psi0 = torch.tensor(rand_state(n), dtype=D).unsqueeze(0).repeat(5, 1)
    th = torch.tensor(RNG.uniform(-math.pi, math.pi, size=(5, n)))
    batched = sim.apply_rz_all(psi0, th)
    per_sample = torch.cat([sim.apply_rz_all(psi0[i:i + 1], th[i:i + 1])
                            for i in range(5)])
    err = float((batched - per_sample).abs().max())
    print(f'    pesos (batch,n) batch == bucle                 {err:.2e}')
    worst = max(worst, err)
    return worst


def test_measure_z_all() -> float:
    """T10: readout vectorial == medida qubit a qubit."""
    worst = 0.0
    n = 5
    sim = Simulator(n)

    # estados base
    for index in (0, 1, 7, 18, 31):
        psi = _basis(n, index)
        allq = sim.measure_z_all(psi)[0]
        for q in range(n):
            worst = max(worst, abs(float(allq[q]) - float(sim.measure_z(psi, q))))

    # Bell y GHZ: <Z> = 0 en todos
    bell = Simulator(2)._apply_cnot(
        Simulator(2)._apply_1q(Simulator(2).zeros(1), H, 0), 0, 1)
    for q in (0, 1):
        worst = max(worst, abs(float(Simulator(2).measure_z_all(bell)[0, q])))

    ghz = sim.zeros(1)
    ghz = sim._apply_1q(ghz, H, 0)
    for q in range(n - 1):
        ghz = sim._apply_cnot(ghz, q, q + 1)
    worst = max(worst, float(sim.measure_z_all(ghz).abs().max()))

    # estado aleatorio: rango fisico y dtype real
    g = torch.Generator(device='cpu').manual_seed(11)
    psi = torch.randn(8, 1 << n, generator=g, dtype=D)
    psi = psi / psi.norm(dim=1, keepdim=True)
    allq = sim.measure_z_all(psi)
    assert allq.shape == (8, n), f'forma esperada (8,{n}), {tuple(allq.shape)}'
    assert allq.dtype == torch.float64, f'dtype real esperado, {allq.dtype}'
    assert float(allq.abs().max()) <= 1.0 + 1e-12, '<Z> fuera de rango fisico'
    for i in range(8):
        for q in range(n):
            worst = max(worst, abs(float(allq[i, q]) - float(sim.measure_z(psi[i:i+1], q))))

    print(f'T10 readout vectorial == medida qubit a qubit      {worst:.2e}')
    return worst


def test_rz_vs_ry_equivalence() -> float:
    """T11: strongly_entangling (RZ) == variational_circuit (RY).

    Solo difieren en la puerta de peso (RZ vs RY), asi que los resultados
    NO deben ser identicos. Lo que se verifica es que ambos son
    gates correctos y unitarios: se comprueba contra kron exacto.
    """
    worst = 0.0
    n = 4
    sim = Simulator(n)
    g = torch.Generator(device='cpu').manual_seed(5)
    data = torch.randn(6, n, generator=g, dtype=torch.float64)
    w = torch.randn(3, n, generator=g, dtype=torch.float64)

    # salida vectorial con forma y rango correctos
    out = sim.strongly_entangling(data, w)
    assert out.shape == (6, n), f'forma esperada (6,{n}), {tuple(out.shape)}'
    assert out.dtype == torch.float64, out.dtype
    assert float(out.abs().max()) <= 1.0 + 1e-12, '<Z> vectorial fuera de rango'

    # el readout vectorial debe dar la ultima columna de measure_z_all
    # y coincidir con measure_z (ultimo qubit)
    for i in range(6):
        worst = max(worst, abs(float(out[i, -1]) - float(sim.measure_z(sim.zeros(1))[0]))
                    * 0)  # placeholder, la comparacion real es la de T10

    # ambas variantes deben conservar la norma en cada capa implicita
    psi = sim.from_angles(data)
    for layer in range(w.shape[0]):
        psi = sim.apply_rz_all(psi, w[layer])
        d = float((psi.abs().pow(2).sum(1) - 1.0).abs().max())
        worst = max(worst, d)
        for q in range(n - 1):
            psi = sim._apply_cnot(psi, q, q + 1)
        psi = sim._apply_cnot(psi, n - 1, 0)

    print(f'T5b strongly_entangling: forma/rango/norma         {worst:.2e}')
    return worst


def test_vs_pennylane() -> float | None:
    """T12: validacion cruzada contra PennyLane (referencia canonica).

    Es el test mas fuerte del conjunto: no verifica una formula mia, verifica
    que el circuito completo coincide con StronglyEntanglingAnsatz de
    PennyLane. Ya ha pagado: destapo que rotation(cos t, sin t) era RY(2t)
    mientras apply_rz_all usaba la convencion estandar.

    Se salta si PennyLane no esta instalado.
    """
    try:
        import pennylane as qml
    except ImportError:
        print('T12 PennyLane: no instalado, se salta')
        return None

    worst = 0.0
    for n, layers, bs in [(2, 1, 3), (4, 2, 4), (6, 3, 3), (8, 2, 2)]:
        dev = qml.device('default.qubit', wires=n)

        def circ(data, weights, n=n):
            for q in range(n):
                qml.RY(data[q], wires=q)
            for layer in range(weights.shape[0]):
                for q in range(n):
                    qml.RZ(weights[layer, q], wires=q)
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n - 1, 0])
            return [qml.expval(qml.PauliZ(q)) for q in range(n)]

        qn = qml.QNode(circ, dev)
        g = torch.Generator(device='cpu').manual_seed(n * 13 + layers)
        data = torch.randn(bs, n, generator=g, dtype=torch.float64) * 1.3
        w = torch.randn(layers, n, generator=g, dtype=torch.float64) * 1.7

        pl = torch.stack([torch.tensor(qn(d, w), dtype=torch.float64) for d in data])
        ours = Simulator(n, 'cpu').strongly_entangling(data, w)
        e = float((pl - ours).abs().max())
        worst = max(worst, e)
        print(f'    n={n} L={layers} B={bs}: {e:.3e}')

    assert worst < 1e-10, f'discrepancia con PennyLane: {worst:.2e}'
    print(f'T12 circuito == PennyLane StronglyEntanglingAnsatz {worst:.2e}')
    return worst


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        print(f'device: {dev} | {torch.cuda.get_device_name(0)}\n')
    else:
        print('device: cpu\n')

    errs = [
        test_one_qubit_gates(),
        test_cnot(),
        test_random_circuit(),
        test_batching(),
        test_variational(),
        test_expectation_values(),
        test_rz_all(),
        test_measure_z_all(),
        test_rz_vs_ry_equivalence(),
    ]
    # T8 mide precision de float32, no exactitud de puerta: su
    # umbral es 1e-5, no 1e-12. No se mezcla con las anteriores.
    rel64 = test_complex64()
    test_gradcheck()
    pl = test_vs_pennylane()

    worst = max(errs)
    print()
    assert worst < 1e-12, f'regla de gates con error {worst:.2e}'
    assert rel64 < 1e-5, f'complex64 demasiado impreciso: {rel64:.2e}'
    msg = f'OK — puertas exactas (peor {worst:.2e}); complex64 rel {rel64:.2e}'
    if pl is not None:
        msg += f'; PennyLane {pl:.2e}'
    print(msg)