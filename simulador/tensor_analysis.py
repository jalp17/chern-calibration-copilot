"""
Evaluación de implementación tensorial para mejorar el modelo, la lógica,
el formalismo, la eficiencia o la practicidad del simulador cuántico.

Estado actual del simulador (qsim_torch.py):
  - Puertas 1-qubit: contracción tensorial einsum('ij,bljr->blir', gate, psi_t)
  - CNOT: indexación/gather (psi[:, swapped]) - cero FLOPs, solo movimiento de datos
  - Readout vectorial: (batch, 2^n) @ (2^n, n) - una sola multiplicación matriz
  - RZ en todas las capas: producto elemento a elemento por vector de fases

Este módulo evalúa dónde se pueden aplicar implementaciones tensoriales
adicionales y sus desventajas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TensorOpportunity:
    """Oportunidad de aplicación tensorial."""
    area: str
    estado_actual: str
    oportunidad_tensorial: str
    beneficio: str
    desventaja: str
    viabilidad: str  # "alta", "media", "baja"
    prioridad: str   # "alta", "media", "baja"


OPPORTUNITIES: List[TensorOpportunity] = [
    TensorOpportunity(
        area="Aplicación de puertas 1-qubit",
        estado_actual="einsum('ij,bljr->blir', gate, psi_t) por cada qubit",
        oportunidad_tensorial="Fusionar todas las puertas 1-qubit de una capa en un solo einsum batched",
        beneficio="Menos lanzamientos de kernel, mejor localidad de caché",
        desventaja="La puerta es diferente por qubit (ángulos distintos), difícil de fusionar sin perder flexibilidad",
        viabilidad="media",
        prioridad="media",
    ),
    TensorOpportunity(
        area="CNOT / puertas 2-qubit",
        estado_actual="Indexación directa psi[:, swapped] - cero FLOPs",
        oportunidad_tensorial="Representar como permutación tensorial sobre los ejes del statevector",
        beneficio="Unifica la notación con puertas 1-qubit, permite optimización del compilador",
        desventaja="La indexación actual es ya óptima (gather, no matmul). El tensorial añadiría overhead sin ganar FLOPs",
        viabilidad="baja",
        prioridad="baja",
    ),
    TensorOpportunity(
        area="Readout de observables",
        estado_actual="(batch, 2^n) @ (2^n, n) - multiplicación matriz-vector",
        oportunidad_tensorial="Tensor de orden 3: (batch, 2^n, n) contraído con matriz de signos (2^n, n)",
        beneficio="Permite readout de múltiples observables (2-local, correlaciones) en una operación",
        desventaja="Aumenta el uso de memoria intermedia (batch, 2^n, n)",
        viabilidad="alta",
        prioridad="alta",
    ),
    TensorOpportunity(
        area="Statevector como tensor de orden n",
        estado_actual="Vector plano (batch, 2^n)",
        oportunidad_tensorial="Tensor de orden n con ejes (batch, 2, 2, ..., 2)",
        beneficio="Permite operaciones locales sin reshape/permute costoso",
        desventaja="PyTorch no optimiza bien tensores de orden > 4. El reshape actual es más eficiente en la práctica",
        viabilidad="baja",
        prioridad="baja",
    ),
    TensorOpportunity(
        area="Redes tensoriales (MPS/MPO)",
        estado_actual="Statevector denso completo",
        oportunidad_tensorial="Representar el estado como Producto de Matrices (MPS) con dimensión de bonchi χ",
        beneficio="Reduce memoria de 2^n a n·χ²·d, permite simular más qubits",
        desventaja="Pierde exactitud (aproximación), requiere truncación, no es statevector exacto",
        viabilidad="baja",
        prioridad="baja",
    ),
    TensorOpportunity(
        area="Evolución temporal (Haldane td)",
        estado_actual="Puertas 1-qubit secuenciales por cada punto k",
        oportunidad_tensorial="Aplicar todas las puertas 1-qubit de todos los k en un solo einsum batched sobre (batch_k, 2^1)",
        beneficio="Paraleliza sobre los puntos k de la malla de Brillouin",
        desventaja="Los k son independientes (producto tensor), ya se paralelizan por batch",
        viabilidad="media",
        prioridad="media",
    ),
    TensorOpportunity(
        area="Matriz de densidad (Lindblad)",
        estado_actual="Matriz densa 2^n × 2^n",
        oportunidad_tensorial="Superoperador como tensor de orden 4: Liouvilliano L_{ijkl}",
        beneficio="Permite operaciones vectorizadas sobre la matriz de densidad",
        desventaja="La matriz de densidad ya es 2^n × 2^n; el superoperador sería 2^{2n} × 2^{2n}, inviable para n > 10",
        viabilidad="baja",
        prioridad="baja",
    ),
]


def print_report() -> None:
    """Imprime el informe completo de oportunidades tensoriales."""
    print("=" * 78)
    print("EVALUACIÓN DE IMPLEMENTACIÓN TENSORIAL")
    print("=" * 78)
    print("\nEstado actual del simulador:")
    print("  - Puertas 1-qubit: einsum contracción tensorial")
    print("  - CNOT: indexación/gather (cero FLOPs)")
    print("  - Readout vectorial: multiplicación matriz (batch, 2^n) @ (2^n, n)")
    print("  - RZ vectorial: producto elemento a elemento")
    print()

    for i, opp in enumerate(OPPORTUNITIES, 1):
        print(f"{i}. {opp.area}")
        print(f"   Estado actual : {opp.estado_actual}")
        print(f"   Oportunidad   : {opp.oportunidad_tensorial}")
        print(f"   Beneficio     : {opp.beneficio}")
        print(f"   Desventaja    : {opp.desventaja}")
        print(f"   Viabilidad    : {opp.viabilidad}")
        print(f"   Prioridad     : {opp.prioridad}")
        print()

    print("=" * 78)
    print("RECOMENDACIÓN GENERAL")
    print("=" * 78)
    print(
        "El simulador ya usa implementación tensorial donde tiene sentido "
        "(einsum para puertas 1-qubit, matmul para readout). Las oportunidades "
        "restantes (redes tensoriales, tensor de orden n) añaden complejidad "
        "sin beneficio práctico para n ≤ 20. El cuello de botella es memoria "
        "(2^n amplitudes), no FLOPs, por lo que optimizar el compute no ayuda."
    )
    print()
    print("Oportunidad de mayor valor: readout de orden 2 (correlaciones")
    print("⟨Z_i Z_j⟩) mediante tensor de orden 3. Viabilidad: ALTA.")
    print("=" * 78)


if __name__ == "__main__":
    print_report()
