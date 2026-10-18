"""
Evaluación del uso de puertas multi-qubit en el simulador.

Estado actual de qsim_torch.py:
  - Puertas 1-qubit: _apply_1q (einsum contracción tensorial)
  - CNOT: _apply_cnot (indexación/gather, 2 qubits)
  - No hay puertas de 3 o más qubits

Este módulo evalúa:
  1. ¿Se usan puertas de más de 1 qubit? (sí: CNOT usa 2)
  2. ¿Se podrían usar puertas de 3+ qubits?
  3. Beneficios y dificultades de cada enfoque.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class GateEval:
    """Evaluación de un tipo de puerta."""
    nombre: str
    num_qubits: int
    usada: bool
    implementacion: str
    beneficio: str
    dificultad: str
    veredicto: str


GATE_EVALS: List[GateEval] = [
    GateEval(
        nombre="Puerta 1-qubit (RY, RZ, H, X, Y, Z)",
        num_qubits=1,
        usada=True,
        implementacion="_apply_1q con einsum('ij,bljr->blir', gate, psi_t)",
        beneficio="Operación básica, vectorizada sobre batch, diagonaliizable (RZ)",
        dificultad="Ninguna. Está optimizada.",
        veredicto="Mantener. Es la piedra angular del simulador.",
    ),
    GateEval(
        nombre="CNOT (2 qubits)",
        num_qubits=2,
        usada=True,
        implementacion="_apply_cnot con indexación psi[:, swapped]",
        beneficio="Cero FLOPs (solo gather), crea entrelazamiento, esencial para universalidad",
        dificultad="Ninguna. Ya está optimizada.",
        veredicto="Mantener. Es la puerta 2-qubit estándar.",
    ),
    GateEval(
        nombre="Toffoli / CCNOT (3 qubits)",
        num_qubits=3,
        usada=False,
        implementacion="No implementada. Requeriría einsum sobre 3 ejes del statevector",
        beneficio="Puerta universal clásica, útil para circuitos de corrección de errores",
        dificultad="Alta. La contracción tensorial sobre 3 ejes es O(2^3) por elemento, "
                   "y el reshape/permute es costoso. Para n qubits totales, el costo es "
                   "O(batch · 2^n) como cualquier puerta, pero con mayor overhead de kernel.",
        veredicto="No implementar. No se necesita para las tareas del hackathon "
                  "(VQC, QAOA, Haldane, Lindblad, surface code).",
    ),
    GateEval(
        nombre="Puerta Controlled-U general (2 qubits + parámetro)",
        num_qubits=2,
        usada=False,
        implementacion="No implementada. Se puede construir como CNOT + rotaciones 1-qubit",
        beneficio="Más expresiva que CNOT para ciertos algoritmos",
        dificultad="Media. Se descompone en CNOT + puertas 1-qubit estándar.",
        veredicto="No implementar. La descomposición en CNOT + 1-qubit es suficiente "
                  "y más eficiente (menos kernels).",
    ),
    GateEval(
        nombre="Swap (2 qubits)",
        num_qubits=2,
        usada=False,
        implementacion="No implementada. Tres CNOTs equivalen a un Swap",
        beneficio="Útil para reordenar qubits en el statevector",
        dificultad="Baja. Se implementa como 3 CNOTs.",
        veredicto="No implementar directamente. Usar 3 CNOTs si es necesario.",
    ),
]


def print_report() -> None:
    """Imprime el informe de evaluación de puertas multi-qubit."""
    print("=" * 78)
    print("EVALUACIÓN DE PUERTAS MULTI-QUBIT")
    print("=" * 78)
    print("\nConvención de qubits: big-endian (qubit 0 = MSB, bit 1<<(n-q-1))")
    print()

    for i, gate in enumerate(GATE_EVALS, 1):
        print(f"{i}. {gate.nombre}")
        print(f"   Qubits       : {gate.num_qubits}")
        print(f"   ¿Usada?      : {'SÍ' if gate.usada else 'NO'}")
        print(f"   Implementación: {gate.implementacion}")
        print(f"   Beneficio    : {gate.beneficio}")
        print(f"   Dificultad   : {gate.dificultad}")
        print(f"   Veredicto    : {gate.veredicto}")
        print()

    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(
        "El simulador usa puertas de 1 y 2 qubits (CNOT). No hay puertas de 3+ qubits. "
        "Esto es suficiente para:"
    )
    print("  - Circuitos variacionales (VQC, StronglyEntanglingAnsatz)")
    print("  - QAOA (MaxCost + mezclador con X de 1 qubit)")
    print("  - Evolución temporal de Haldane (puertas 1-qubit por punto k)")
    print("  - Surface code (estabilizadores de 4 qubits = producto de 4 CNOT/1-qubit)")
    print()
    print(
        "Las puertas de 3+ qubits (Toffoli, etc.) no son necesarias para el "
        "alcante del proyecto y añadirían complejidad sin beneficio práctico."
    )
    print("=" * 78)


if __name__ == "__main__":
    print_report()
