# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from collections.abc import Collection

from numpy import concatenate
from numpy.random import default_rng
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


################################################################################
## COMPUTE-UNCOMPUTE
################################################################################
def compute_uncompute(
    circuit: QuantumCircuit, *, barrier: bool = False, inplace: bool = False
) -> QuantumCircuit:
    """Build compute-uncompute version of input quantum circuit.

    Args:
        circuit: the original quantum circuit to build compute-uncompute version of.
        barrier: whether to insert a barrier between the compute and uncompute
            halves of the output circuit.
        inplace: if True, modify the input circuit. Otherwise make a copy.

    Returns:
        Compute-uncompute version of input quantum circuit with optional barrier.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"Invalid circuit type '{type(circuit)}', expected 'QuantumCircuit'.")
    original = circuit
    if not inplace:
        circuit = original.copy()
    if barrier:
        circuit.barrier()
    circuit.compose(original.inverse(), inplace=True)
    return circuit


################################################################################
## RANDOM COMPUTATIONAL BASIS STATE
################################################################################
# TODO: QuantumCircuit subclass
def build_random_basis_state(
    num_qubits: int,
    num_excitations: int,
    *,
    favored_qubits: Collection[int] | None = None,
    seed: int | None = None,
) -> QuantumCircuit:
    """Build a random computational basis state quantum circuit.

    Args:
        num_qubits: number of qubits (needs to be even).
        num_excitations: number of qubits in the excited (1) state.
        favored_qubits: qubits to favor when locating random excitations.
        seed: to randomly choose excitations.

    Returns:
        A random computational basis state quantum circuit.
    """
    favored_qubits = set(favored_qubits or {})
    unfavored_qubits = [qubit for qubit in range(num_qubits) if not qubit in favored_qubits]

    rng = default_rng(seed)
    favored_qubits = rng.permutation(list(favored_qubits))
    unfavored_qubits = rng.permutation(list(unfavored_qubits))
    excited_qubits = concatenate([favored_qubits, unfavored_qubits])[:num_excitations]

    circuit = QuantumCircuit(num_qubits)
    circuit.x(excited_qubits)
    return circuit


################################################################################
## QAOA
################################################################################
def build_qaoa_circuit(
    num_qubits: int,
    depth: int,
    *,
    measurements: bool = True,
    barriers: bool = False,
) -> QuantumCircuit:
    """Build parameterized QAOA quantum circuit.

    Args:
        num_qubits: number of qubits.
        depth: two-qubit depth (needs to be even).
        measurements: if True adds measurements at the end.
        barriers: if True adds barriers between layers.

    Returns:
        A dense QAOA quantum circuit for a linear, non-cyclic, graph
        with cost parameter-vector γ, and mixer parameter-vector β.
    """
    if num_qubits <= 2:
        raise ValueError("Number of qubits must be greater than two.")
    if depth % 2 != 0:
        raise ValueError("Depth must be even.")

    gammas = ParameterVector("γ", depth // 2)
    betas = ParameterVector("β", depth // 2)

    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    for layer in range(depth // 2):
        if barriers:
            circuit.barrier()
        for qubit in range(0, num_qubits - 1, 2):
            circuit.rzz(gammas[layer], qubit, qubit + 1)
        for qubit in range(1, num_qubits - 1, 2):
            circuit.rzz(gammas[layer], qubit, qubit + 1)
        for qubit in range(num_qubits):
            circuit.rx(betas[layer], qubit)

    if measurements:
        circuit.measure_all()
    return circuit
