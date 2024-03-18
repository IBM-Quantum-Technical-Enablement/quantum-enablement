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

"""Quantum Approximate Optimization Algorithm (QAOA) quantum circuits.

[1] Farhi et.al. A Quantum Approximate Optimization Algorithm,
    https://arxiv.org/abs/1411.4028
"""


from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# TODO: add graph max-cut weights input arg
class QAOAPathCircuit(QuantumCircuit):
    """Parameterized QAOA acyclic line graph quantum circuit.

    The cost parameter-vector is labeled γ, and the mixer
    parameter-vector β. Overall, there will be one parameter per
    unit of two-qubit depth: half in γ and half in ß [1].
    Weights in the generating max-cut graph are all equal to one.

    Args:
        num_qubits: number of qubits (must be even).
        depth: two-qubit depth (must be even).
        barriers: if True adds barriers between layers.
        measurements: if True adds measurements at the end.

    Notes:
        [1] Farhi et.al. A Quantum Approximate Optimization Algorithm,
            https://arxiv.org/abs/1411.4028
    """

    def __init__(
        self, num_qubits: int, depth: int, *, barriers: bool = False, measurements: bool = False
    ) -> None:
        num_qubits = _validate_qaoa_num_qubits(num_qubits)
        depth = _validate_qaoa_depth(depth)
        barriers = bool(barriers)
        measurements = bool(measurements)

        super().__init__(num_qubits, name=f"QAOAPathCircuit<{num_qubits}, {depth}>")

        gammas = ParameterVector("γ", depth // 2)
        betas = ParameterVector("β", depth // 2)

        self.h(range(num_qubits))
        for layer in range(depth // 2):
            if barriers:
                self.barrier()
            for qubit in range(0, num_qubits - 1, 2):
                self.rzz(gammas[layer], qubit, qubit + 1)
            for qubit in range(1, num_qubits - 1, 2):
                self.rzz(gammas[layer], qubit, qubit + 1)
            for qubit in range(num_qubits):
                self.rx(betas[layer], qubit)
        if measurements:
            self.measure_all()


def _validate_qaoa_num_qubits(num_qubits: int) -> int:
    """Validate number of qubits for QAOA circuits."""
    # pylint: disable=duplicate-code
    if not isinstance(num_qubits, int):
        raise TypeError(f"Invalid num. qubits type {type(num_qubits)}, expected <int>.")
    if num_qubits <= 2:
        raise ValueError(f"Number of qubits ({num_qubits}) must be greater than two.")
    if num_qubits % 2:
        raise ValueError(f"Number of qubits ({num_qubits}) must be even.")
    return num_qubits


def _validate_qaoa_depth(depth: int) -> int:
    """Validate depth for QAOA circuits."""
    # pylint: disable=duplicate-code
    if not isinstance(depth, int):
        raise TypeError(f"Invalid depth type {type(depth)}, expected <int>.")
    if depth < 0:
        raise ValueError(f"Depth ({depth}) must be positive.")
    if depth % 2:
        raise ValueError(f"Depth ({depth}) must be even.")
    return depth
