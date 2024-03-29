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

"""Many-body localization (MBL) quantum circuits.

[1] Shtanko et.al. Uncovering Local Integrability in Quantum Many-Body Dynamics,
    https://arxiv.org/abs/2307.07552
"""


from numpy import pi
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit


class MBLChainCircuit(QuantumCircuit):
    """Parameterized MBL non-periodic chain (i.e. 1D) quantum circuit.

    Parameters correspond to interaction strength (θ), and
    disorders vector (φ) with one entry per qubit. In 1D,
    θ < 0.16π ≈ 0.5 corresponds to the MBL regime; beyond such
    critical value the dynamics become ergodic (i.e. thermal).
    Disorders are random on a qubit by qubit basis [1].

    Args:
        num_qubits: number of qubits (must be even).
        depth: two-qubit depth (must be even).
        barriers: if True adds barriers between layers.
        measurements: if True adds measurements at the end.

    Notes:
        [1] Shtanko et.al. Uncovering Local Integrability in Quantum
            Many-Body Dynamics, https://arxiv.org/abs/2307.07552
    """

    def __init__(
        self, num_qubits: int, depth: int, *, barriers: bool = False, measurements: bool = False
    ) -> None:
        num_qubits = _validate_mbl_num_qubits(num_qubits)
        depth = _validate_mbl_depth(depth)
        barriers = bool(barriers)
        measurements = bool(measurements)

        super().__init__(num_qubits, name=f"MBLChainCircuit<{num_qubits}, {depth}>")

        self.x(range(1, num_qubits, 2))  # TODO: add optional initial state arg
        if barriers and depth > 0:
            self.barrier()
        evolution = MBLChainEvolution(num_qubits, depth, barriers=barriers)
        self.compose(evolution, inplace=True)
        if measurements:
            self.measure_all(inplace=True, add_bits=True)


class MBLChainEvolution(QuantumCircuit):
    """Parameterized MBL non-periodic chain (i.e. 1D) evolution quantum circuit.

    Parameters correspond to interaction strength (θ), and
    disorders vector (φ) with one entry per qubit. In 1D,
    θ < 0.16π ≈ 0.5 corresponds to the MBL regime; beyond such
    critical value the dynamics become ergodic (i.e. thermal).
    Disorders are random on a qubit by qubit basis [1].

    Args:
        num_qubits: number of qubits (must be even).
        depth: two-qubit depth.
        barriers: if True adds barriers between layers.

    Notes:
        [1] Shtanko et.al. Uncovering Local Integrability in Quantum
            Many-Body Dynamics, https://arxiv.org/abs/2307.07552
    """

    def __init__(self, num_qubits: int, depth: int, *, barriers: bool = False) -> None:
        num_qubits = _validate_mbl_num_qubits(num_qubits)
        depth = _validate_mbl_depth(depth)
        barriers = bool(barriers)

        super().__init__(num_qubits, name=f"MBLChainEvolution<{num_qubits}, {depth}>")

        theta = Parameter("θ")
        phis = ParameterVector("φ", num_qubits)

        for layer in range(depth):
            layer_parity = layer % 2
            if barriers and layer > 0:
                self.barrier()
            for qubit in range(layer_parity, num_qubits - 1, 2):
                self.cz(qubit, qubit + 1)
                self.u(theta, 0, pi, qubit)
                self.u(theta, 0, pi, qubit + 1)
            for qubit in range(num_qubits):
                self.p(phis[qubit], qubit)


def _validate_mbl_num_qubits(num_qubits: int) -> int:
    """Validate number of qubits for MBL circuits."""
    if not isinstance(num_qubits, int):
        raise TypeError(f"Invalid num. qubits type {type(num_qubits)}, expected <int>.")
    if num_qubits <= 2:
        raise ValueError(f"Number of qubits ({num_qubits}) must be greater than two.")
    if num_qubits % 2:
        raise ValueError(f"Number of qubits ({num_qubits}) must be even.")
    return num_qubits


def _validate_mbl_depth(depth: int) -> int:
    """Validate depth for MBL circuits."""
    if not isinstance(depth, int):
        raise TypeError(f"Invalid depth type {type(depth)}, expected <int>.")
    if depth < 0:
        raise ValueError(f"Depth ({depth}) must be positive.")
    if depth % 2:
        raise ValueError(f"Depth ({depth}) must be even.")
    return depth
