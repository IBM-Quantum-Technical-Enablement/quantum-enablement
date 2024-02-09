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

"""Twirling tools and techniques."""

from collections import namedtuple
from collections.abc import Iterable, Iterator
from functools import singledispatch
from itertools import product

from numpy import allclose, angle, eye, kron, matrix, pi
from numpy.random import default_rng
from numpy.typing import ArrayLike, NDArray
from qiskit.circuit import Gate, QuantumRegister
from qiskit.circuit.library import PauliGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass

Twirl = namedtuple("Twirl", ["pre", "post", "phase"])

################################################################################
## APPLY TWIRLS
################################################################################
STANDARD_GATES = get_standard_gate_name_mapping()
TWO_QUBIT_GATES = {name: gate for name, gate in STANDARD_GATES.items() if gate.num_qubits == 2}
UNPARAMETERIZED_TWO_QUBIT_GATES = {
    name: gate for name, gate in TWO_QUBIT_GATES.items() if not gate.is_parameterized()
}


class TwoQubitPauliTwirl(TransformationPass):
    """Pauli twirl two-qubit gates in input circuit randomly.

    Args:
        target_gates: names of gates to twirl. If None, all unparameterized
            standard two-qubit gates will be twirled.
        seed: seed for random number generator.

    Notes:
        Adapted from the reference:
        https://quantum-enablement.org/posts/2023/2023-02-02-pauli_twirling.html
    """

    def __init__(self, target_gates: list[str] | None = None, *, seed: int | None = None):
        super().__init__()
        target_gates = self._validate_target_gates(target_gates)
        self._gate_twirls = {gate: tuple(generate_pauli_twirls(gate)) for gate in target_gates}
        self._rng = default_rng(seed)

    def _validate_target_gates(self, target_gates: list[str] | None) -> list[str]:
        if target_gates is None:
            return list(UNPARAMETERIZED_TWO_QUBIT_GATES.keys())
        if not isinstance(target_gates, Iterable) or not all(
            isinstance(gate, str) for gate in target_gates
        ):
            raise TypeError("Invalid input target gates, expected list of gate names (str).")
        return target_gates

    def run(self, dag: DAGCircuit):
        target_gates = self._gate_twirls.keys()
        target_nodes = (node for run in dag.collect_runs(target_gates) for node in run)
        for node in target_nodes:
            twirl_dag = self._build_random_twirl_dag(node)
            dag.substitute_node_with_dag(node, twirl_dag)
        return dag

    def _build_random_twirl_dag(self, node) -> DAGCircuit:
        """Build random twirl dag for a given node."""
        twirl = self._get_random_twirl(node.op.name)
        dag = DAGCircuit()
        dag.add_qreg(qubits := QuantumRegister(2))
        dag.global_phase += twirl.phase
        dag.apply_operation_back(PauliGate(twirl.pre), [qubits[0], qubits[1]])
        dag.apply_operation_back(node.op, [qubits[0], qubits[1]])
        dag.apply_operation_back(PauliGate(twirl.post), [qubits[0], qubits[1]])
        return dag

    def _get_random_twirl(self, gate: str) -> Twirl:
        """Get random twirl for the input gate."""
        twirls = self._gate_twirls.get(gate)
        choice = self._rng.choice(len(twirls))
        return twirls[choice]


################################################################################
## COMPUTE TWIRLS
################################################################################
PAULI_MATRICES = {
    "I": matrix([[1, 0], [0, 1]]),
    "X": matrix([[0, 1], [1, 0]]),
    "Y": matrix([[0, -1j], [1j, 0]]),
    "Z": matrix([[1, 0], [0, -1]]),
}
TWO_QUBIT_PAULI_MATRICES = {
    f"{p1}{p0}": kron(PAULI_MATRICES[p1], PAULI_MATRICES[p0])
    for p0, p1 in product(PAULI_MATRICES.keys(), repeat=2)
}


@singledispatch
def generate_pauli_twirls(unitary: ArrayLike | Gate | str) -> Iterator[Twirl]:
    """Generate Pauli twirls for input two-qubit unitary.

    Args:
        unitary: the two-qubit unitary to compute twirls for.

    Yields:
        Tuples of possible twirls. These are themselves composed of the
        Pauli gates to apply before (pre) and after (post) the unitary,
        and an associated global phase. Such gates are represented by
        Pauli strings, where qubit order is given by the input.

    Notes:
        Adapted from the reference:
        https://quantum-enablement.org/posts/2023/2023-02-02-pauli_twirling.html
    """
    unitary = _validate_two_qubit_unitary(unitary)
    for pre, post in product(TWO_QUBIT_PAULI_MATRICES, repeat=2):
        twirled = TWO_QUBIT_PAULI_MATRICES[post] * unitary * TWO_QUBIT_PAULI_MATRICES[pre]
        check = twirled.H * unitary
        phase_factor = check[0, 0]
        if phase_factor != 0 and allclose(check / phase_factor, eye(4)):
            yield Twirl(pre=pre, post=post, phase=angle(phase_factor))


@generate_pauli_twirls.register(Gate)
def _(unitary):
    if unitary.name in PAULI_TWIRLS:
        yield from PAULI_TWIRLS.get(unitary.name)
    unitary = unitary.to_matrix()
    yield from generate_pauli_twirls(unitary)


@generate_pauli_twirls.register(str)
def _(unitary):
    unitary = STANDARD_GATES.get(unitary, None)
    if unitary is None:
        raise ValueError(f"Unitary '{unitary}' not found in standard set.")
    yield from generate_pauli_twirls(unitary)


def _validate_two_qubit_unitary(unitary: ArrayLike) -> NDArray:
    try:
        unitary = matrix(unitary)
    except ValueError as error:
        raise TypeError("Invalid unitary provided.") from error
    if unitary.shape != (4, 4):
        raise ValueError(f"Invalid unitary with shape {unitary.shape}, expected (4, 4) instead.")
    if not allclose(unitary.H * unitary, eye(4)):
        raise ValueError("Non-unitary matrix provided.")
    return unitary


################################################################################
## TWIRL LIBRARY
################################################################################
CH_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="YI", post="YZ", phase=0.0),
    Twirl(pre="IZ", post="IZ", phase=0.0),
    Twirl(pre="YZ", post="YI", phase=0.0),
)
CS_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="ZI", post="ZI", phase=0.0),
    Twirl(pre="IZ", post="IZ", phase=0.0),
    Twirl(pre="ZZ", post="ZZ", phase=0.0),
)
CSDG_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="ZI", post="ZI", phase=0.0),
    Twirl(pre="IZ", post="IZ", phase=0.0),
    Twirl(pre="ZZ", post="ZZ", phase=0.0),
)
CSX_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="XI", post="XI", phase=0.0),
    Twirl(pre="IZ", post="IZ", phase=0.0),
    Twirl(pre="XZ", post="XZ", phase=0.0),
)
CX_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="IX", post="XX", phase=0.0),
    Twirl(pre="IY", post="XY", phase=0.0),
    Twirl(pre="IZ", post="IZ", phase=0.0),
    Twirl(pre="XI", post="XI", phase=0.0),
    Twirl(pre="XX", post="IX", phase=0.0),
    Twirl(pre="XY", post="IY", phase=0.0),
    Twirl(pre="XZ", post="XZ", phase=0.0),
    Twirl(pre="YI", post="YZ", phase=0.0),
    Twirl(pre="YX", post="ZY", phase=0.0),
    Twirl(pre="YY", post="ZX", phase=pi),
    Twirl(pre="YZ", post="YI", phase=0.0),
    Twirl(pre="ZI", post="ZZ", phase=0.0),
    Twirl(pre="ZX", post="YY", phase=pi),
    Twirl(pre="ZY", post="YX", phase=0.0),
    Twirl(pre="ZZ", post="ZI", phase=0.0),
)
CY_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="XI", post="XZ", phase=0.0),
    Twirl(pre="YI", post="YI", phase=0.0),
    Twirl(pre="ZI", post="ZZ", phase=0.0),
    Twirl(pre="IX", post="YX", phase=0.0),
    Twirl(pre="XX", post="ZY", phase=pi),
    Twirl(pre="YX", post="IX", phase=0.0),
    Twirl(pre="ZX", post="XY", phase=0.0),
    Twirl(pre="IY", post="YY", phase=0.0),
    Twirl(pre="XY", post="ZX", phase=0.0),
    Twirl(pre="YY", post="IY", phase=0.0),
    Twirl(pre="ZY", post="XX", phase=pi),
    Twirl(pre="IZ", post="IZ", phase=0.0),
    Twirl(pre="XZ", post="XI", phase=0.0),
    Twirl(pre="YZ", post="YZ", phase=0.0),
    Twirl(pre="ZZ", post="ZI", phase=0.0),
)
CZ_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="IX", post="ZX", phase=0.0),
    Twirl(pre="IY", post="ZY", phase=0.0),
    Twirl(pre="IZ", post="IZ", phase=0.0),
    Twirl(pre="XI", post="XZ", phase=0.0),
    Twirl(pre="XX", post="YY", phase=0.0),
    Twirl(pre="XY", post="YX", phase=pi),
    Twirl(pre="XZ", post="XI", phase=0.0),
    Twirl(pre="YI", post="YZ", phase=0.0),
    Twirl(pre="YX", post="XY", phase=pi),
    Twirl(pre="YY", post="XX", phase=0.0),
    Twirl(pre="YZ", post="YI", phase=0.0),
    Twirl(pre="ZI", post="ZI", phase=0.0),
    Twirl(pre="ZX", post="IX", phase=0.0),
    Twirl(pre="ZY", post="IY", phase=0.0),
    Twirl(pre="ZZ", post="ZZ", phase=0.0),
)
DCX_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="XI", post="XX", phase=0.0),
    Twirl(pre="YI", post="XY", phase=0.0),
    Twirl(pre="ZI", post="IZ", phase=0.0),
    Twirl(pre="IX", post="XI", phase=0.0),
    Twirl(pre="XX", post="IX", phase=0.0),
    Twirl(pre="YX", post="IY", phase=0.0),
    Twirl(pre="ZX", post="XZ", phase=0.0),
    Twirl(pre="IY", post="YZ", phase=0.0),
    Twirl(pre="XY", post="ZY", phase=0.0),
    Twirl(pre="YY", post="ZX", phase=pi),
    Twirl(pre="ZY", post="YI", phase=0.0),
    Twirl(pre="IZ", post="ZZ", phase=0.0),
    Twirl(pre="XZ", post="YY", phase=pi),
    Twirl(pre="YZ", post="YX", phase=0.0),
    Twirl(pre="ZZ", post="ZI", phase=0.0),
)
ECR_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="IX", post="XY", phase=pi),
    Twirl(pre="IY", post="XX", phase=pi),
    Twirl(pre="IZ", post="IZ", phase=pi),
    Twirl(pre="XI", post="XI", phase=0.0),
    Twirl(pre="XX", post="IY", phase=pi),
    Twirl(pre="XY", post="IX", phase=pi),
    Twirl(pre="XZ", post="XZ", phase=pi),
    Twirl(pre="YI", post="ZZ", phase=pi),
    Twirl(pre="YX", post="YX", phase=0.0),
    Twirl(pre="YY", post="YY", phase=pi),
    Twirl(pre="YZ", post="ZI", phase=0.0),
    Twirl(pre="ZI", post="YZ", phase=0.0),
    Twirl(pre="ZX", post="ZX", phase=0.0),
    Twirl(pre="ZY", post="ZY", phase=pi),
    Twirl(pre="ZZ", post="YI", phase=pi),
)
ISWAP_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="IX", post="YZ", phase=0.0),
    Twirl(pre="IY", post="XZ", phase=pi),
    Twirl(pre="IZ", post="ZI", phase=0.0),
    Twirl(pre="XI", post="ZY", phase=0.0),
    Twirl(pre="XX", post="XX", phase=0.0),
    Twirl(pre="XY", post="YX", phase=0.0),
    Twirl(pre="XZ", post="IY", phase=0.0),
    Twirl(pre="YI", post="ZX", phase=pi),
    Twirl(pre="YX", post="XY", phase=0.0),
    Twirl(pre="YY", post="YY", phase=0.0),
    Twirl(pre="YZ", post="IX", phase=pi),
    Twirl(pre="ZI", post="IZ", phase=0.0),
    Twirl(pre="ZX", post="YI", phase=0.0),
    Twirl(pre="ZY", post="XI", phase=pi),
    Twirl(pre="ZZ", post="ZZ", phase=0.0),
)
SWAP_PAULI_TWIRLS = (
    Twirl(pre="II", post="II", phase=0.0),
    Twirl(pre="XI", post="IX", phase=0.0),
    Twirl(pre="YI", post="IY", phase=0.0),
    Twirl(pre="ZI", post="IZ", phase=0.0),
    Twirl(pre="IX", post="XI", phase=0.0),
    Twirl(pre="XX", post="XX", phase=0.0),
    Twirl(pre="YX", post="XY", phase=0.0),
    Twirl(pre="ZX", post="XZ", phase=0.0),
    Twirl(pre="IY", post="YI", phase=0.0),
    Twirl(pre="XY", post="YX", phase=0.0),
    Twirl(pre="YY", post="YY", phase=0.0),
    Twirl(pre="ZY", post="YZ", phase=0.0),
    Twirl(pre="IZ", post="ZI", phase=0.0),
    Twirl(pre="XZ", post="ZX", phase=0.0),
    Twirl(pre="YZ", post="ZY", phase=0.0),
    Twirl(pre="ZZ", post="ZZ", phase=0.0),
)

PAULI_TWIRLS = {
    "ch": CH_PAULI_TWIRLS,
    "cs": CS_PAULI_TWIRLS,
    "csdg": CSDG_PAULI_TWIRLS,
    "csx": CSX_PAULI_TWIRLS,
    "cx": CX_PAULI_TWIRLS,
    "cy": CY_PAULI_TWIRLS,
    "cz": CZ_PAULI_TWIRLS,
    "dcx": DCX_PAULI_TWIRLS,
    "ecr": ECR_PAULI_TWIRLS,
    "iswap": ISWAP_PAULI_TWIRLS,
    "swap": SWAP_PAULI_TWIRLS,
}
