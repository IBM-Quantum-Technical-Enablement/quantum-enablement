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

from __future__ import annotations

from collections.abc import Iterator
from functools import singledispatch
from itertools import product
from typing import Any

from numpy import allclose, angle, array, exp, eye, isclose, ndarray, pi
from numpy.random import default_rng
from numpy.typing import ArrayLike
from qiskit.circuit import Gate, Operation, QuantumRegister
from qiskit.circuit.library import PauliGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass

STANDARD_GATES = get_standard_gate_name_mapping()


################################################################################
## TWIRL CLASS
################################################################################
class PauliTwirl:
    """Pauli twirl.

    This class holds information about a Pauli twirl, independently
    of what operation it is later applied to. Therefore, applying
    the represented twirl to an arbitrary operation has no guaranty
    of preserving such operation's original action.

    Attributes:
        pre: Pauli gate to apply before the twirled operation.
        post: Pauli gate to apply after the twirled operation.
        phase: global phase induced by the twirling.
        num_qubits: number of qubits that the twirl applies to.

    Args:
        pre: Pauli gate to apply before the twirled operation.
        post: Pauli gate to apply after the twirled operation.
        phase: global phase induced by the twirling.

    Raises:
        TypeError: if pre or post are not Pauli gates or str.
        QiskitError: if pre or post are not valid Pauli str.
        TypeError: if phase is not float compatible.
        ValueError: if phase is not a valid float.
        ValueError: if pre and post operations don't apply to the same number of qubits.
    """

    __slots__ = ("_pre", "_post", "_phase")

    def __init__(self, pre: PauliGate | str, post: PauliGate | str, phase: float = 0.0) -> None:
        self._pre: PauliGate = self._validate_gate(pre)
        self._post: PauliGate = self._validate_gate(post)
        self._phase: float = float(phase)
        if self.pre.num_qubits != self.post.num_qubits:  # pylint: disable=no-member
            raise ValueError(
                "Twirling pre and post operations don't apply to the same number of qubits."
            )

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        pre = self.pre.params[0]  # pylint: disable=no-member
        post = self.post.params[0]  # pylint: disable=no-member
        phase = self.phase
        return f"{cls_name}({pre=}, {post=}, {phase=:.5f})"

    def __eq__(self, other: Any, /) -> bool:
        if not isinstance(other, type(self)):
            return False
        pre = self.pre == other.pre
        post = self.post == other.post
        phase = isclose(self.phase, other.phase)
        return pre and post and phase  # type: ignore

    ################################## PROPERTIES ##################################
    @property
    def pre(self) -> PauliGate:
        """Pre-operation in the twirl."""
        return self._pre

    @property
    def post(self) -> PauliGate:
        """Post-operation in the twirl."""
        return self._post

    @property
    def phase(self) -> float:
        """Global phase added by the twirl."""
        return self._phase

    @property
    def num_qubits(self) -> int:
        """Number of qubits that the twirl applies to."""
        return self.pre.num_qubits  # pylint: disable=no-member

    ################################## PUBLIC API ##################################
    def apply_to_node(self, node: DAGOpNode) -> DAGCircuit:
        """Apply twirl to input DAG operation node."""
        node = self._validate_node(node)
        dag = DAGCircuit()
        qubits = QuantumRegister(self.num_qubits)
        dag.add_qreg(qubits)
        dag.apply_operation_back(self.pre, qubits)
        dag.apply_operation_back(node.op, qubits)
        dag.apply_operation_back(self.post, qubits)
        dag.global_phase += self.phase
        return dag

    def apply_to_unitary(self, unitary: ArrayLike | Gate | str) -> ndarray:
        """Apply twirl to input unitary."""
        unitary = self._validate_unitary(unitary)
        pre = self.pre.to_matrix()  # pylint: disable=no-member
        post = self.post.to_matrix()  # pylint: disable=no-member
        phase_factor = exp(1j * self.phase)
        return (post @ unitary @ pre) * phase_factor

    def preserves_unitary(self, unitary: ArrayLike | Gate | str) -> bool:
        """Checks whether the twirl preserves the input unitary."""
        twirled = self.apply_to_unitary(unitary)
        unitary = self._validate_unitary(unitary)  # Note: revalidation, ArrayLike more expensive
        return allclose(twirled.conj().T @ unitary, eye(2**self.num_qubits))

    @classmethod
    def build_trivial(cls, num_qubits: int) -> PauliTwirl:
        """Build trivial twirl for a given number of qubits."""
        identity = "I" * int(num_qubits)
        return cls(pre=identity, post=identity, phase=0.0)

    def is_trivial(self) -> bool:
        """Check if twirl is the trivial twirl."""
        trivial = self.build_trivial(self.num_qubits)
        return self == trivial

    ################################## VALIDATION ##################################
    def _validate_gate(self, gate: PauliGate | str) -> PauliGate:
        # TODO: handle phase
        if isinstance(gate, PauliGate):
            return gate
        if isinstance(gate, str):
            return PauliGate(gate)
        raise TypeError(f"Invalid gate type {type(gate)}, expected <PauliGate>.")

    def _validate_node(self, node: DAGOpNode) -> DAGOpNode:
        """Validate node."""
        if not isinstance(node, DAGOpNode):
            raise TypeError(f"Invalid node type {type(node)}, expected <DAGOpNode>.")
        if node.op.num_qubits != self.num_qubits:
            raise ValueError(
                f"Number of qubits in input node ({node.op.num_qubits}) "
                f"does not match twirl's number of qubits ({self.num_qubits})."
            )
        return node

    def _validate_unitary(self, unitary: ArrayLike | Gate | str) -> ndarray:
        unitary = _validate_unitary(unitary)
        dimension = unitary.shape[0]
        num_qubits = dimension.bit_length() - 1  # Note: dimension == 2**num_qubits
        if num_qubits != self.num_qubits:
            raise ValueError(
                f"Number of qubits in input unitary ({num_qubits}) "
                f"does not match twirl's number of qubits ({self.num_qubits})."
            )
        return unitary


################################################################################
## COMPUTE TWIRLS
################################################################################
def generate_pauli_twirls(unitary: ArrayLike | Gate | str) -> Iterator[PauliTwirl]:
    """Generate operation-preserving Pauli twirls for input unitary.

    Args:
        unitary: the unitary to compute twirls for.

    Yields:
        Twirls preserving the unitary operation. Qubit order is given by the input.
    """
    if isinstance(unitary, str) and unitary in PAULI_TWIRLS:
        yield from PAULI_TWIRLS.get(unitary)
    elif isinstance(unitary, Gate) and unitary.name in PAULI_TWIRLS:
        yield from PAULI_TWIRLS.get(unitary.name)
    else:
        unitary = _validate_unitary(unitary)
        yield from _generate_pauli_twirls(unitary)


def _generate_pauli_twirls(unitary: ndarray) -> Iterator[PauliTwirl]:
    dimension = unitary.shape[0]
    num_qubits = dimension.bit_length() - 1  # Note: dimension == 2**num_qubits
    n_qubit_paulis = ("".join(pauli) for pauli in product("IXYZ", repeat=num_qubits))
    for pre, post in product(n_qubit_paulis, repeat=2):
        twirl = PauliTwirl(pre, post, phase=0.0)
        twirled = twirl.apply_to_unitary(unitary)  # TODO: avoid revalidation
        check = twirled.conj().T @ unitary
        phase_factor = check[0, 0]
        if not isclose(phase_factor, 0) and allclose(check / phase_factor, eye(dimension)):
            yield PauliTwirl(pre=pre, post=post, phase=angle(phase_factor))


################################################################################
## APPLY TWIRLS
################################################################################
class PauliTwirlPass(TransformationPass):
    """Pauli twirl gates in input circuit randomly.

    Both non-unitary and parametrized gates are not supported and will be skipped.

    Args:
        seed: seed for random number generator.
    """

    def __init__(self, *, seed: int | None = None):
        super().__init__()
        self._rng = default_rng(seed)

    def run(self, dag: DAGCircuit):
        """Pauli twirl target gates randomly for input DAGCircuit inplace."""
        target_nodes = (node for node in dag.op_nodes() if self._is_target_op(node.op))
        for node in target_nodes:
            twirl = self._get_random_twirl(node.op)
            twirl_dag = twirl.apply_to_node(node)
            dag.substitute_node_with_dag(node, twirl_dag)
        return dag

    def _is_target_op(self, op: Operation) -> bool:
        """Check whether operation should be twirled or not."""
        if not isinstance(op, Gate):
            return False  # Note: Skip non-gate nodes (e.g. barriers, measurements)
        if op.is_parameterized():
            return False  # Note: Skip parametrized gates
        return True

    def _get_random_twirl(self, gate: Gate) -> PauliTwirl:
        """Get random twirl for the input gate."""
        twirls = generate_pauli_twirls(gate)  # TODO: cache
        return self._rng.choice(list(twirls))  # type: ignore


class TwoQubitPauliTwirlPass(PauliTwirlPass):
    """Pauli twirl two-qubit gates in input circuit randomly.

    Both non-unitary and parametrized gates are not supported and will be skipped.

    Args:
        seed: seed for random number generator.
    """

    def _is_target_op(self, op: DAGOpNode) -> bool:
        if op.num_qubits != 2:
            return False  # Note: Only twirl two-qubit gates
        return super()._is_target_op(op)


################################################################################
## VALIDATION
################################################################################
@singledispatch
def _validate_unitary(unitary: ArrayLike | Gate | str) -> ndarray:
    unitary = array(unitary)
    if unitary.dtype.kind not in "buifc":
        raise TypeError("Invalid unitary type, expected numeric <ArrayLike | Gate | str>.")
    if unitary.ndim != 2:
        raise ValueError(f"Invalid unitary matrix with {unitary.ndim} axes, expected 2.")
    if unitary.shape[0] != unitary.shape[1]:
        raise ValueError(f"Invalid unitary matrix with non-square shape {unitary.shape}.")
    dimension = unitary.shape[0]
    if dimension & (dimension - 1) != 0:  # Note: integer power of two
        raise ValueError(f"Invalid unitary dimension ({dimension}), expected integer power of two.")
    if not allclose(unitary.conj().T @ unitary, eye(dimension)):
        raise ValueError("Invalid matrix, expected unitary.")
    return unitary


@_validate_unitary.register
def _(unitary: Gate) -> ndarray:
    if unitary.is_parameterized():
        raise ValueError("Twirls cannot be computed for parametrized gates.")
    return unitary.to_matrix()  # Note: Gates represent unitary matrices, avoid revalidation


@_validate_unitary.register
def _(unitary: str) -> ndarray:
    gate = STANDARD_GATES.get(unitary, None)
    if gate is None:
        raise ValueError(f"Unitary '{unitary}' not found in standard set.")
    return _validate_unitary(gate)


################################################################################
## TWIRL LIBRARY
################################################################################
CH_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IZ", post="IZ", phase=0.0),
    PauliTwirl(pre="YI", post="YZ", phase=0.0),
    PauliTwirl(pre="YZ", post="YI", phase=0.0),
)
CS_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IZ", post="IZ", phase=0.0),
    PauliTwirl(pre="ZI", post="ZI", phase=0.0),
    PauliTwirl(pre="ZZ", post="ZZ", phase=0.0),
)
CSDG_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IZ", post="IZ", phase=0.0),
    PauliTwirl(pre="ZI", post="ZI", phase=0.0),
    PauliTwirl(pre="ZZ", post="ZZ", phase=0.0),
)
CSX_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IZ", post="IZ", phase=0.0),
    PauliTwirl(pre="XI", post="XI", phase=0.0),
    PauliTwirl(pre="XZ", post="XZ", phase=0.0),
)
CX_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IX", post="XX", phase=0.0),
    PauliTwirl(pre="IY", post="XY", phase=0.0),
    PauliTwirl(pre="IZ", post="IZ", phase=0.0),
    PauliTwirl(pre="XI", post="XI", phase=0.0),
    PauliTwirl(pre="XX", post="IX", phase=0.0),
    PauliTwirl(pre="XY", post="IY", phase=0.0),
    PauliTwirl(pre="XZ", post="XZ", phase=0.0),
    PauliTwirl(pre="YI", post="YZ", phase=0.0),
    PauliTwirl(pre="YX", post="ZY", phase=0.0),
    PauliTwirl(pre="YY", post="ZX", phase=pi),
    PauliTwirl(pre="YZ", post="YI", phase=0.0),
    PauliTwirl(pre="ZI", post="ZZ", phase=0.0),
    PauliTwirl(pre="ZX", post="YY", phase=pi),
    PauliTwirl(pre="ZY", post="YX", phase=0.0),
    PauliTwirl(pre="ZZ", post="ZI", phase=0.0),
)
CY_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IX", post="YX", phase=0.0),
    PauliTwirl(pre="IY", post="YY", phase=0.0),
    PauliTwirl(pre="IZ", post="IZ", phase=0.0),
    PauliTwirl(pre="XI", post="XZ", phase=0.0),
    PauliTwirl(pre="XX", post="ZY", phase=pi),
    PauliTwirl(pre="XY", post="ZX", phase=0.0),
    PauliTwirl(pre="XZ", post="XI", phase=0.0),
    PauliTwirl(pre="YI", post="YI", phase=0.0),
    PauliTwirl(pre="YX", post="IX", phase=0.0),
    PauliTwirl(pre="YY", post="IY", phase=0.0),
    PauliTwirl(pre="YZ", post="YZ", phase=0.0),
    PauliTwirl(pre="ZI", post="ZZ", phase=0.0),
    PauliTwirl(pre="ZX", post="XY", phase=0.0),
    PauliTwirl(pre="ZY", post="XX", phase=pi),
    PauliTwirl(pre="ZZ", post="ZI", phase=0.0),
)
CZ_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IX", post="ZX", phase=0.0),
    PauliTwirl(pre="IY", post="ZY", phase=0.0),
    PauliTwirl(pre="IZ", post="IZ", phase=0.0),
    PauliTwirl(pre="XI", post="XZ", phase=0.0),
    PauliTwirl(pre="XX", post="YY", phase=0.0),
    PauliTwirl(pre="XY", post="YX", phase=pi),
    PauliTwirl(pre="XZ", post="XI", phase=0.0),
    PauliTwirl(pre="YI", post="YZ", phase=0.0),
    PauliTwirl(pre="YX", post="XY", phase=pi),
    PauliTwirl(pre="YY", post="XX", phase=0.0),
    PauliTwirl(pre="YZ", post="YI", phase=0.0),
    PauliTwirl(pre="ZI", post="ZI", phase=0.0),
    PauliTwirl(pre="ZX", post="IX", phase=0.0),
    PauliTwirl(pre="ZY", post="IY", phase=0.0),
    PauliTwirl(pre="ZZ", post="ZZ", phase=0.0),
)
DCX_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IX", post="XI", phase=0.0),
    PauliTwirl(pre="IY", post="YZ", phase=0.0),
    PauliTwirl(pre="IZ", post="ZZ", phase=0.0),
    PauliTwirl(pre="XI", post="XX", phase=0.0),
    PauliTwirl(pre="XX", post="IX", phase=0.0),
    PauliTwirl(pre="XY", post="ZY", phase=0.0),
    PauliTwirl(pre="XZ", post="YY", phase=pi),
    PauliTwirl(pre="YI", post="XY", phase=0.0),
    PauliTwirl(pre="YX", post="IY", phase=0.0),
    PauliTwirl(pre="YY", post="ZX", phase=pi),
    PauliTwirl(pre="YZ", post="YX", phase=0.0),
    PauliTwirl(pre="ZI", post="IZ", phase=0.0),
    PauliTwirl(pre="ZX", post="XZ", phase=0.0),
    PauliTwirl(pre="ZY", post="YI", phase=0.0),
    PauliTwirl(pre="ZZ", post="ZI", phase=0.0),
)
ECR_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IX", post="XY", phase=pi),
    PauliTwirl(pre="IY", post="XX", phase=pi),
    PauliTwirl(pre="IZ", post="IZ", phase=pi),
    PauliTwirl(pre="XI", post="XI", phase=0.0),
    PauliTwirl(pre="XX", post="IY", phase=pi),
    PauliTwirl(pre="XY", post="IX", phase=pi),
    PauliTwirl(pre="XZ", post="XZ", phase=pi),
    PauliTwirl(pre="YI", post="ZZ", phase=pi),
    PauliTwirl(pre="YX", post="YX", phase=0.0),
    PauliTwirl(pre="YY", post="YY", phase=pi),
    PauliTwirl(pre="YZ", post="ZI", phase=0.0),
    PauliTwirl(pre="ZI", post="YZ", phase=0.0),
    PauliTwirl(pre="ZX", post="ZX", phase=0.0),
    PauliTwirl(pre="ZY", post="ZY", phase=pi),
    PauliTwirl(pre="ZZ", post="YI", phase=pi),
)
ISWAP_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IX", post="YZ", phase=0.0),
    PauliTwirl(pre="IY", post="XZ", phase=pi),
    PauliTwirl(pre="IZ", post="ZI", phase=0.0),
    PauliTwirl(pre="XI", post="ZY", phase=0.0),
    PauliTwirl(pre="XX", post="XX", phase=0.0),
    PauliTwirl(pre="XY", post="YX", phase=0.0),
    PauliTwirl(pre="XZ", post="IY", phase=0.0),
    PauliTwirl(pre="YI", post="ZX", phase=pi),
    PauliTwirl(pre="YX", post="XY", phase=0.0),
    PauliTwirl(pre="YY", post="YY", phase=0.0),
    PauliTwirl(pre="YZ", post="IX", phase=pi),
    PauliTwirl(pre="ZI", post="IZ", phase=0.0),
    PauliTwirl(pre="ZX", post="YI", phase=0.0),
    PauliTwirl(pre="ZY", post="XI", phase=pi),
    PauliTwirl(pre="ZZ", post="ZZ", phase=0.0),
)
SWAP_PAULI_TWIRLS = (
    PauliTwirl(pre="II", post="II", phase=0.0),
    PauliTwirl(pre="IX", post="XI", phase=0.0),
    PauliTwirl(pre="IY", post="YI", phase=0.0),
    PauliTwirl(pre="IZ", post="ZI", phase=0.0),
    PauliTwirl(pre="XI", post="IX", phase=0.0),
    PauliTwirl(pre="XX", post="XX", phase=0.0),
    PauliTwirl(pre="XY", post="YX", phase=0.0),
    PauliTwirl(pre="XZ", post="ZX", phase=0.0),
    PauliTwirl(pre="YI", post="IY", phase=0.0),
    PauliTwirl(pre="YX", post="XY", phase=0.0),
    PauliTwirl(pre="YY", post="YY", phase=0.0),
    PauliTwirl(pre="YZ", post="ZY", phase=0.0),
    PauliTwirl(pre="ZI", post="IZ", phase=0.0),
    PauliTwirl(pre="ZX", post="XZ", phase=0.0),
    PauliTwirl(pre="ZY", post="YZ", phase=0.0),
    PauliTwirl(pre="ZZ", post="ZZ", phase=0.0),
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
