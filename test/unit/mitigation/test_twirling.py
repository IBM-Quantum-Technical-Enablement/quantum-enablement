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

"""Test twirling module."""

from copy import deepcopy

from numpy import allclose, angle, exp, eye, isclose, pi
from pytest import mark, raises  # pylint: disable=import-error
from qiskit.circuit import Gate, Parameter, QuantumCircuit
from qiskit.circuit.library import PauliGate, RXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.exceptions import QiskitError

from quantum_enablement.mitigation._twirling import (
    PAULI_TWIRLS,
    STANDARD_GATES,
    PauliTwirl,
    TwoQubitPauliTwirlPass,
    generate_pauli_twirls,
)


################################################################################
## CASES
################################################################################
def generate_test_pauli_twirls():
    """Generate test twirls."""
    yield ("I", "I", 0.0)
    yield ("X", "Y", 0.5)
    yield ("Y", "Z", 1.0)
    yield ("II", "II", 0.0)
    yield ("XY", "YZ", 0.5)
    yield ("YI", "ZX", 1.0)
    yield ("III", "III", 0.0)
    yield ("XYZ", "YZX", 0.5)
    yield ("YIZ", "ZXI", 1.0)


def generate_test_dags():
    """Generate test DAG circuits."""
    circuit = QuantumCircuit(1)
    yield circuit_to_dag(circuit)

    circuit.id(0)
    yield circuit_to_dag(circuit)

    circuit.x(0)
    yield circuit_to_dag(circuit)

    circuit.h(0)
    yield circuit_to_dag(circuit)

    circuit.measure_all()
    yield circuit_to_dag(circuit)

    circuit = QuantumCircuit(2)
    yield circuit_to_dag(circuit)

    circuit.h(0)
    yield circuit_to_dag(circuit)

    circuit.cx(0, 1)
    yield circuit_to_dag(circuit)

    circuit.x(1)
    yield circuit_to_dag(circuit)

    circuit.measure_all()
    yield circuit_to_dag(circuit)

    circuit = QuantumCircuit(3)
    yield circuit_to_dag(circuit)

    circuit.h(0)
    yield circuit_to_dag(circuit)

    circuit.cx(0, 1)
    yield circuit_to_dag(circuit)

    circuit.cx(1, 2)
    yield circuit_to_dag(circuit)

    circuit.measure_all()
    yield circuit_to_dag(circuit)


################################################################################
## TESTS
################################################################################
class TestPauliTwirl:
    """Test suite for the PauliTwirl class."""

    @mark.parametrize("pre, post, phase", generate_test_pauli_twirls())
    def test_creation(self, pre, post, phase):
        """Test PauliTwirl creation."""
        twirl1 = PauliTwirl(pre, post, phase)
        twirl2 = PauliTwirl(PauliGate(pre), PauliGate(post), phase)
        assert twirl1.pre == twirl2.pre == PauliGate(pre), "Wrong pre Pauli gate."
        assert twirl1.post == twirl2.post == PauliGate(post), "Wrong post Pauli gate."
        assert twirl1.phase == twirl2.phase == float(phase), "Wrong phase."

    def test_validate_gate(self):
        """Test PauliTwirl gate validation."""
        error_message = r"Invalid gate type .*, expected <PauliGate>\."
        with raises(TypeError, match=error_message):
            PauliTwirl(None, "I", 0.0)
        with raises(TypeError, match=error_message):
            PauliTwirl("I", None, 0.0)
        with raises(QiskitError):
            PauliTwirl("H", "I", 0.0)
        with raises(QiskitError):
            PauliTwirl("I", "H", 0.0)

    def test_incompatible_operations(self):
        """Test PauliTwirl incompatible Paulis."""
        error_message = r"Twirling pre and post operations "
        error_message += r"don't apply to the same number of qubits\."
        with raises(ValueError, match=error_message):
            PauliTwirl("I", "II", 0.0)

    def test_defaults(self):
        """Test PauliTwirl defaults."""
        twirl = PauliTwirl("I", "I")
        assert twirl.phase == 0.0, "Wrong default phase."

    @mark.parametrize("pre, post, phase", generate_test_pauli_twirls())
    def test_equal(self, pre, post, phase):
        """Test PauliTwirl equality."""
        twirl1 = PauliTwirl(pre, post, phase)
        twirl2 = PauliTwirl(pre, post, phase)
        assert twirl1 is not twirl2, "PauliTwirls should not be the same object."
        assert twirl1 == twirl2, "PauliTwirls should be equal."
        assert twirl1 != "PauliTwirl", "PauliTwirls should not be equal to other types."

    @mark.parametrize("pre, post, phase", generate_test_pauli_twirls())
    def test_repr(self, pre, post, phase):
        """Test PauliTwirl repr."""
        twirl = PauliTwirl(pre, post, phase)
        assert repr(twirl) == f"PauliTwirl({pre=}, {post=}, {phase=:.5f})", "Wrong repr."

    def test_set_pre(self):
        """Test PauliTwirl set pre."""
        twirl = PauliTwirl("I", "I", 0.0)
        error_message = r"property 'pre' of 'PauliTwirl' object has no setter"
        with raises(AttributeError, match=error_message):
            twirl.pre = "I"

    def test_set_post(self):
        """Test PauliTwirl set post."""
        twirl = PauliTwirl("I", "I", 0.0)
        error_message = r"property 'post' of 'PauliTwirl' object has no setter"
        with raises(AttributeError, match=error_message):
            twirl.post = "I"

    def test_set_phase(self):
        """Test PauliTwirl set phase."""
        twirl = PauliTwirl("I", "I", 0.0)
        error_message = r"property 'phase' of 'PauliTwirl' object has no setter"
        with raises(AttributeError, match=error_message):
            twirl.phase = 0.0

    @mark.parametrize("pre, post, phase", generate_test_pauli_twirls())
    def test_num_qubits(self, pre, post, phase):
        """Test PauliTwirl num_qubits."""
        twirl = PauliTwirl(pre, post, phase)
        assert twirl.num_qubits == len(pre), "Wrong number of qubits."
        error_message = r"property 'num_qubits' of 'PauliTwirl' object has no setter"
        with raises(AttributeError, match=error_message):
            twirl.num_qubits = 0

    @mark.parametrize("num_qubits", [-1, 0, 1, 2, 3, 4])
    def test_build_trivial(self, num_qubits):
        """Test PauliTwirl build_trivial."""
        identity = "I" * num_qubits
        twirl = PauliTwirl(identity, identity, 0.0)
        assert twirl.build_trivial(num_qubits) == twirl, "Wrong trivial Pauli twirl."

    @mark.parametrize("num_qubits", [-1, 0, 1, 2, 3, 4])
    def test_is_trivial(self, num_qubits):
        """Test PauliTwirl is_trivial."""
        identity = "I" * num_qubits
        twirl = PauliTwirl(identity, identity, 0.0)
        assert twirl.is_trivial(), "Wrong triviality."

    @mark.parametrize("pre, post, phase", generate_test_pauli_twirls())
    def test_apply_to_node(self, pre, post, phase):
        """Test PauliTwirl apply_to_node."""
        twirl = PauliTwirl(pre, post, phase)
        operation = PauliGate("I" * len(pre))  # Note: dummy gate.
        node = DAGOpNode(operation)
        twirled_dag = twirl.apply_to_node(node)
        op_nodes = twirled_dag.op_nodes()
        assert len(op_nodes) == 3, "Wrong number of op. nodes."
        assert op_nodes[0].op == PauliGate(pre), "Wrong pre operation."
        assert op_nodes[1].op == node.op, "Wrong twirled operation."
        assert op_nodes[2].op == PauliGate(post), "Wrong post operation."
        assert op_nodes[0].qargs == op_nodes[1].qargs == op_nodes[2].qargs, "Wrong qargs."
        assert op_nodes[0].cargs == op_nodes[1].cargs == op_nodes[2].cargs, "Wrong cargs."

    def test_validate_node(self):
        """Test PauliTwirl node validation."""
        with raises(TypeError, match=r"Invalid node type .*, expected <DAGOpNode>\."):
            PauliTwirl("I", "I", 0.0).apply_to_node(None)

        error_message = r"Number of qubits in input node \(1\) "
        error_message += r"does not match twirl's number of qubits \(2\)\."
        with raises(ValueError, match=error_message):
            operation = PauliGate("I")
            PauliTwirl("II", "II", 0.0).apply_to_node(DAGOpNode(operation))

    @mark.parametrize("pre, post, phase", generate_test_pauli_twirls())
    def test_apply_to_unitary(self, pre, post, phase):
        """Test PauliTwirl apply_to_unitary."""
        twirl = PauliTwirl(pre, post, phase)
        dimension = 2 ** len(pre)
        unitary = eye(dimension)  # Note: dummy unitary.
        twirled_unitary = twirl.apply_to_unitary(unitary)
        assert twirled_unitary.shape == (dimension, dimension), "Wrong twirled unitary shape."
        pre = PauliGate(pre).to_matrix()
        post = PauliGate(post).to_matrix()
        phase_factor = exp(1j * phase)
        expected_unitary = (post @ unitary @ pre) * phase_factor
        assert allclose(twirled_unitary, expected_unitary), "Wrong twirled unitary."

    @mark.parametrize("pre, post, phase", generate_test_pauli_twirls())
    def test_preserves_unitary(self, pre, post, phase):
        """Test PauliTwirl preserves_unitary."""
        twirl = PauliTwirl(pre, post, phase)
        dimension = 2 ** len(pre)
        unitary = eye(dimension)  # Note: dummy unitary.
        pre = PauliGate(pre).to_matrix()
        post = PauliGate(post).to_matrix()
        phase_factor = exp(1j * phase)
        expected_unitary = (post @ unitary @ pre) * phase_factor
        if allclose(expected_unitary, eye(dimension)):
            assert twirl.preserves_unitary(unitary), "Twirl should preserve unitary."
        else:
            assert not twirl.preserves_unitary(unitary), "Twirl should not preserve unitary."

    @mark.parametrize(
        "gate, twirl", ((gate, twirl) for gate, twirls in PAULI_TWIRLS.items() for twirl in twirls)
    )
    def test_pre_computed_pauli_twirls(self, gate, twirl):
        """Test pre-computed PAULI_TWIRLS."""
        assert twirl.preserves_unitary(gate), "Twirl should preserve <str> unitary."
        gate = STANDARD_GATES[gate]
        assert twirl.preserves_unitary(gate), "Twirl should preserve <Gate> unitary."
        gate = gate.to_matrix()
        assert twirl.preserves_unitary(gate), "Twirl should preserve <ndarray> unitary."

    def test_validate_unitary(self):
        """Test PauliTwirl unitary validation."""
        error_message = r"Number of qubits in input unitary \(2\) "
        error_message += r"does not match twirl's number of qubits \(1\)\."
        with raises(ValueError, match=error_message):
            PauliTwirl("I", "I", 0.0).apply_to_unitary(eye(4))

        error_message = r"Invalid unitary type, expected numeric <ArrayLike \| Gate \| str>."
        with raises(TypeError, match=error_message):
            PauliTwirl("I", "I", 0.0).apply_to_unitary(None)

        error_message = r"Invalid unitary matrix with 0 axes, expected 2\."
        with raises(ValueError, match=error_message):
            PauliTwirl("I", "I", 0.0).apply_to_unitary(0)

        error_message = r"Invalid unitary matrix with non-square shape \(2, 3\)\."
        with raises(ValueError, match=error_message):
            PauliTwirl("I", "I", 0.0).apply_to_unitary([[0, 0, 0], [0, 0, 0]])

        error_message = r"Invalid unitary dimension \(3\), expected integer power of two\."
        with raises(ValueError, match=error_message):
            PauliTwirl("I", "I", 0.0).apply_to_unitary([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        error_message = r"Invalid matrix, expected unitary\."
        with raises(ValueError, match=error_message):
            PauliTwirl("I", "I", 0.0).apply_to_unitary([[0, 0], [0, 0]])

        error_message = r"Twirls cannot be computed for parametrized gates\."
        with raises(ValueError, match=error_message):
            gate = RXGate(Parameter("theta"))
            PauliTwirl("I", "I", 0.0).apply_to_unitary(gate)

        error_message = r"Unitary '.*' not found in standard set\."
        with raises(ValueError, match=error_message):
            PauliTwirl("I", "I", 0.0).apply_to_unitary("unitary")


class TestGeneratePauliTwirls:
    """Test suite for generate_pauli_twirls."""

    @mark.parametrize("gate, twirls", PAULI_TWIRLS.items())
    def test_generate(self, gate, twirls):
        """Test generate_pauli_twirls."""
        gate = STANDARD_GATES[gate].to_matrix()
        for twirl, expected in zip(generate_pauli_twirls(gate), twirls):
            assert twirl is not expected, "PauliTwirls should not be the same object."
            assert twirl == expected, "Wrong twirl."
            assert twirl.preserves_unitary(gate), "Twirl should preserve unitary."

    @mark.parametrize("gate, twirls", PAULI_TWIRLS.items())
    def test_cache(self, gate, twirls):
        """Test generate_pauli_twirls cache."""
        for twirl, expected in zip(generate_pauli_twirls(gate), twirls):
            assert twirl is expected, "PauliTwirls should be the same object."

        gate = STANDARD_GATES[gate]
        for twirl, expected in zip(generate_pauli_twirls(gate), twirls):
            assert twirl is expected, "PauliTwirls should be the same object."

    def test_non_cache(self):
        """Test generate_pauli_twirls not in the cache."""
        gate = "id"
        twirls = [PauliTwirl(pauli, pauli) for pauli in "IXYZ"]
        for twirl, expected in zip(generate_pauli_twirls(gate), twirls):
            assert twirl is not expected, "PauliTwirls should not be the same object."
            assert twirl == expected, "Wrong twirl."
            assert twirl.preserves_unitary(gate), "Twirl should preserve unitary."

        gate = STANDARD_GATES[gate]
        for twirl, expected in zip(generate_pauli_twirls(gate), twirls):
            assert twirl is not expected, "PauliTwirls should not be the same object."
            assert twirl == expected, "Wrong twirl."
            assert twirl.preserves_unitary(gate), "Twirl should preserve unitary."


class TestTwoQubitPauliTwirlPass:
    """Test suite for the TwoQubitPauliTwirlPass class."""

    @mark.parametrize("dag", generate_test_dags())
    def test_run(self, dag):  # pylint: disable=too-many-locals
        """Test TwoQubitPauliTwirlPass run."""
        original_dag = deepcopy(dag)
        twirled_dag = TwoQubitPauliTwirlPass(seed=0).run(dag)
        assert twirled_dag is dag, "Twirled DAG should be the same object."

        twirled_nodes = iter(twirled_dag.topological_op_nodes())
        twirl_phase = 0.0  # Note: to check new global phase.
        for node in original_dag.topological_op_nodes():
            if isinstance(node.op, Gate) and node.op.num_qubits == 2:

                # Check that the twirling was performed.
                pre = next(twirled_nodes)
                twirled = next(twirled_nodes)
                post = next(twirled_nodes)
                assert (
                    isinstance(pre.op, PauliGate)
                    and twirled.op == node.op
                    and isinstance(post.op, PauliGate)
                    and pre.qargs == post.qargs == twirled.qargs == node.qargs
                    and pre.cargs == post.cargs == twirled.cargs == node.cargs
                ), "Two-qubit operations should be twirled."

                # Check that the unitary was preserved up to a global phase.
                pre = pre.op.to_matrix()
                original = twirled.op.to_matrix()
                post = post.op.to_matrix()
                twirled = post @ original @ pre
                check = twirled.conj().T @ original
                phase_factor = check[0, 0]
                assert allclose(
                    check / phase_factor, eye(4)
                ), "Twirled unitary should be preserved up to a global phase."
                twirl_phase += angle(phase_factor)

            else:
                twirled = next(twirled_nodes)
                assert (
                    twirled.op == node.op
                    and twirled.qargs == node.qargs
                    and twirled.cargs == node.cargs
                ), "Non-two-qubit operations should be preserved."

        # Check that the global phase was updated.
        original_circuit = dag_to_circuit(original_dag)
        twirled_circuit = dag_to_circuit(twirled_dag)
        phase_diff = twirled_circuit.global_phase - original_circuit.global_phase
        check = (phase_diff - twirl_phase) % (2 * pi)
        assert isclose(check, 0.0), "Global phase should be updated."

    def test_parametrized(self):
        """Test TwoQubitPauliTwirlPass with parametrized gates."""
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.crx(theta, 0, 1)
        dag = circuit_to_dag(circuit)
        original = deepcopy(dag)
        twirled = TwoQubitPauliTwirlPass().run(dag)
        assert twirled is dag, "Twirled DAG should be the same object."
        assert twirled == original, "Twirled DAG should be the same as the original."
