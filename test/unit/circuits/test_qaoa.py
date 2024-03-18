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

"""Test QAOA circuits module."""


from pytest import mark, raises  # pylint: disable=import-error
from qiskit import QuantumCircuit

from quantum_enablement.circuits._qaoa import QAOAPathCircuit


def two_qubit_depth_filter(instruction):
    """Filter instructions which do not contribute to two-qubit depth."""
    num_qubits = instruction.operation.num_qubits
    name = instruction.operation.name
    return num_qubits > 1 and name != "barrier"


class TestQAOAPathCircuit:
    """Test suite for the QAOAPathCircuit class."""

    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_creation(self, num_qubits, depth):
        """Test QAOAPathCircuit creation."""
        circuit = QAOAPathCircuit(num_qubits, depth, barriers=False, measurements=False)
        op_counts = circuit.count_ops()
        assert isinstance(circuit, QuantumCircuit), "QAOAPathCircuit must be a QuantumCircuit."
        assert circuit.num_qubits == num_qubits, "Wrong number of qubits."
        assert circuit.depth(two_qubit_depth_filter) == depth, "Wrong two-qubit depth."
        assert circuit.num_parameters == depth, "Wrong number of parameters."
        assert op_counts.get("h", 0) == num_qubits, "Wrong number of H gates."
        assert op_counts.get("rzz", 0) == (num_qubits - 1) * (
            depth // 2
        ), "Wrong number of RZZ gates."
        assert op_counts.get("rx", 0) == num_qubits * (depth // 2), "Wrong number of RX gates."
        assert op_counts.get("barrier", 0) == 0, "Wrong number of barriers."
        assert op_counts.get("measure", 0) == 0, "Wrong number of measurements."

    def test_defaults(self):
        """Test QAOAPathCircuit defaults."""
        circuit = QAOAPathCircuit(4, 4)  # Note: default barriers and measurements are False.
        op_counts = circuit.count_ops()
        assert op_counts.get("barrier", 0) == 0, "By default barriers should not be present."
        assert op_counts.get("measure", 0) == 0, "By default measurements should not be present."

    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_barriers(self, num_qubits, depth):
        """Test QAOAPathCircuit barriers."""
        circuit = QAOAPathCircuit(num_qubits, depth, barriers=True, measurements=False)
        op_counts = circuit.count_ops()
        assert op_counts.get("barrier", 0) == depth // 2, "Wrong number of barriers."
        # circuit.remove_operations("barrier")
        # assert circuit == QAOAPathCircuit(num_qubits, depth, measurements=False, barriers=False)

    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_measurements(self, num_qubits, depth):
        """Test QAOAPathCircuit measurements."""
        circuit = QAOAPathCircuit(num_qubits, depth, barriers=False, measurements=True)
        op_counts = circuit.count_ops()
        assert op_counts.get("measure", 0) == num_qubits, "Wrong number of measurements."
        assert op_counts.get("barrier", 0) == 1, "Measurements should be preceded by a barrier."
        # circuit.remove_final_measurements()
        # assert circuit == QAOAPathCircuit(num_qubits, depth, measurements=False, barriers=False)

    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_name(self, num_qubits, depth):
        """Test QAOAPathCircuit name."""
        circuit = QAOAPathCircuit(num_qubits, depth)
        assert circuit.name == f"QAOAPathCircuit<{num_qubits}, {depth}>", "Wrong circuit name."

    def test_invalid_num_qubits(self):
        """Test QAOAPathCircuit with invalid number of qubits."""
        with raises(ValueError, match=r"Number of qubits \(2\) must be greater than two."):
            QAOAPathCircuit(2, 2)

        with raises(ValueError, match=r"Number of qubits \(3\) must be even\."):
            QAOAPathCircuit(3, 2)

        with raises(TypeError, match=r"Invalid num\. qubits type .*, expected <int>\."):
            QAOAPathCircuit("4", 2)  # type: ignore

    def test_invalid_depth(self):
        """Test QAOAPathCircuit with invalid depth."""
        with raises(ValueError, match=r"Depth \(-1\) must be positive\."):
            QAOAPathCircuit(4, -1)

        with raises(ValueError, match=r"Depth \(1\) must be even\."):
            QAOAPathCircuit(4, 1)

        with raises(TypeError, match=r"Invalid depth type .*, expected <int>\."):
            QAOAPathCircuit(4, "2")  # type: ignore
