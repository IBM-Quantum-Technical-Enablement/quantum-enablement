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

"""Test MBL circuits module."""

from pytest import mark, raises  # pylint: disable=import-error
from qiskit import QuantumCircuit

from quantum_enablement.circuits._mbl import MBLChainCircuit


def two_qubit_depth_filter(instruction):
    """Filter instructions which do not contribute to two-qubit depth."""
    num_qubits = instruction.operation.num_qubits
    name = instruction.operation.name
    return num_qubits > 1 and name != "barrier"


class TestMBLChainCircuit:
    """Test suite for the MBLChainCircuit class."""

    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_creation(self, num_qubits, depth):
        """Test MBLChainCircuit creation."""
        circuit = MBLChainCircuit(num_qubits, depth, barriers=False, measurements=False)
        op_counts = circuit.count_ops()
        assert isinstance(circuit, QuantumCircuit), "MBLChainCircuit must be a QuantumCircuit."
        assert circuit.num_qubits == num_qubits, "Wrong number of qubits."
        assert circuit.depth(two_qubit_depth_filter) == depth, "Wrong two-qubit depth."
        assert circuit.num_parameters == (
            (num_qubits + 1) if depth else 0
        ), "Wrong number of parameters."
        assert op_counts.get("x", 0) == num_qubits // 2, "Wrong number of X gates."
        assert op_counts.get("cz", 0) == (num_qubits - 1) * depth // 2, "Wrong number of CZ gates."
        assert op_counts.get("u", 0) == (num_qubits - 1) * depth, "Wrong number of U gates."
        assert op_counts.get("p", 0) == num_qubits * depth, "Wrong number of P gates."
        assert op_counts.get("barrier", 0) == 0, "Wrong number of barriers."
        assert op_counts.get("measure", 0) == 0, "Wrong number of measurements."

    def test_defaults(self):
        """Test MBLChainCircuit defaults."""
        circuit = MBLChainCircuit(4, 4)  # Note: default barriers and measurements are False.
        op_counts = circuit.count_ops()
        assert op_counts.get("barrier", 0) == 0, "By default barriers should not be present."
        assert op_counts.get("measure", 0) == 0, "By default measurements should not be present."

    # pylint: disable=duplicate-code
    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_barriers(self, num_qubits, depth):
        """Test MBLChainCircuit barriers."""
        circuit = MBLChainCircuit(num_qubits, depth, measurements=False, barriers=True)
        op_counts = circuit.count_ops()
        assert op_counts.get("barrier", 0) == depth, "Wrong number of barriers."
        # circuit.remove_operations("barrier")
        # assert circuit == MBLChainCircuit(num_qubits, depth, measurements=False, barriers=False)

    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_measurements(self, num_qubits, depth):
        """Test MBLChainCircuit measurements."""
        circuit = MBLChainCircuit(num_qubits, depth, measurements=True, barriers=False)
        op_counts = circuit.count_ops()
        assert op_counts.get("measure", 0) == num_qubits, "Wrong number of measurements."
        # circuit.remove_final_measurements()
        # assert circuit == MBLChainCircuit(num_qubits, depth, measurements=False, barriers=False)

    @mark.parametrize("num_qubits", [4, 6])
    @mark.parametrize("depth", [0, 2, 4])
    def test_name(self, num_qubits, depth):
        """Test MBLChainCircuit name."""
        circuit = MBLChainCircuit(num_qubits, depth)
        assert circuit.name == f"MBLChainCircuit<{num_qubits}, {depth}>", "Wrong circuit name."

    def test_invalid_num_qubits(self):
        """Test MBLChainCircuit with invalid number of qubits."""
        with raises(ValueError, match=r"Number of qubits \(2\) must be greater than two."):
            MBLChainCircuit(2, 2)

        with raises(ValueError, match=r"Number of qubits \(3\) must be even\."):
            MBLChainCircuit(3, 2)

        with raises(TypeError, match=r"Invalid num\. qubits type .*, expected <int>\."):
            MBLChainCircuit("4", 2)  # type: ignore

    def test_invalid_depth(self):
        """Test MBLChainCircuit with invalid depth."""
        with raises(ValueError, match=r"Depth \(-1\) must be positive\."):
            MBLChainCircuit(4, -1)

        with raises(ValueError, match=r"Depth \(1\) must be even\."):
            MBLChainCircuit(4, 1)

        with raises(TypeError, match=r"Invalid depth type .*, expected <int>\."):
            MBLChainCircuit(4, "2")  # type: ignore
