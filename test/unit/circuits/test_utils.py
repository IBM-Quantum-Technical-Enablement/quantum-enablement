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

"""Test circuit utils module."""

from test import TYPES

from pytest import fixture, mark, raises  # pylint: disable=import-error
from qiskit import QuantumCircuit

from quantum_enablement.circuits import compute_uncompute


def generate_test_circuits():
    """Generate example quantum circuits to test against."""
    # Empty circuit
    circuit = QuantumCircuit(2)
    yield circuit

    # Circuit with gates
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    yield circuit

    # Circuit with barrier
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.barrier()
    circuit.cx(0, 1)
    yield circuit


@fixture(scope="function", params=generate_test_circuits())
def test_circuit(request):
    """Quantum circuit to test against."""
    return request.param


class TestComputeUncompute:
    """Test suite for the compute_uncompute function."""

    def test_output_type(self, test_circuit):  # pylint: disable=redefined-outer-name
        """Test output type."""
        result_circuit = compute_uncompute(test_circuit)
        output_type = type(result_circuit)
        assert isinstance(result_circuit, QuantumCircuit), f"Invalid output type: {output_type}"

    def test_default(self, test_circuit):  # pylint: disable=redefined-outer-name
        """Test with default parameters."""
        result_circuit = compute_uncompute(test_circuit)
        expected_circuit = compute_uncompute(test_circuit, barrier=True, inplace=False)
        assert result_circuit is not test_circuit, "Incorrect default: 'inplace=True'"
        assert result_circuit == expected_circuit, "Incorrect default: 'barrier=False'"

    def test_barrier_true(self, test_circuit):  # pylint: disable=redefined-outer-name
        """Test with barrier=True."""
        result_circuit = compute_uncompute(test_circuit, barrier=True, inplace=False)
        expected_circuit = test_circuit.copy()
        expected_circuit.barrier()
        expected_circuit.compose(test_circuit.inverse(), inplace=True)
        assert result_circuit == expected_circuit, "Unexpected result circuit"

    def test_barrier_false(self, test_circuit):  # pylint: disable=redefined-outer-name
        """Test with barrier=False."""
        result_circuit = compute_uncompute(test_circuit, barrier=False, inplace=False)
        expected_circuit = test_circuit.copy()
        expected_circuit.compose(test_circuit.inverse(), inplace=True)
        assert result_circuit == expected_circuit, "Unexpected result circuit"

    def test_inplace_true(self, test_circuit):  # pylint: disable=redefined-outer-name
        """Test with inplace=True."""
        expected_circuit = compute_uncompute(test_circuit, barrier=True, inplace=False)
        result_circuit = compute_uncompute(test_circuit, barrier=True, inplace=True)
        assert result_circuit is test_circuit, "Circuit construction not inplace"
        assert result_circuit == expected_circuit, "Unexpected result circuit"

    def test_inplace_false(self, test_circuit):  # pylint: disable=redefined-outer-name
        """Test with inplace=False."""
        result_circuit = compute_uncompute(test_circuit, barrier=True, inplace=False)
        expected_circuit = compute_uncompute(test_circuit, barrier=True, inplace=True)
        assert result_circuit is not test_circuit, "Circuit construction inplace"
        assert result_circuit == expected_circuit, "Unexpected result circuit"

    @mark.parametrize("invalid_circuit", TYPES)
    def test_invalid_input_type(self, invalid_circuit):
        """Test compute_uncompute with invalid input type."""
        with raises(TypeError, match=r"Invalid circuit type .*, expected <QuantumCircuit>."):
            compute_uncompute(invalid_circuit)  # type: ignore
