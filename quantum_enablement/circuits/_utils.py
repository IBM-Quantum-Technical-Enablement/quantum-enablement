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

"""Utility tools for quantum circuits."""


from qiskit import QuantumCircuit


################################################################################
## COMPUTE-UNCOMPUTE
################################################################################
def compute_uncompute(
    circuit: QuantumCircuit, *, barrier: bool = True, inplace: bool = False
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
        raise TypeError(f"Invalid circuit type {type(circuit)}, expected <QuantumCircuit>.")
    inverse = circuit.inverse()
    if not inplace:
        circuit = circuit.copy()
    if barrier:
        circuit.barrier()
    circuit.compose(inverse, inplace=True)
    return circuit
