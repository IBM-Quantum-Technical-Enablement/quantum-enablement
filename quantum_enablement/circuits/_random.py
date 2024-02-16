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

"""Miscellaneous tools and circuits."""


from collections.abc import Collection

from numpy import concatenate
from numpy.random import default_rng
from qiskit import QuantumCircuit


class RandomBasisStateCircuit(QuantumCircuit):
    """Random computational basis state quantum circuit.

    Args:
        num_qubits: number of qubits.
        num_excitations: number of qubits in the excited (1) state.
        favored_qubits: qubits to first assign when locating random excitations.
        seed: to randomly choose excitations.
    """

    def __init__(
        self,
        num_qubits: int,
        num_excitations: int,
        *,
        favored_qubits: Collection[int] | None = None,
        seed: int | None = None,
    ) -> None:
        num_qubits = int(num_qubits)
        num_excitations = int(num_excitations)
        favored_qubits = set(favored_qubits or {})  # TODO: validation
        unfavored_qubits = (qubit for qubit in range(num_qubits) if qubit not in favored_qubits)

        rng = default_rng(seed)
        favored_qubits = rng.permutation(list(favored_qubits))
        unfavored_qubits = rng.permutation(list(unfavored_qubits))  # type: ignore
        excited_qubits = concatenate([favored_qubits, unfavored_qubits])[:num_excitations]

        super().__init__(num_qubits)
        self.x(excited_qubits)
