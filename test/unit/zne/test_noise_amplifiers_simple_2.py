import pytest
import numpy as np
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QuantumVolume
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import PassManager

from quantum_enablement.zne.noise_amplification.noise_amplifiers import (
    Local2qAmplifier,
    GlobalAmplifier
)


# @pytest.fixture
# def get_circuit(request):
#     circuit = QuantumVolume(
#         num_qubits=request.param[0],
#         depth=request.param[1],
#         seed=request.param[2]
#     )

#     circuit = transpile(
#         circuits=circuit,
#         basis_gates=['id', 'sx', 'x', 'rz', 'cx'],
#         optimization_level=0
#     )

#     return circuit


def test_local_2q_amplifier():
    circuit = QuantumVolume(
        num_qubits=4,
        depth=4,
        seed=0
    ).decompose(reps=3)
    two_q_gate_name = "cx"
    
    for noise_factor in [1, 3, 5]:
        folded_circuit = PassManager(
            Local2qAmplifier(
            noise_factor=noise_factor,
            two_q_gate_name=two_q_gate_name
        )).run(circuit)

        assert folded_circuit.count_ops()['cx'] == (
            circuit.count_ops()['cx'] * noise_factor
        )


def test_global_amplifier():
    circuit = QuantumVolume(
        num_qubits=4,
        depth=4,
        seed=0
    ).decompose(reps=3)

    for noise_factor in [1, 3, 5]:
        folded_circuit = PassManager(
            GlobalAmplifier(noise_factor=noise_factor)).run(circuit)

        assert folded_circuit.count_ops()['cx'] == (
            circuit.count_ops()['cx'] * noise_factor
        )

        assert sum(folded_circuit.count_ops().values()) == (
            sum(circuit.count_ops().values()) * noise_factor
        )
