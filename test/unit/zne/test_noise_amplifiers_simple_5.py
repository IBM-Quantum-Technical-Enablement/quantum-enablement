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


@pytest.fixture
def get_circuit(request): # notice the argument name
    return QuantumVolume(
        num_qubits=request.param[0], # notice the attribute of `request`
        depth=request.param[1],
        seed=request.param[2]
    ).decompose(reps=3)


@pytest.mark.parametrize(
    'get_circuit',
    [(4, 4, 6), (10, 5, 10), (4, 4, 0)],
    indirect=True
)
@pytest.mark.parametrize(
    'noise_factor',
    [1, 3, 5],
)
def test_local_2q_amplifier(get_circuit, noise_factor):
    two_q_gate_name = "cx"
    
    folded_circuit = PassManager(
        Local2qAmplifier(
        noise_factor=noise_factor,
        two_q_gate_name=two_q_gate_name
    )).run(get_circuit) # unintuitive. pytest automatically calls `get_circuit()`

    assert folded_circuit.count_ops()['cx'] == (
        get_circuit.count_ops()['cx'] * noise_factor
    )


@pytest.mark.parametrize(
    'get_circuit',
    [(8, 4, 6), (2, 1, 10)],
    indirect=True
)
@pytest.mark.parametrize(
    'noise_factor',
    [1, 3, 5],
)
def test_global_amplifier(get_circuit, noise_factor):
    folded_circuit = PassManager(
        GlobalAmplifier(noise_factor=noise_factor)).run(get_circuit)

    assert folded_circuit.count_ops()['cx'] == (
        get_circuit.count_ops()['cx'] * noise_factor
    )

    assert sum(folded_circuit.count_ops().values()) == (
        sum(get_circuit.count_ops().values()) * noise_factor
    )
