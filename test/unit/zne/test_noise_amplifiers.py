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
def get_circuit(request):
    circuit = QuantumVolume(
        num_qubits=request.param[0],
        depth=request.param[1],
        seed=request.param[2]
    )

    circuit = transpile(
        circuits=circuit,
        basis_gates=['id', 'sx', 'x', 'rz', 'cx'],
        optimization_level=0
    )

    return circuit

@pytest.mark.parametrize(
    'get_circuit',
    product(
        np.random.choice(a=range(2, 10), size=1, replace=False),
        np.random.choice(a=range(1, 10), size=1, replace=False),
        np.random.choice(a=range(100), size=3, replace=False)
    ),
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
    )).run(get_circuit)

    assert folded_circuit.count_ops()['cx'] == (
        get_circuit.count_ops()['cx'] * noise_factor
    )


@pytest.mark.parametrize(
    'get_circuit',
    product(
        np.random.choice(a=range(2, 10), size=1, replace=False),
        np.random.choice(a=range(1, 10), size=1, replace=False),
        np.random.choice(a=range(100), size=3, replace=False)
    ),
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
