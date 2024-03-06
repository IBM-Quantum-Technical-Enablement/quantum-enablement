# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Library of quantum circuits."""

from numpy import pi, concatenate
from numpy.random import default_rng
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


################################################################################
## SPIN CHAIN
################################################################################
def build_random_spin_chain_circuit(
        num_qubits: int, imbalance: int = 0, *, seed: int | None = None
    ) -> QuantumCircuit:
    """Build quantum circuit preparing a random spin chain in the computational basis.
    Args:
        num_qubits: number of qubits (needs to be even).
        imbalance: imbalance between the number of excited sites and the
            number of unexcited sites from the reference anti-ferromagnetic
            state. Positive (negative) means more (un)excited states.
        seed: to randomly choose excitations.
    Returns: 
        A computational basis state quantum circuit representing a random spin chain.
    """
    if num_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even.")
    if abs(imbalance) > (num_qubits // 2):
        raise ValueError("Imbalance (abs) cannot exceed half the total number of qubits.")

    rng = default_rng(seed)
    even_qubits = rng.permutation(range(0, num_qubits, 2))
    odd_qubits = rng.permutation(range(1, num_qubits, 2))
    num_excitations = (num_qubits // 2) + imbalance
    excited_qubits = concatenate((even_qubits, odd_qubits))[:num_excitations]
    
    circuit = QuantumCircuit(num_qubits)
    for qubit in excited_qubits:
        circuit.x(qubit)
    return circuit


################################################################################
## MBL
################################################################################
def build_mbl_evolution_circuit(
        num_qubits: int, depth: int, *, barriers: bool = False
    ) -> QuantumCircuit:
    """Build parameterized MBL evolution quantum circuit.
    
    Args:
        num_qubit: number of qubits.
        depth: two-qubit depth.
        barriers: if True adds barriers between layers.
    Returns:
        An MBL evolution quantum circuit with regime parameter θ,
        and disorders' parameter-vector φ.
    """
    if num_qubits <= 2:
        raise ValueError("Number of qubits must be greater than two.")
    if num_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even.")
    
    theta = Parameter('θ') 
    phis = ParameterVector('φ', num_qubits)
    circuit = QuantumCircuit(num_qubits)

    for layer in range(depth):
        layer_parity = layer % 2
        if barriers:
            circuit.barrier()
        for qubit in range(layer_parity, num_qubits - 1, 2):
            circuit.cz(qubit, qubit + 1)
            circuit.u(theta, 0, pi, qubit)
            circuit.u(theta, 0, pi, qubit + 1)
        for qubit in range(num_qubits):
            circuit.p(phis[qubit], qubit)
    
    return circuit


def build_periodic_mbl_evolution_circuit(
        num_qubits: int, depth: int, *, barriers: bool = False
    ) -> QuantumCircuit:
    """Build parameterized MBL evolution quantum circuit.
    
    Args:
        num_qubit: number of qubits.
        depth: two-qubit depth.
        barriers: if True adds barriers between layers.
    Returns:
        An MBL evolution quantum circuit with regime parameter θ,
        and disorders' parameter-vector φ.
    """
    if num_qubits <= 2:
        raise ValueError("Number of qubits must be greater than two.")
    if num_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even.")
    
    theta = Parameter('θ') 
    phis = ParameterVector('φ', num_qubits)
    circuit = QuantumCircuit(num_qubits)

    for layer in range(depth):
        layer_parity = layer % 2
        if barriers:
            circuit.barrier()
        for qubit in range(layer_parity, num_qubits - 1, 2):
            circuit.cz(qubit, qubit + 1)
            circuit.u(theta, 0, pi, qubit)
            circuit.u(theta, 0, pi, qubit + 1)
        if layer_parity == 1:
            circuit.cz(num_qubits-1,0)
            circuit.u(theta, 0, pi, num_qubits-1)
            circuit.u(theta, 0, pi, 0)
        for qubit in range(num_qubits):
            circuit.p(phis[qubit], qubit)
    
    return circuit


def build_mbl_circuit(
        num_qubits: int, 
        depth: int, 
        imbalance: int = 0, 
        periodic: bool = True,
        measurements: bool = True, 
        barriers: bool = False, 
        seed: int | None = None,
    ) -> QuantumCircuit:
    """Build parameterized MBL quantum circuit.
    
    Args:
        num_qubits: number of qubits (needs to be even).
        depth: two-qubit depth.
        imbalance: imbalance for the initial spin chain.
        measurements: if True adds measurements at the end.
        barriers: if True adds barriers between layers.
        seed: for the initial random spin chain.
    Returns:
        An MBL quantum circuit with regime parameter θ,
        and disorders' parameter-vector φ.
    """
    circuit = build_random_spin_chain_circuit(num_qubits, imbalance, seed=seed)
    if periodic:
        evolution = build_periodic_mbl_evolution_circuit(num_qubits, depth, barriers=barriers)
    else:
        evolution = build_mbl_evolution_circuit(num_qubits, depth, barriers=barriers)
    circuit.compose(evolution, inplace=True)
    if measurements:
        circuit.measure_all(inplace=True)
    return circuit


def produce_mbl_parameters(interaction_strength, num_qubits, *, seed=None):
    """Produce MBL parameters.
    Args:
        interaction_strength: the strength of interaction relative to pi.
            For a 1D chain the critical strength separating the MBL and
            thermal/ergodic regimes is found to be approximately 0.16 [1].
        num_qubits: the number of qubits.
        seed: a seed to generate the random disorders.
    Returns:
        A tuple holding the interaction angle theta, and the disorders phis.
    Notes:
        [1] Shtanko et.al. Uncovering Local Integrability in Quantum
            Many-Body Dynamics, https://arxiv.org/abs/2307.07552
    """
    rng = default_rng(seed=seed)
    theta = interaction_strength * pi
    phis = rng.uniform(-pi, pi, size=num_qubits).tolist()
    return theta, phis


def build_compute_uncompute_mbl_circuit(
        num_qubits: int, 
        depth: int, 
        imbalance: int = 0, 
        periodic: bool = True,
        measurements: bool = True, 
        barriers: bool = False, 
        seed: int | None = None,
    ) -> QuantumCircuit:
    """Build a parametrized compute-uncompute MBL quantum circuit."""
    if depth % 2 != 0:
       raise ValueError("Depth must be even.")

    initial_state = build_random_spin_chain_circuit(num_qubits, imbalance, seed=seed)
    if periodic:
        evolution = build_periodic_mbl_evolution_circuit(num_qubits, depth, barriers=barriers)
    else:
        evolution = build_mbl_evolution_circuit(num_qubits, depth, barriers=barriers)
    
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(initial_state, inplace=True)
    circuit.compose(evolution, inplace=True)
    circuit.barrier()
    circuit.compose(evolution.inverse(), inplace=True)
    circuit.compose(initial_state.inverse(), inplace=True)
    if measurements:
        circuit.measure_all(inplace=True)
    
    return circuit


def build_clifford_mbl_evolution_circuit(
        num_qubits: int, 
        depth: int, 
        *, 
        barriers: bool = False, 
        seed: int | None = None,
    ) -> QuantumCircuit:
    """Build Clifford MBL evolution quantum circuit.
    
    Args:
        num_qubit: number of qubits.
        depth: two-qubit depth.
        barriers: if True adds barriers between layers.
    Returns:
        A Clifford MBL evolution quantum circuit with regime
        parameter `θ = pi/2`, and disorders' parameters `φ[i]`
        randomly chosen from `{0, pi/2, pi}`.
    """
    if num_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even.")
    
    qc = QuantumCircuit(num_qubits)
    rng = default_rng(seed)
    disorders = rng.choice(['I', 'Z', 'S'], size=num_qubits)

    for d in range(depth):
        layer_parity = d % 2
        if barriers:
            qc.barrier()
        for qubit in range(layer_parity, num_qubits - 1, 2):
            qc.cz(qubit, qubit + 1)
            qc.h(qubit)
            qc.h(qubit + 1)
        for qubit, disorder in enumerate(disorders):
            if disorder == 'I':
                qc.id(qubit)
            elif disorder == 'Z':
                qc.z(qubit)
            elif disorder == 'S':
                qc.s(qubit)
            else:
                raise RuntimeError("Invalid disorder generated.")
    
    return qc


def build_clifford_mbl_circuit(
        num_qubits: int, 
        depth: int, 
        imbalance: int = 0, 
        *,
        measurements: bool = True, 
        barriers: bool = False, 
        seed: int | None = None,
    ) -> QuantumCircuit:
    """Build Clifford MBL quantum circuit.
    
    Args:
        num_qubits: number of qubits (needs to be even).
        depth: two-qubit depth.
        imbalance: imbalance for the initial spin chain.
        measurements: if True adds measurements at the end.
        barriers: if True adds barriers between layers.
        seed: for the initial random spin chain.
    Returns:
        A Clifford MBL quantum circuit with regime parameter
        `θ = pi/2`, and disorders' parameters `φ[i]` randomly
        chosen from `{0, pi/2, pi}`.
    """
    circuit = build_random_spin_chain_circuit(num_qubits, imbalance, seed=seed)
    evolution = build_clifford_mbl_evolution_circuit(num_qubits, depth, barriers=barriers, seed=seed)
    circuit.compose(evolution, inplace=True)
    if measurements:
        circuit.measure_all(inplace=True)
    return circuit


################################################################################
## QAOA
################################################################################
def build_qaoa_circuit(
        num_qubits: int, 
        depth: int, 
        *,
        measurements: bool = True, 
        barriers: bool = False, 
) -> QuantumCircuit:
    """Build parameterized QAOA quantum circuit.
    
    Args:
        num_qubits: number of qubits.
        depth: two-qubit depth (needs to be even).
        measurements: if True adds measurements at the end.
        barriers: if True adds barriers between layers.
    Returns:
        A dense QAOA quantum circuit for a linear, non-cyclic, graph
        with cost parameter-vector γ, and mixer parameter-vector β.
    """
    if num_qubits <= 2:
        raise ValueError("Number of qubits must be greater than two.")
    if depth % 2 != 0:
        raise ValueError("Depth must be even.")
    
    gammas = ParameterVector('γ', depth // 2) 
    betas = ParameterVector('β', depth // 2)

    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    for layer in range(depth // 2):
        if barriers:
            circuit.barrier()
        for qubit in range(0, num_qubits - 1, 2):
            circuit.rzz(gammas[layer], qubit, qubit + 1)
        for qubit in range(1, num_qubits - 1, 2):
            circuit.rzz(gammas[layer], qubit, qubit + 1)
        for qubit in range(num_qubits):
            circuit.rx(betas[layer], qubit)
    
    if measurements:
        circuit.measure_all()
    return circuit


def produce_qaoa_parameters(num_layers, *, seed=None):
    """Produce random QAOA parameters.
    Args:
        num_layers: the number of QAOA layers.
        seed: a seed to generate the random angles.
    Returns:
        A tuple holding the beta and gamma angles.
    """
    rng = default_rng(seed=seed)
    betas = rng.uniform(-pi, pi, size=num_layers).tolist()
    gammas = rng.uniform(-pi, pi, size=num_layers).tolist()
    return betas, gammas


def build_compute_uncompute_qaoa_circuit(
        num_qubits: int, 
        depth: int, 
        *,
        measurements: bool = True, 
        barriers: bool = False, 
    ) -> QuantumCircuit:
    """Build a parametrized compute-uncompute QAOA quantum circuit."""
    if depth % 4 != 0:
       raise ValueError("Depth must be a multiple of four.")

    qaoa = build_qaoa_circuit(num_qubits, depth // 2, measurements=False, barriers=barriers)
    
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(qaoa, inplace=True)
    circuit.barrier()
    circuit.compose(qaoa.inverse(), inplace=True)
    if measurements:
        circuit.measure_all(inplace=True)
    
    return circuit