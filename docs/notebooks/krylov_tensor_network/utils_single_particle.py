import numpy as np
#from openfermion import QubitOperator
#from openfermion import get_sparse_operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import SparsePauliOp

#############################################
# single-particle Hamiltonian 
#############################################

def heis_1p(i, j, N):
    """
    i: site1
    j: site 2:
    N: total n_qubits
    """
    mat = np.diag(np.ones((N), dtype = complex))
    mat[i, i] = -1
    mat[i, j] = 2
    mat[j, i] = 2
    mat[j, j] = -1
    return mat

def XX_YY_1p(i, j, N):
    mat = np.diag(np.zeros((N), dtype = complex))
    mat[i, j] = 2
    mat[j, i] = 2
    
    return mat

def XX_1p(i, j, N):
    mat = np.diag(np.zeros((N), dtype = complex))
    mat[i, j] = 1
    mat[j, i] = 1
    
    return mat

def YY_1p(i, j, N):
    mat = np.diag(np.zeros((N), dtype = complex))
    mat[i, j] = 1
    mat[j, i] = 1
    
    return mat

def ZZ_1p(i, j, N):
    mat = np.diag(np.ones((N), dtype = complex))
    mat[i, i] = -1
    mat[j, j] = -1
    return mat

def get_1particle_hamiltonian(H:SparsePauliOp, ):
    #if n_swaps > 0:
        #raise NotImplementedError()
        
    #interaction_all = sum(interaction_list, [])
    n_qubit = H.num_qubits
    interaction_config = extract_interaction_config_from_Heisenberg(H)
    mat = np.zeros((n_qubit, n_qubit), dtype = complex)
    for pair, coeff in interaction_config.items():
        mat += coeff * heis_1p(pair[0], pair[1], n_qubit)
    return mat


def extract_interaction_config_from_Heisenberg(H:SparsePauliOp):
    """
    extracts interacting site indices from Heisenberg interaction.
    """
    n_qubit = H.num_qubits
    int_config = {}

    for prim in H:
        #prim = H[0]
        pauli_string, coeff = prim.to_list()[0]
        pair = tuple([i for i in range(n_qubit) if pauli_string[n_qubit - i - 1] != "I"])
        
        assert len(pair) < 3, "3-local hamiltonian is not assumed in this code"

        if pair not in int_config:
            int_config[pair] = coeff
    return int_config

def heisen_hamiltonian_1p(H:SparsePauliOp):
    n_qubit = H.num_qubits
    mat = np.zeros((n_qubit, n_qubit), dtype = complex)
    interaction_config = extract_interaction_config_from_Heisenberg(H)
    for pair,coeff in interaction_config.items():
        mat += coeff * heis_1p(pair[0], pair[1], n_qubit)
    return mat


def heisen_hamiltonian(interaction_list, n_swaps=0, as_qiskit = False, n_qubits = None):
    from openfermion import QubitOperator
    
    def heisen(i, j):
        return QubitOperator(f"X{i} X{j}") + QubitOperator(f"Y{i} Y{j}") + QubitOperator(f"Z{i} Z{j}")
    
    # TODO: implement this
    if n_swaps >0:
        raise NotImplementedError
    interaction_all = sum(interaction_list, [])
    
    op = QubitOperator()
    for pair in interaction_all:
        op += heisen(*pair)

    if as_qiskit:
        return openfermion_to_SparsePauliOp(op, n_qubit = n_qubits)
    return op

#############################################
# time evolution
#############################################

from qiskit import QuantumCircuit
import copy
def simulate_unitary_1particle(circuit:QuantumCircuit, init_state, param_input = None):
    n_qubit = circuit.num_qubits
    state_buf = init_state.copy()
    for gate in circuit:
        assert len(gate.qubits) == 2, "not Heisenberg evolution?"
        qubit_id_list = [_qubit.index for _qubit in gate.qubits]
        
        if gate.operation.name == "swap":
            state_buf0 = state_buf.copy()
            state_buf[qubit_id_list[0]] = state_buf0[qubit_id_list[1]]
            state_buf[qubit_id_list[1]] = state_buf0[qubit_id_list[0]]
            #state_buf = swap_1p(qubit_id_list[0], qubit_id_list[1])
        else:
            if param_input is not None:
                param = copy.deepcopy(param_input)
            else:
                param = float(gate.operation.params[0])        
            state_buf = heisen_1p_evolution(qubit_id_list[0], qubit_id_list[1], param, n_qubit) @ state_buf    
            
    return state_buf

def _rotation_heis(theta):
    "This is verified via sympy"
    vals = np.array([-3, 1])
    vecs = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [-1/np.sqrt(2), 1/np.sqrt(2)],
           ])
    
    #return vecs @ np.diag(np.exp(-1j * vals * angle/2)) @ vecs.T
    return np.array([
        [np.exp(3j * theta/2)/2 + np.exp(-1j * theta /2)/2, -np.exp(3j * theta/2)/2 + np.exp(-1j * theta /2)/2,],
        [-np.exp(3j * theta/2)/2 + np.exp(-1j * theta /2)/2, np.exp(3j * theta/2)/2 + np.exp(-1j * theta /2)/2,],
    ])


def heisen_1p_evolution(i, j, param, n_qubit):
    #TODO : direct implementation of unitary
    heis_mat = heis_1p(i, j, n_qubit)
    u_small = _rotation_heis(param)
    u = np.zeros((n_qubit, n_qubit), dtype = complex)
    
    #
    u[i, i] = u_small[0, 0]
    u[i, j] = u_small[0, 1]
    u[j, i] = u_small[1, 0]
    u[j, j] = u_small[1, 1]
    
    for k in range(n_qubit):
        if k!=i and k!=j:
            u[k, k] = np.exp(-1j * param/2)
    return u

# this is too slow
def heisen_1p_evolution_old(i, j, param, n_qubit):
    #TODO : direct implementation of unitary
    heis_mat = heis_1p(i, j, n_qubit)
    #u_small = _rotation_heis(param)
    return scipy.linalg.expm(-1j * heis_mat * param/2)


import scipy
import itertools

#############################################
# Subspace construction
#############################################

def dimer_state_1particle(i, j, N):
    state = np.zeros(N, dtype = complex)
    state[i] = 1/np.sqrt(2)
    state[j] = -1/np.sqrt(2)
    return state

def get_indexes_for_m_particles(n, m):
    """
    returns a list of indices with Hamming weight m, for n qubits system
    """
    indexes = []
    
    for one_positions in itertools.combinations(range(n), m):
        binarray = ["0" for _ in range(n)]
        for idx in one_positions:
            binarray[n - idx - 1] = "1"
        indexes.append(int("".join(binarray), base = 2))
    return sorted(indexes)

def extract_submatrix(matrix, indexes):
    """
    extract submatrix
    """
    return matrix[np.ix_(indexes, indexes)]

def extract_subvector(vector, indexes):
    """
    extract subvector
    """
    return vector[np.ix_(indexes)]


def _to_paulistring(paulis, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    for tup in paulis:
        l[n_qubit - 1 - tup[0]] = tup[1]
    paulistring = "".join(l)
    return paulistring
    

def openfermion_to_SparsePauliOp(operator, n_qubit=None):
    """
    Converts QubitOperator into SparsePauliOp
    """
    if n_qubit is None:
        from openfermion import count_qubits
        n_qubit = count_qubits(operator)
        
    paulistring_list = []
    coeff_list = []
    for paulis, coeff in operator.terms.items():
        paulistring = _to_paulistring(paulis, n_qubit)
        
        paulistring_list.append(paulistring)
        coeff_list.append(coeff)
        
    new_op = SparsePauliOp(paulistring_list, coeff_list)
    return new_op