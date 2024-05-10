import scipy
import copy
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from utils_single_particle import extract_interaction_config_from_Heisenberg, get_indexes_for_m_particles

###################################
# k-particle operators
###################################
def swap(binary, i, j):
    new_list = [b for b in binary]
    N = len(binary)
    new_list[N - 1- i] = binary[N - 1 - j]
    new_list[N - 1 - j] = binary[N - 1 - i]
    return "".join(new_list)

def binary_to_index(binary, ):
    return int(binary, base = 2)

def XX_kp(i, j, N, indexes, as_sparse = False):
    dim = len(indexes)
    if not as_sparse:
        mat = np.zeros((dim, dim))
    else:
        mat = scipy.sparse.lil_matrix((dim, dim))
    
    #binaries = [bin(ind)[2:].zfill(N) for ind in indexes]
    #interacting_indexes = [get_swapped_index(index, i, j, N) for index in indexes]
    interacting_indexes = get_swapped_index(indexes, i, j, N)
    
    orders = list(range(len(indexes)))
    interacting_orders = np.argsort(interacting_indexes)
    
    for index, arg1, arg2 in zip(indexes, orders, interacting_orders):
        if ((index >> i) ^ (index>>j))%2 == 0: # i-th and j-th binary are same
            continue
        else:
            mat[arg1, arg2] = 1
            mat[arg2, arg1] = 1

    if as_sparse:
        return mat.tocsr()
    return mat

def YY_kp(i, j, N, indexes, as_sparse = False):
    return XX_kp(i, j, N, indexes, as_sparse = as_sparse)

def ZZ_kp(i, j, N, indexes, as_sparse = False):
    dim = len(indexes)
    if not as_sparse:
        mat = np.zeros((dim, dim))
    else:
        mat = scipy.sparse.lil_matrix((dim, dim))
    
    for arg, ind in enumerate(indexes):
        if ((ind >> i) ^ (ind>>j))%2 == 0: # i-th and j-th binary are same
            mat[arg, arg] = 1.0
        else:
            mat[arg, arg] = -1.0
        
    if as_sparse:
        return mat.tocsr()

    return mat

def heis_kp(i, j, N, indexes, as_sparse = False, mat = None, coeff = 1):
    dim = len(indexes)

    interacting_indexes = get_swapped_index(indexes, i, j, N)
    orders = list(range(len(indexes)))
    interacting_orders = np.argsort(interacting_indexes)

    if mat is None:
        if not as_sparse:
            mat = np.zeros((dim, dim), dtype = complex)
        else:
            mat = scipy.sparse.lil_matrix((dim, dim), dtype = complex)

    for index, arg1, arg2 in zip(indexes, orders, interacting_orders):
        # XX+YY
        if ((index >> i) ^ (index>>j))%2 == 0: # i-th and j-th binary are same
            #continue
            #mat[arg1, arg1] += 1 * coeff #ZZ
            mat[arg1, arg1] = 1 #ZZ
        else:
            #mat[arg1, arg2] += 2 * coeff # XX+YY
            #mat[arg2, arg1] += 2 * coeff# XX+YY
            #mat[arg1, arg1] += -1 *coeff# ZZ
            mat[arg1, arg2] = 2 # XX+YY
            mat[arg2, arg1] = 2# XX+YY
            mat[arg1, arg1] = -1# ZZ

    if as_sparse:
        mat = mat.tocsr()
    return mat

def heisen_hamiltonian_kp(H:SparsePauliOp, k, as_sparse = False):
    n_qubit = H.num_qubits
    indexes = get_indexes_for_m_particles(n_qubit , k)
    
    dim = len(indexes)
    if not as_sparse:
        mat = np.zeros((dim, dim), dtype = complex)
    else:
        mat = scipy.sparse.csr_matrix((dim, dim),dtype= complex)
    
    interaction_config = extract_interaction_config_from_Heisenberg(H)
    for pair,coeff in interaction_config.items():
        if pair == ():
            continue
        mat += heis_kp(pair[0], pair[1], n_qubit, indexes, as_sparse = as_sparse, )
    if as_sparse:
        mat = mat.tocsr()
    return mat


########################################
# k-particle evolution unitary
########################################

from utils_single_particle import _rotation_heis

def get_swapped_index(indexes, i, j, N):
    indexes_np = np.copy(indexes)

    bitmask = (1 << i | (1 << j))

    # i と j の位置のビットが異なる場合のみ XOR 演算でビットを反転
    indexes_np = indexes_np ^ (((indexes_np >> i) ^ (indexes_np >> j))%2 * bitmask)
    return indexes_np

def get_swapped_index_slow(index, i, j, N):
    binary = bin(index)[2:].zfill(N)
    l = [b for b in binary]
    l_new = copy.deepcopy(l)
    l_new[N - 1- i] = l[N - 1 - j]
    l_new[N - 1- j] = l[N - 1- i]
    return binary_to_index("".join(l_new))

def heisen_kp_evolution(i, j, param, n_qubit, k, as_sparse = False, indexes = None, u_uninitialized = None):

    if indexes is None:
        indexes = get_indexes_for_m_particles(n_qubit ,k)
    #interacting_indexes = [get_swapped_index(index, i, j, n_qubit) for index in indexes]
    interacting_indexes = get_swapped_index(indexes, i, j, n_qubit)
    
    orders = list(range(len(indexes)))
    #interacting_orders = [indexes.index(i) for i in interacting_indexes]
    interacting_orders = list(np.argsort(interacting_indexes))
    u_small = _rotation_heis(param)
    
    dim = len(indexes)
    if as_sparse:
        u = scipy.sparse.lil_matrix((dim, dim), dtype = complex)
    else:
        if u_uninitialized is not None:
            u = np.zeros_like(u_uninitialized)
        else:
            u = np.zeros((dim, dim), dtype= complex)
        
    for ind1, ind2 in zip(orders, interacting_orders):
        if ind1 != ind2:
            u[ind1, ind1] = u_small[0, 0]
            u[ind1, ind2] = u_small[0, 1]
            u[ind2, ind1] = u_small[1, 0]
            u[ind2, ind2] = u_small[1, 1]
        else:
            u[ind1, ind1] = np.exp(-1j * param/2)
    return u


from qiskit import QuantumCircuit
import copy
def to_binary_array(indexes, N):
    return [bin(i)[2:].zfill(N) for i in indexes]

def swap_qubit(state_buf, i, j, N, indexes):
    binaries = to_binary_array(indexes, N)
    new_binaries = [swap(binary, i, j) for binary in binaries]
    
    new_indexes = [binary_to_index(binary) for binary in new_binaries]
    order = [indexes.index(i) for i in new_indexes]
    return state_buf[np.ix_(order)]

from utils_single_particle import heisen_hamiltonian
def simulate_unitary_kparticle_from_interaction_list(interaction_list,init_state, N, k, n_T_step, param, as_sparse = True,n_swaps=0):
    assert n_swaps == 0, "n_swaps >= 1 not implemented."
    assert type(interaction_list[0][0]) in [list, tuple], "make sure you have List[List[List]] for interaction_list."
    assert as_sparse, "dense matrix simulation not implemented yet."
    state_buf = init_state.copy()
    indexes = get_indexes_for_m_particles(N, k)
    
    hamiltonians = [heisen_hamiltonian([interaction_list[i]], as_qiskit = True, n_qubits = N) for i in range(len(interaction_list))]
    hamiltonians_kp = [heisen_hamiltonian_kp(ham, k, as_sparse = True) for ham in hamiltonians]
    for t in range(n_T_step):
        if t%2 == 0:
            for ham_kp in hamiltonians_kp:
                state_buf = scipy.sparse.linalg.expm_multiply(-1j * ham_kp * param/2, state_buf)
        else:
            for ham_kp in hamiltonians_kp[::-1]:
                state_buf = scipy.sparse.linalg.expm_multiply(-1j * ham_kp * param/2, state_buf)
            
    return state_buf

def simulate_unitary_kparticle_slow(circuit:QuantumCircuit, init_state, k, param_input = None, as_sparse = True):
    n_qubit = circuit.num_qubits
    state_buf = init_state.copy()
    indexes = get_indexes_for_m_particles(n_qubit, k)
    
    for gate in circuit:
        assert len(gate.qubits) == 2, "not Heisenber evolution?"
        qubit_id_list = [_qubit.index for _qubit in gate.qubits]
        
        if gate.operation.name == "swap":
            state_buf = swap_qubit(state_buf, qubit_id_list[0], qubit_id_list[1], indexes)
        else:
            if param_input is not None:
                param = copy.deepcopy(param_input)
            else:
                param = float(gate.operation.params[0])
            #state_buf = heisen_kp_evolution(qubit_id_list[0], qubit_id_list[1], param, n_qubit, k, indexes = indexes, as_sparse = as_sparse) @ state_buf
                
    return state_buf

#def evolve_by_heisen_kp(i, j, param, N, k, state_buf):

#######################################
# k-particle state
######################################

def k_particle_excitation(n_qubits:int, excitation_pos:list, indexes:list):
    import math
    k = len(excitation_pos)
    vec = np.zeros(math.comb(n_qubits, k), dtype = complex)
    
    binary_list = ["0" for _ in range(n_qubits)]
    for pos in excitation_pos:
        binary_list[n_qubits - 1 - pos] = "1"
    binary = "".join(binary_list)
    
    vec[indexes.index(binary_to_index(binary))] = 1
    return vec



################
# legacy
###############
def XX_kp_slow(i, j, N, indexes, as_sparse = False):
    dim = len(indexes)
    if not as_sparse:
        mat = np.zeros((dim, dim))
    else:
        mat = scipy.sparse.csr_matrix((dim, dim))
    
    binaries = [bin(ind)[2:].zfill(N) for ind in indexes]
    
    unseen_binaries = copy.deepcopy(binaries)
    
    while len(unseen_binaries) > 0:
        binary = unseen_binaries.pop(0)
        ind = binary_to_index(binary, )
        
        # currently based on little endian
        if binary[N-i -1] == binary[N - j - 1]:
            pass
        else:
            new_binary = swap(binary, i, j)
            new_ind = binary_to_index(new_binary)
            
            mat_ind = indexes.index(ind)
            mat_new_ind = indexes.index(new_ind)
            mat[mat_ind, mat_new_ind] = 1
            mat[mat_new_ind, mat_ind] = 1
            unseen_binaries.remove(new_binary)
    return mat

def ZZ_kp_slow(i, j, N, indexes, as_sparse = False):
    dim = len(indexes)
    if not as_sparse:
        mat = np.zeros((dim, dim))
    else:
        mat = scipy.sparse.csr_matrix((dim, dim))
    
    binaries = [bin(ind)[2:].zfill(N) for ind in indexes]
    
    for binary in binaries:
        ind = binary_to_index(binary)
        mat_ind = indexes.index(ind)
        
        if binary[N - 1- i] == binary[N - 1- j]:
            mat[mat_ind, mat_ind] = 1
        elif binary[N - 1- i] != binary[N - 1- j]:
            mat[mat_ind, mat_ind] = -1
            
        
    return mat

###########################
### Get Krylov matrices ###
###########################

from qiskit.quantum_info import Statevector

def compute_krylov_matrices(n_qubits, excitations, H, interaction_list, n_T_step, dt, D, with_trotter_error=True, vectors_kp=None, H_kp=None):

    k = len(excitations)

    # Krylov vectors in k-particle representation
    if vectors_kp is None:
        indices = get_indexes_for_m_particles(n_qubits, k)
        initial_state_kp = k_particle_excitation(n_qubits, excitations, indices)
        vectors_kp = [simulate_unitary_kparticle_from_interaction_list(interaction_list, initial_state_kp, n_qubits, k, n_T_step, param = (i*dt)/n_T_step, as_sparse = True) for i in range(D)]

    # Hamiltonian in k-particle representation
    if H_kp is None:
        H_kp = heisen_hamiltonian_kp(H, k, as_sparse = True)

    # Standardize form of vectors
    if isinstance(vectors_kp[0], Statevector):
        vectors = [_vec.data for _vec in vectors_kp]
    else:
        vectors = np.copy(vectors_kp)
        
    if with_trotter_error: # Version of circuit used in experiment
        H_krylov = np.zeros((D, D), dtype  = complex)
        S_krylov = np.zeros((D, D), dtype  = complex)

        for jmi in range(D):
            h_jmi = vectors[0].conj() @ H_kp @ vectors[jmi]
            s_jmi = vectors[0].conj() @ vectors[jmi]

            for i in range(D-jmi):
                H_krylov[i, jmi + i] = np.copy(h_jmi)
                S_krylov[i, jmi + i] = np.copy(s_jmi)
                if jmi !=0:
                    H_krylov[jmi+i, i] = np.copy(h_jmi.conj())
                    S_krylov[jmi+i, i] = np.copy(s_jmi.conj())    

    else: # without the "commutation approximation"
        H_krylov = np.zeros((D, D), dtype  = complex)
        S_krylov = np.zeros((D, D), dtype  = complex)

        for i in range(D):
            for j in range(i, D):
                h_ij = vectors[i].conj() @ H_kp @ vectors[j]
                s_ij = vectors[i].conj() @ vectors[j]

                H_krylov[i, j] = np.copy(h_ij)
                S_krylov[i, j] = np.copy(s_ij)
                if i!=j:
                    H_krylov[j, i] = np.copy(h_ij.conj())
                    S_krylov[j, i] = np.copy(s_ij.conj())

    return H_krylov, S_krylov, vectors, H_kp

###################################
### Compute Pauli matrix values ###
###################################

def insert_ancilla(i, ancilla):
    if i< ancilla:
        return i
    elif i >= ancilla:
        return i + 1
    
def get_XX_paulistring_real(i, j, ancilla, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "X"
    l[n_qubit - 1 - j] = "X"
    l[n_qubit - 1 - ancilla] = "X"
    return "".join(l)

def get_XX_paulistring_imag(i, j, ancilla, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "X"
    l[n_qubit - 1 - j] = "X"
    l[n_qubit - 1 - ancilla] = "Y"
    return "".join(l)

def get_YY_paulistring_real(i, j, ancilla, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "Y"
    l[n_qubit - 1 - j] = "Y"
    l[n_qubit - 1 - ancilla] = "X"
    return "".join(l)

def get_YY_paulistring_imag(i, j, ancilla, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "Y"
    l[n_qubit - 1 - j] = "Y"
    l[n_qubit - 1 - ancilla] = "Y"
    return "".join(l)

def get_ZZ_paulistring_real(i, j, ancilla, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "Z"
    l[n_qubit - 1 - j] = "Z"
    l[n_qubit - 1 - ancilla] = "X"
    return "".join(l)

def get_ZZ_paulistring_imag(i, j, ancilla, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "Z"
    l[n_qubit - 1 - j] = "Z"
    l[n_qubit - 1 - ancilla] = "Y"
    return "".join(l)

def get_XX_paulistring(i, j, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "X"
    l[n_qubit - 1 - j] = "X"
    return "".join(l)

def get_YY_paulistring(i, j, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "Y"
    l[n_qubit - 1 - j] = "Y"
    return "".join(l)

def get_ZZ_paulistring(i, j, n_qubit):
    l = ["I" for _ in range(n_qubit)]
    l[n_qubit - 1 - i] = "Z"
    l[n_qubit - 1 - j] = "Z"
    return "".join(l)



def compute_pauli_matrices(n_qubits, control, excitations, H, interaction_list, n_T_step, dt, D_list, vectors_kp=None):

    relative_phases = {d : np.exp(-1j*sum([c for p,c in H.to_list() if 'Z' in p])*d*dt/2) for d in D_list}

    k = len(excitations)
    indices = get_indexes_for_m_particles(n_qubits, k)

    # Krylov vectors in k-particle representation
    if vectors_kp is None:
        initial_state_kp = k_particle_excitation(n_qubits, excitations, indices)
        vectors_kp = [simulate_unitary_kparticle_from_interaction_list(interaction_list, initial_state_kp, n_qubits, k, n_T_step, param = (i*dt)/n_T_step, as_sparse = True) for i in range(D)]

    # Standardize form of vectors
    if isinstance(vectors_kp[0], Statevector):
        vectors = [_vec.data for _vec in vectors_kp]
    else:
        vectors = np.copy(vectors_kp)

    interaction_config = extract_interaction_config_from_Heisenberg(H)

    all_exp_vals = []
        
    for d in D_list:

        expectation_values = {}
        for pair in interaction_config.keys():
            q1_original = insert_ancilla(pair[0], control)
            q2_original = insert_ancilla(pair[1], control)

            # XX
            xx_sp = XX_kp(pair[0], pair[1], n_qubits, indices, as_sparse = True)
            experiment_value = (vectors[0].conj() @ xx_sp @ vectors[d]) / relative_phases[d]
            expectation_values[get_XX_paulistring_real(q1_original, q2_original, control, n_qubits + 1)] = experiment_value.real
            expectation_values[get_XX_paulistring_imag(q1_original, q2_original, control, n_qubits + 1)] = experiment_value.imag

            # YY
            yy_sp = YY_kp(pair[0], pair[1], n_qubits, indices, as_sparse = True)
            experiment_value = (vectors[0].conj() @ yy_sp @ vectors[d]) / relative_phases[d]
            expectation_values[get_YY_paulistring_real(q1_original, q2_original, control, n_qubits + 1)] = experiment_value.real
            expectation_values[get_YY_paulistring_imag(q1_original, q2_original, control, n_qubits + 1)] = experiment_value.imag

            # ZZ
            zz_sp = ZZ_kp(pair[0], pair[1], n_qubits, indices, as_sparse = True)
            experiment_value = (vectors[0].conj() @ zz_sp @ vectors[d]) / relative_phases[d]
            expectation_values[get_ZZ_paulistring_real(q1_original, q2_original, control, n_qubits + 1)] = experiment_value.real
            expectation_values[get_ZZ_paulistring_imag(q1_original, q2_original, control, n_qubits + 1)] = experiment_value.imag

        all_exp_vals.append(expectation_values)

    return all_exp_vals, vectors
    