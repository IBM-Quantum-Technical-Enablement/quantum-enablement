from TN_circuit import *
from lattice_afi import *
import numpy as np
from scipy.linalg import expm, toeplitz
import scipy as sp

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.eye(2)
XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
zero = np.array([1,0])
one = np.array([0,1])

class HeavyHexHeisenbergKrylovTNSim(TNCircuit):

    def __init__(self, nx: int, ny: int, chi_max:int, J: float=1.0, g:float=1.0, mps_order:list[int]|str = None)-> None:
        '''
        Initializes the Tensor Network simulator for the Quantum Krylov Algorithm on the 
        Heisenberg XXZ model on a Heavy-Hexagonal lattice, using the generalized Hadamard test for systems
        with a known initial eigenstate and / or U(1) symmetry,
        Parameters:
            nx, ny: Linear dimensions of Heavy Hex lattice
            chi_max: Maximum bond dimensions to truncate to for all computations
            J: Heisenberg coupling constant
            g: Anisotropy in ZZ couplings for XXZ model (=/= 1 away from SU(2) point)
            mps_order: Either 'zigzag' or another descriptive key for pre-defined MPS order. Defaults to zigzag. If
                        a list is passed, the explicit mps order defined by the list of qubit positions in the mps is used.
        Returns:
            None
        '''
        self.lattice = lattice_2d(tp='heavy_hex', nx=nx, ny=ny, mps_order=mps_order)
        TNCircuit.__init__(self, mps_order=[0]+[self.lattice.mps_to_qubit[i]+1 for i in range(self.lattice.n_qubits)], chi_max=chi_max)

        #+1 for ancilla
        self.J = J
        self.g = g

        self.pre_measure_states = None
        self.flip_inds = None

        self.layer1_links = [(self.lattice.lat_to_mps_idx(k[0])+1, self.lattice.lat_to_mps_idx(k[1])+1) for k,v in self.lattice.couplings.items() 
                             if self.lattice.lat_to_mps_idx(k[0]) < self.lattice.lat_to_mps_idx(k[1]) and v==1]
        self.layer2_links = [(self.lattice.lat_to_mps_idx(k[0])+1, self.lattice.lat_to_mps_idx(k[1])+1) for k,v in self.lattice.couplings.items() 
                             if self.lattice.lat_to_mps_idx(k[0]) < self.lattice.lat_to_mps_idx(k[1]) and v==2]
        self.layer3_links = [(self.lattice.lat_to_mps_idx(k[0])+1, self.lattice.lat_to_mps_idx(k[1])+1) for k,v in self.lattice.couplings.items() 
                             if self.lattice.lat_to_mps_idx(k[0]) < self.lattice.lat_to_mps_idx(k[1]) and v==3]
        return
        

    def initialize_state(self, initial_state: list[int] | int | str = 'GHZ', coeffs:list[complex]|None = None, flip_inds:list[int]|None = None) -> None:
        '''
        Prepares initial state for the Krylov algorithm.
        Parameters:
            initial_state: Either a list of bits indicating the initial product state, or a single bit indicating that all qubits 
                            are in the same state, or 'GHZ'. 
            coeffs (optional): Only used if initial_state == 'GHZ'. Length 2 list of coefficients that weighs the superposition in the 
                                GHZ-like state.
            flip_inds (optional): Only used if initial_state == 'GHZ'. List of integers specifying what indices of the system qubits to 
                                    apply controlled X gates to for GHZ-like state preparation in a particular particle sector. Note, indices
                                    do not include ancilla. i.e an index of 0 indicates the first system qubit, not the ancilla.
        Returns:
            None
        '''
        self.flip_inds = [self.lattice.lat_to_mps_idx(flip_inds[i]) + 1 for i in range(len(flip_inds))]
        super().initialize_state(initial_state, coeffs, [0]+self.flip_inds)
        return
    
    
    def apply_trotter_layer(self, t:float) -> None:
        '''
        Applies a single layer of Trotter evolution on all links of the Heavy-Hex lattice according to the three coloring of the lattice.
        Parameters:
            t: Time to evolve by in Trotterized evolution
        Returns:
            None
        '''
        self.apply_two_q_gate_layer([expm(-1j*self.J*t*XX) for _ in range(len(self.layer1_links))], self.layer1_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*self.J*t*YY) for _ in range(len(self.layer1_links))], self.layer1_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*self.g*ZZ) for _ in range(len(self.layer1_links))], self.layer1_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*self.J*XX) for _ in range(len(self.layer2_links))], self.layer2_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*self.J*YY) for _ in range(len(self.layer2_links))], self.layer2_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*self.g*ZZ) for _ in range(len(self.layer2_links))], self.layer2_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*self.J*t*XX) for _ in range(len(self.layer3_links))], self.layer3_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*self.J*t*YY) for _ in range(len(self.layer3_links))], self.layer3_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*self.g*t*ZZ) for _ in range(len(self.layer3_links))], self.layer3_links, chi=self.chi_max)
        return
    
    def apply_controlled_state_prep(self, control_val:int, target_inds:list[int]|None = None, Us:list[np.ndarray]|None = None) -> None:
        '''
        Applies state preparation circuit consisting of CNOTs, controlled by ancilla value. 
        Assumes initial state is a layer of controlled unitaries applied to the all |0> state, controlled on the ancilla qubit
        Parameters:
            control_val: 0 or 1, indicating what value of the ancilla the state preparation circuit is conditioned on
            target_inds (optional): If specified, applies the controlled X gate to indices in this list, where the indices
                                    are in the MPS order (IMPORTANT). If not specified, the indices used in preparing the initial
                                    GHZ-like state are automatically used, which is likely the desired behaviour.
            Us (optional): list of 1 qubit unitaries (specified as 2X2 numpy arrays) to apply as conditional operations. Defaults to X
                            gates, i.e CNOTs.
        Returns:
            None
        '''
        if target_inds is None:
            target_inds = self.flip_inds
        if Us is None:
            Us = [X for _ in range(len(target_inds))]
        if control_val == 1:
            CUs = [np.kron(np.outer(zero,zero), I) + np.kron(np.outer(one,one), U) for U in Us]
        elif control_val == 0:
            CUs = [np.kron(np.outer(one,one), I) + np.kron(np.outer(zero,zero), U) for U in Us]
        self.apply_two_q_gate_layer(CUs, [(0, t) for t in target_inds], chi=self.chi_max)
        return

    def meas_hadamard_test(self, pauli_op_string:str, ancilla_op:str) -> list[complex]:
        '''
        Measures the X/Y expectation value of the ancilla together with the expectation value of a Pauli string operator
        on the system qubits.
        Parameters:
            pauli_op_string: String of 'X', 'Y', 'Z', 'I' denoting the Pauli operator on the system qubits.
            ancilla_op: Single character 'X', 'Y' denoting the ancilla Pauli for the Hadamard test.
        Returns:
            List of n_qubits + 1 expectation values of the associated ancilla + system Pauli operator string.
        '''
        op_string = (ancilla_op + pauli_op_string).lower()
        op_list = [*op_string]
        op_list = ['Sigma'+op_list[i] if op_list[i] != 'i' else 'Id' for i in range(len(op_list))]
        self.state.canonical_form()
        return self.state.expectation_value_multi_sites(op_list, 0)

    def get_pre_measure_states(self, krylov_dim:int, dt:float, trotter_order:int):
        '''
        Generates the Krylov subspace basis states from Trotterized time evolution. Note that this function can be re-run with a 
        larger krylov dimension without re-instantiating the class.
        Parameters:
            krylov_dim: Dimension of the Krylov subspace (number of basis vectors / time evolution steps). 
            dt: Time step for evolution between each Krylov space basis vector
            trotter_order: Order of Trotter approximation. Note that we do not use the approximation generated by the
                            full Lie-Trotter formula. Instead, we use a simplified version of higher-order Trotter approximations
                            where an nth order approximation consists of applying a Trotter layer evolving by time dt/n repeatedly 
                            for n iterations.
        Returns:
            A list of tenpy MPS objects that represent the time evolved states, i,e the Krylov basis vectors.
        '''
        t_evolve_step = dt/trotter_order
        if self.pre_measure_states is None:
            unique_t_evolved_states = [self.state.copy()]
            for i in range(krylov_dim-1):
                for n in range(trotter_order):
                    self.apply_trotter_layer(t_evolve_step)
                copy_state = self.state.copy()
                copy_state.canonical_form()
                unique_t_evolved_states.append(copy_state)
        else:
            unique_t_evolved_states = self.pre_measure_states
            self.state = self.pre_measure_states[-1]
            for i in range(krylov_dim-len(self.pre_measure_states)):
                for n in range(trotter_order):
                    self.apply_trotter_layer(t_evolve_step)
                copy_state = self.state.copy()
                copy_state.canonical_form()
                unique_t_evolved_states.append(copy_state)

        self.pre_measure_states = unique_t_evolved_states
        return unique_t_evolved_states
    

    def heisenberg_pauli_strings(self) -> tuple[list[str], list[complex]]:
        '''
        Generates all Pauli terms and associated coefficients in the Pauli decomposition of the 
        XXZ Hamiltonian. For the XXZ model, this follows from the definition of the Hamiltonian.
        Parameters:
            None
        Returns:
            op_strings: List of Pauli strings
            coeffs: List of associated coefficients
        '''
        op_strings = []
        coeffs = []
        for pair in self.layer1_links+self.layer2_links+self.layer3_links:
            char_list = ['X' if i in pair else 'I' for i in range(1,self.L)]
            op_strings.append(''.join(char_list))
            coeffs.append(self.J)
            char_list = ['Y' if i in pair else 'I' for i in range(1,self.L)]
            op_strings.append(''.join(char_list))
            coeffs.append(self.J)
            char_list = ['Z' if i in pair else 'I' for i in range(1,self.L)]
            op_strings.append(''.join(char_list))
            coeffs.append(self.g)
        return op_strings, coeffs


    def krylov_exp_vals(self, krylov_dim:int, dt:float, trotter_order:int, type:str='H'):
        '''
        Generates the first row of the Krylov H or S matrix by measuring expectation values of Pauli strings
        with the generalized Hadamard test.
        Implementation note: 1. This function destroys the internal quantum state in its current form.
                                2.  Can be used for increasing krylov dimensinos without re-initializing 
                                    the class as long as dt is not changed.
                                    3. Although the internal state is modified, the Krylov basis vectors are stored
                                    in the instance variable pre_measure_states (list of tenpy MPS).
        Parameters:
            krylov_dim: Dimension of Krylov space 
            dt: Time step for evolution between Krylov vectors
            trotter_order: Order of trotter approximation. See note above in 'get_pre_measure_states' function.
            type: Either 'H', or 'S' to denote which Matrix to compute.
        
        Returns:
            1-D complex array of expectation values that make up the first row of the H/S Toeplitz Matrix
            generated by the modified Hadamard test.
        '''
        if type == 'H':
            pauli_ops, coeffs = self.heisenberg_pauli_strings()
        elif type == 'S':
            pauli_ops = ['I'*(self.L-1)]
            coeffs = [1.0]
        if self.pre_measure_states is None:
            self.pre_measure_states = self.get_pre_measure_states(krylov_dim, dt, trotter_order)
        krylov_expvals = np.zeros(krylov_dim, dtype=np.complex128)
        for i in range(krylov_dim):
            self.state = self.pre_measure_states[i].copy()
            self.apply_controlled_state_prep(control_val=0)
            for j in range(len(pauli_ops)):
                krylov_expvals[i] += (self.meas_hadamard_test(pauli_ops[j], ancilla_op = 'X') 
                                       + 1j*self.meas_hadamard_test(pauli_ops[j], ancilla_op = 'Y'))*coeffs[j]        
        return krylov_expvals
    

    def krylov_H(self, krylov_dim:int=10, dt:float=0.1, trotter_order:int=2)->np.ndarray:    
        '''
        Generates the Krylov subspace Hamiltonian.
        Parameters:
            krylov_dim: Dimension of Krylov space 
            dt: Time step for evolution between Krylov vectors
            trotter_order: Order of trotter approximation. See note above in 'get_pre_measure_states' function.
        Returns:
            Toeplitz matrix denoting the Krylov subspace Hamiltonian generated by the modified Hadamard test.
        '''
        first_row_H = self.krylov_exp_vals(krylov_dim, dt, trotter_order, type='H')
        return toeplitz(first_row_H.conj())


    def krylov_S(self, krylov_dim:int=10, dt:float=0.1, trotter_order:int=2)->np.ndarray:    
        '''
        Generates the Krylov subspace metric (S).
        Parameters:
            krylov_dim: Dimension of Krylov space 
            dt: Time step for evolution between Krylov vectors
            trotter_order: Order of trotter approximation. See note above in 'get_pre_measure_states' function.
        Returns:
            Toeplitz matrix denoting the Krylov subspace metric matrix generated by the modified Hadamard test.
        '''
        first_row_S = self.krylov_exp_vals(krylov_dim, dt, trotter_order, type='S')
        return toeplitz(first_row_S.conj())
    

def solve_regularized_gen_eig(h: np.ndarray, s: np.ndarray, k:int=1, threshold:float=1e-15, return_vecs:bool=False) -> tuple[np.ndarray, np.ndarray]|np.ndarray:
    '''
    Solves the generalized eigenvalue problem h@v = lambda*s@v with regularization.
    Parameters:
        h: Krylov subspace Hamiltonian
        s: Krylov subspace Metric matrix
        k: Number of eigenvalues and / or eigenvectors to return (in ascending order of magnitude)
        threshold: Minimum norm of Krylov vector below which dimensions are projected onto the eigenvectors of the metric. Ensures that the subspace is well-conditioned.
        return_vecs: If True, also returns the corresponding eigenvector matrix whose columns are the associated eigenvectors.
    Returns:
        Array of eigenvalues in ascending order. If return_vecs, also returns matrix with columns as eigenvectors.
    '''
    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = [vec for val, vec in zip(s_vals, s_vecs) if val > threshold]
    if not good_vecs:
        raise AssertionError('WHOLE SUBSPACE ILL-CONDITIONED')
    good_vecs = np.array(good_vecs).T

    h_reg = good_vecs.conj().T @ h @ good_vecs
    s_reg = good_vecs.conj().T @ s @ good_vecs
    vals, vecs = sp.linalg.eigh(h_reg, s_reg)
    if return_vecs:
        return vals[0:k], good_vecs @ vecs[:,0:k]
    else:
        return vals[0:k]
    
