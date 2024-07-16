from quspin.operators import hamiltonian 
from quspin.basis import spin_basis_1d 
import numpy as np 
from scipy.sparse.linalg import expm, eigsh
from scipy import sparse
import scipy as sp

def XXZ_sparse_block_hamiltonian(L, J=1.0, g=0.5):
    blocks = []
    for m in range(L+1):
        for k in range(L):
            basis=spin_basis_1d(L=L, a=1, kblock=k, Nup=m)
            # define PBC site-coupling lists for operators
            ZZ =[[g,i,(i+1)%L] for i in range(L)]
            XXYY = [[J,i,(i+1)%L] for i in range(L)] # PBC
            # static and dynamic lists
            static=[["xx", XXYY], ["yy", XXYY], ['zz', ZZ]]
            dynamic=[]
            ###### construct Hamiltonian
            if basis.Ns > 0:
                H=hamiltonian(static,dynamic,static_fmt="dia",dtype=np.complex128,basis=basis).tocsr()
                blocks.append(H)
    return blocks

def get_basis_dims(L, return_empty=False):
    dims = []
    for m in range(L+1):
        for k in range(L):
            d = spin_basis_1d(L=L, a=1, kblock=k, Nup=m).Ns
            if return_empty:
                dims.append(d)
            else:
                if d>0:
                    dims.append(d)
    return dims

def XXZ_sparse_block_hamiltonian_trotter(L, J=1.0, g=0.5):
    blocks = []
    for m in range(L+1):
        for k in range(L):
            basis=spin_basis_1d(L=L, a=1, kblock=k, Nup=m)
            # define PBC site-coupling lists for operators
            ZZ =[[g,i,(i+1)%L] for i in range(L)]
            XXYY = [[J,i,(i+1)%L] for i in range(L)] # PBC
            # static and dynamic lists
            static1=[["zz",ZZ]]
            static2=[["yy",XXYY], ["xx",XXYY]]
            dynamic=[]
            ###### construct Hamiltonian
            if basis.Ns > 0:
                H1 =hamiltonian(static1, dynamic, static_fmt="dia", dtype=np.complex128, basis=basis).tocsr()
                H2 =hamiltonian(static2, dynamic, static_fmt="dia", dtype=np.complex128, basis=basis).tocsr()
                blocks.append([H1, H2])
    return blocks

def exact_ground_state(H):
    e_min = np.inf
    for h in H:
        if h.size>1:
            evals = eigsh(h,k=1,which='SA',return_eigenvectors=False)
        else:
            evals = [h[0,0]]
        if evals[0] < e_min:
            e_min = evals[0]
    return e_min

def block_unitaries(t, blocks):
    return [expm(-1j*t*b) for b in blocks]

def trotter_block_unitaries(t, N_steps, trotter_blocks, order=2):
    dt = t/N_steps
    block_unitaries = []
    for H_list in trotter_blocks:
        U = 1.0
        evolved_t = 0
        while evolved_t < t:
            if isinstance(U, (int, float)) or U.size == 1:
                U = trotter_approx(H_list, dt, order)*U
            else:
                U = trotter_approx(H_list, dt, order)@U
            evolved_t += dt
        if isinstance(U, (int, float)):
            U = sparse.coo_array(np.array([[U]]), dtype=np.complex128).tocsr()
        block_unitaries.append(U)
    return block_unitaries

def trotter_approx(H_list, t, order):
    assert order >= 2 and order %2 == 0
    if order == 2:
        prod = 1.0
        for h in H_list:
            if h.size != 0:
                if isinstance(prod, (int, float)) or prod.size == 1:
                    prod = expm(-1j*t/2*h)*prod
                else:
                    prod = expm(-1j*t/2*h)@prod
        for h in H_list[::-1]:
            if h.size != 0:
                if isinstance(prod, (int, float)) or prod.size == 1:
                    prod = expm(-1j*t/2*h)*prod
                else:
                    prod = expm(-1j*t/2*h)@prod
        return prod
    else:
        uk = 1/(4-4**(1/(order-1)))
        lower_approx1 = trotter_approx(H_list, uk*t, order-2)
        lower_approx2 = trotter_approx(H_list, (1-4*uk)*t, order-2)
    return lower_approx1@lower_approx1@lower_approx2@lower_approx1@lower_approx1

def generate_random_state(dims):
    n = sum(dims)
    real_parts = np.random.randn(n)
    imag_parts = np.random.randn(n)
    coeffs = []
    for i in range(n):
        z = complex(real_parts[i], imag_parts[i])
        coeffs.append(z)
    norm = sum(np.abs(z)**2 for z in coeffs)
    normalized_coeffs = [z / np.sqrt(norm) for z in coeffs]
    start = 0
    block_coeffs = []
    for i,d in enumerate(dims):
        block_coeffs.append(np.array(normalized_coeffs[start:start+d], dtype=np.complex128))
        start += d
    return block_coeffs

def matrix_element_blocks(state1_blocks, state2_blocks, op_blocks):
    assert len(state1_blocks) == len(state2_blocks) == len(op_blocks)
    tot = 0
    num_blocks = len(state1_blocks)
    for i in range(num_blocks):
        assert len(state1_blocks[i]) == len(state2_blocks[i]) == op_blocks[i].shape[0] == op_blocks[i].shape[1]
        tot += (state2_blocks[i].conj()[:,None].T@op_blocks[i]@state1_blocks[i])
    return tot

def trotter_fidelity(ideal_unitary, trotter_unitary, state):
    op_blocks = []
    for i in range(len(ideal_unitary)):
        op_blocks.append(ideal_unitary[i].toarray().conj().T@trotter_unitary[i].toarray())
    return np.abs(matrix_element_blocks(state, state, op_blocks))**2


def krylov_matrices(psi0, H, num_states, dt, time_evolution='exact', Htrotter_terms = None, trotter_order = 2, trotter_steps = 10):
    assert time_evolution in ['trotter', 'exact']
    Hkrylov = np.zeros((num_states, num_states), dtype=np.complex128)
    S = np.zeros((num_states, num_states), dtype=np.complex128)

    if time_evolution == 'trotter':
        assert isinstance(trotter_order, int) and trotter_order >= 2 and trotter_order%2 == 0
        assert isinstance(trotter_steps, int)
        assert Htrotter_terms is not None
        Us = [trotter_block_unitaries(j*dt, trotter_steps, Htrotter_terms, trotter_order) for j in range(1, num_states+1)]
    else:
        Us = [block_unitaries(j*dt, H) for j in range(1, num_states+1)]
    
    for j in range(num_states):
        U1 = Us[j]
        for k in range(j+1):
            U2 = Us[k]
            assert len(U1) == len(U2) == len(H) == len(psi0)
            for i in range(len(U1)):
                Hkrylov[j,k] += psi0[i].conj()[:,None].T@U1[i].toarray()@H[i].toarray()@U2[i].toarray().conj().T@psi0[i]
                S[j,k] += psi0[i].conj()[:,None].T@U1[i].toarray()@U2[i].toarray().conj().T@psi0[i]
            if k != j:
                Hkrylov[k,j] = np.conj(Hkrylov[j,k])
                S[k,j] = np.conj(S[j,k])
    return Hkrylov, S


def solve_regularized_gen_eig(h, s, k=1, threshold=1e-15, return_vecs=False):

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

def krylov_diagonalization(Hkrylov, Skrylov, k=1, initial_threshold = 1e-15):
    success = False
    threshold = initial_threshold
    count = 0
    while not success and count < 16:
        try:
            evals = solve_regularized_gen_eig(Hkrylov, Skrylov, threshold=threshold, k=k)
            success = True
        except:
            threshold *= 10
            count += 1
    if not success:
        print('Did not converge')
    return evals, threshold

def spectral_norm(H):
    sing_max = -np.inf
    for h in H:
        u,s,vh = np.linalg.svd(h.toarray(), full_matrices=False)
        if max(s) > sing_max:
            sing_max = max(s)
    return sing_max




#def XXZ_initial_guess() q