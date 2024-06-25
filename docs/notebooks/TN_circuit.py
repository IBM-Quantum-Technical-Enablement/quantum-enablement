import numpy as np
import scipy as sp
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite
import tenpy.linalg.np_conserved as npc
from scipy.linalg import block_diag


Id = np.eye(2)

class TNCircuit():

    def __init__(self, mps_order: list | np.ndarray, conserve : str = 'None', chi_max : int = 100):
        
        assert set(mps_order) == set(range(len(mps_order))) and len(mps_order) == len(range(max(mps_order)+1))
        
        self.L = len(mps_order)
        self.mps_order = mps_order
        self.qubit_to_mps_ordering = {i:mps_order[i] for i in range(len(mps_order))}
        self.mps_to_qubit_ordering = {v:k for (k,v) in self.qubit_to_mps_ordering.items()} 
        self.lat = Chain(self.L, SpinHalfSite(conserve=conserve), bc='open', bc_MPS='finite')
        self.state = None
        self.chi_max = chi_max
        return 
    
    def map_to_mps_ind(self, inds):
        if type(inds) is int:
            return self.qubit_to_mps_ordering[inds]
        else:
            return [self.qubit_to_mps_ordering[i] for i in inds]

    def map_to_qubit_ind(self, inds):
        if type(inds) is int:
            return self.mps_to_qubit_ordering[inds]
        else:
            return [self.mps_to_qubit_ordering[i] for i in inds]
    
    def initialize_state(self, initial_state: list | int | str = 0, coeffs : list = None) -> None:
        
        if type(initial_state) is list:
            product_state = [state_int_to_str(i) for i in initial_state]
        elif initial_state in [0, 'GHZ']:
            product_state = ['up' for _ in range(self.L)]
        elif initial_state == 1:
            product_state = ['down' for _ in range(self.L)]
        
        psi = MPS.from_product_state(self.lat.mps_sites(), product_state, self.lat.bc_MPS)

        if initial_state == 'GHZ':
            psi2 = MPS.from_product_state(self.lat.mps_sites(), ['down' for _ in range(self.L)], self.lat.bc_MPS)
            if coeffs is None:
                coeffs = [1/np.sqrt(2), 1/np.sqrt(2)]
            psi.add(psi2, alpha=coeffs[0], beta=coeffs[1])
        self.state = psi
        return
    
    def apply_single_q_gate(self, qubit: int, gate: np.ndarray | str) -> None:
        qubit = self.map_to_qubit_ind(qubit)
        if type(gate) is str:
            assert gate in ['x', 'y', 'z']
            op = 'Sigma'+gate
        else:
            op = npc.Array.from_ndarray_trivial(gate, labels=['p', 'p*'])
        self.state.apply_local_op(qubit, op)
        return
        
    def apply_controlled_gates(self, controls: int, targets: int, gates: list | np.ndarray) -> None:
        controls = self.map_to_mps_ind(controls)
        targets = self.map_to_mps_ind(targets)
        if type(controls) is int:
            controls = [controls]
        if type(targets) is int:
            targets = [targets]
        if type(gates) is not list:
            gates = [gates]
        
        intervals = list(zip(controls, targets))
        if check_overlap(intervals):
            Exception('Invalid controls and targets: control-target intervals must be non-overlapping for a single MPO layer.')
        CU_MPO = CU_MPO_layer(self.lat.mps_sites(), gates, controls, targets)
        CU_MPO.apply(self.state, {'compression_method':'SVD', 'trunc_params':{'chi_max':self.chi_max}})
        return
    
    #TODO
    def time_evolve_TEBD():
        return
    
    #TODO
    def time_evolve_W2():
        return
    
    #must include idenntity operators in between non-contiguous strings
    def meas_product_op_expectation_val(self, left_ind, ops):
        left_ind = self.map_to_mps_ind(left_ind)
        assert all([type(ops[0]) is type(ops[i]) for i in range(1,len(ops))])
        if type(ops[0]) is not str:
            ops = [npc.Array.from_ndarray_trivial(ops[i], labels=['p', 'p*']) for i in range(len(ops))]
        return complex(self.state.expectation_value_multi_sites(ops, left_ind))


def check_overlap(intervals):
    # Sort intervals by their start points
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    # Iterate through sorted intervals and check for overlaps
    for i in range(1, len(sorted_intervals)):
        if sorted_intervals[i][0] < sorted_intervals[i-1][1]:
            # Overlap found
            return True
    # If no overlaps found
    return False

def state_int_to_str(i):
    assert i in [0,1]
    if i == 0:
        return 'up'
    else:
        return 'down'


#only works for non-local CUs with non-overlapping identity strings in betweem
def CU_MPO_layer(sites, Us, controls, targets):
    I_tensor = np.eye(4).reshape(2,2,2,2).swapaxes(1,2)
    CUs = [block_diag(Id, Us[i]) for i in range(len(Us))]
    L = len(sites) 
    Ws = [None for _ in range(L)]
    first = min(controls+targets)
    second = max(controls+targets)
    for i in range(first):
        Ws[i] = (npc.Array.from_ndarray_trivial(Id[None,None,:,:], labels=['wL', 'wR', 'p', 'p*']))
    seen = []
    for n, (control, target) in enumerate(zip(controls, targets)):
        #cin,tin,cout,tout
        CU = np.reshape(CUs[n], (2,2,2,2))
        if control < target:
            #cin, cout, tin, tout -> c,t
            CU = np.reshape(np.transpose(CU,[0,2,1,3]), (4,4))
        else:
            #tin, tout, cin, cout -> t,c
            CU = np.reshape(np.transpose(CU,[1,3,0,2]), (4,4))
        u,s,vh = sp.linalg.svd(CU, full_matrices=False)
        sprime = np.trim_zeros(s)
        uprime = u[:,:len(sprime)]
        vhprime = vh[:len(sprime),:]
        
        #cin, cout, bondr
        A = np.reshape(uprime@np.diag(sprime), (2,2,len(sprime)))
        A = np.transpose(A, [2,0,1]) #bondr, cin, cout
        A = A[None, :, : , :] #bondl, bondr, cin, cout
        
        #bondl, tin, tout
        B = np.reshape(vhprime, (len(sprime),2,2))
        B = B[:,None,:,:] #bondl, bondr, tin, tout
        Ws[min(control,target)] = (npc.Array.from_ndarray_trivial(A, labels=['wL', 'wR', 'p', 'p*']))
        Ws[max(control,target)] = (npc.Array.from_ndarray_trivial(B, labels=['wL', 'wR', 'p', 'p*']))
        seen.append(control)
        seen.append(target)
        for i in range(min(control,target)+1, max(control,target)):
            Ws[i] = (npc.Array.from_ndarray_trivial(I_tensor, labels=['wL', 'wR', 'p', 'p*']))
            seen.append(i)
    for i in range(second+1, L):
        Ws[i] = (npc.Array.from_ndarray_trivial(Id[None,None,:,:], labels=['wL', 'wR', 'p', 'p*']))

    remaining = set(range(first, second+1)) - set(seen)
    for i in remaining:
        Ws[i] = (npc.Array.from_ndarray_trivial(Id[None,None,:,:], labels=['wL', 'wR', 'p', 'p*']))
    return MPO(sites, Ws, 'finite', 0, -1)