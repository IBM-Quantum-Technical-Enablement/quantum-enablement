import numpy as np
import scipy as sp
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.truncation import svd_theta
from scipy.linalg import block_diag


Id = np.eye(2)

class TNCircuit():

    def __init__(self, mps_order: list | np.ndarray, conserve : str = 'None', chi_max : int = 1028):
        
        assert set(mps_order) == set(range(len(mps_order))) and len(mps_order) == len(range(max(mps_order)+1))
        
        self.L = len(mps_order)
        self.mps_order = mps_order
        self.conserve = conserve
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
    
    def initialize_state(self, initial_state: list | int | str = 0, coeffs = None, flip_inds = None) -> None:
        
        if type(initial_state) is list:
            product_state = [state_int_to_str(i) for i in initial_state]
        elif initial_state in [0, 'GHZ']:
            product_state = ['up' for _ in range(self.L)]
        elif initial_state == 1:
            product_state = ['down' for _ in range(self.L)]
        
        psi = MPS.from_product_state(self.lat.mps_sites(), product_state, self.lat.bc_MPS)
        if initial_state == 'GHZ':
            if flip_inds is None:
                psi2 = MPS.from_product_state(self.lat.mps_sites(), ['down' for _ in range(self.L)], self.lat.bc_MPS)
            else:
                psi2 = MPS.from_product_state(self.lat.mps_sites(), ['down' if i in flip_inds else 'up' for i in range(self.L) ], self.lat.bc_MPS)
            if coeffs is None:
                coeffs = [1/np.sqrt(2), 1/np.sqrt(2)]
            psi = psi.add(psi2, alpha=coeffs[0], beta=coeffs[1])
        psi.canonical_form()
        self.state = psi
        return
    
    def apply_single_q_gate(self, qubit: int, gate: np.ndarray | str) -> None:
        qubit = self.map_to_qubit_ind(qubit)
        if type(gate) is str:
            assert gate in ['x', 'y', 'z']
            op = 'Sigma'+gate
        else:
            if self.conserve == 'Sz':
                chinfo = npc.ChargeInfo([1], ['2*Sz'])
                leg_charge_p1 = npc.LegCharge.from_qflat(chinfo, [[-1],[1]], qconj=1,)
                leg_charge_p2 = npc.LegCharge.from_qflat(chinfo, [[-1], [1]], qconj=-1)
                op = npc.Array.from_ndarray(op, [leg_charge_p1, leg_charge_p2], labels=['p', 'p*'])
            else:
                op = npc.Array.from_ndarray_trivial(gate, labels=['p', 'p*'])
        self.state.apply_local_op(qubit, op)
        return

    def apply_two_q_gate_layer(self, Us, inds, compression='SVD', chi=1028):
        remaining_inds = inds
        remaining_Us = Us
        while len(remaining_inds) > 0:
            intervals_no_overlap, U_inds = maxDisjointIntervals(remaining_inds)
            Us_no_overlap = [remaining_Us[i] for i in U_inds]
            two_q_layer = two_q_gate_mpo(self.state.sites, Us_no_overlap, intervals_no_overlap, self.conserve)
            two_q_layer.apply(self.state, {'compression_method':compression, 'trunc_params':{'chi_max':chi}})
            remaining_inds = [i for j, i in enumerate(remaining_inds) if j not in U_inds]
            remaining_Us = [i for j, i in enumerate(remaining_Us) if j not in U_inds]
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


def two_q_gate_mpo(sites, gates, inds: list[tuple], conserve=None):
    if conserve == 'Sz':
        chinfo = npc.ChargeInfo([1], ['2*Sz'])
        leg_charge_p1 = npc.LegCharge.from_qflat(chinfo, [[-1],[1]], qconj=1,)
        leg_charge_p2 = npc.LegCharge.from_qflat(chinfo, [[-1], [1]], qconj=-1)
        leg_charge_left = npc.LegCharge.from_qflat(chinfo, [[0]], qconj=1)

    L = len(sites) 
    Ws = [None for _ in range(L)]
    first = min(list(sum(inds, ())))
    for i in range(first):
        if conserve == 'Sz':
            if i == 0:
                charge_idL = npc.detect_legcharge(Id[None,None,:,:],chinfo,[leg_charge_left, None, leg_charge_p1, leg_charge_p2], qconj=-1)
            else:
                charge_idL = npc.detect_legcharge(Id[None,None,:,:],chinfo,[Ws[i-1].legs[1].conj(), None, leg_charge_p1, leg_charge_p2], qconj=-1)
            Ws[i] = npc.Array.from_ndarray(Id[None,None,:,:], charge_idL, labels=['wL', 'wR', 'p', 'p*'])
        else:
            Ws[i] = npc.Array.from_ndarray_trivial(Id[None,None,:,:], labels=['wL', 'wR', 'p', 'p*'],)
    for n, (ind1, ind2) in enumerate(inds):
        U = np.reshape(gates[n],(2,2,2,2))
        U = np.transpose(U,[0,2,1,3])
        if conserve == 'Sz':
            U = npc.Array.from_ndarray(U, [leg_charge_p1, leg_charge_p2, leg_charge_p1, leg_charge_p2], labels=['p1','p1*','p2','p2*'])
        else:
            U = npc.Array.from_ndarray_trivial(U, labels=['p1','p1*','p2','p2*'])
        U = U.combine_legs([[0,1],[2,3]], qconj=[+1,-1])
        uprime,sprime,vhprime, trunc_err, renormalization = svd_theta(U,inner_labels=['wR', 'wL'], trunc_par={'svd_min':1e-12})
        
        ulabels = ['p', 'p*', 'wR']
        vlabels = ['wL','p', 'p*']
        
        U_new = uprime.split_legs(0).iset_leg_labels(ulabels)
        V_new = vhprime.split_legs(1).iset_leg_labels(vlabels)
        S = renormalization * sprime
        US = U_new.iscale_axis(S, 'wR')
                
        A = US.add_trivial_leg(0, 'wL')
        A.itranspose(['wL', 'wR', 'p', 'p*'])
        B = V_new.add_trivial_leg(-1, 'wR')
        B.itranspose(['wL', 'wR', 'p', 'p*'])

        assert A.shape[1] == B.shape[0] 
        bond_dim = A.shape[1]
        I_tensor = (np.kron(np.eye(bond_dim), np.eye(2))).reshape(bond_dim,2,bond_dim,2).swapaxes(1,2)
        
        Ws[ind1] = A
    
        for i in range(ind1+1, ind2):
            if conserve == 'Sz':
                charge_Itensor = npc.detect_legcharge(I_tensor,chinfo,[Ws[i-1].legs[1].conj(), None, leg_charge_p1, leg_charge_p2], qconj=-1)
                Ws[i] = npc.Array.from_ndarray(I_tensor, charge_Itensor, labels=['wL', 'wR', 'p', 'p*'])
            else:
                Ws[i] = (npc.Array.from_ndarray_trivial(I_tensor, labels=['wL', 'wR', 'p', 'p*']))
        
        Ws[ind2] = B

        if ind2 != len(sites)-1:
            if n < len(inds)-1:
                next = inds[n+1][0]
            else:
                next = len(sites)
            for i in range(ind2+1, next):
                if conserve == 'Sz':
                    charge_Idmid = npc.detect_legcharge(Id[None,None,:,:],chinfo,[Ws[i-1].legs[1].conj(), None, leg_charge_p1, leg_charge_p2], qconj=-1)
                    Ws[i] = npc.Array.from_ndarray(Id[None,None,:,:], charge_Idmid, labels=['wL', 'wR', 'p', 'p*'])
                else:
                    Ws[i] = npc.Array.from_ndarray_trivial(Id[None,None,:,:], labels=['wL', 'wR', 'p', 'p*'])
    return MPO(sites, Ws, 'finite', 0, -1)


def maxDisjointIntervals(intervals):

    intervals.sort(key = lambda x: x[1])
    return_intervals = [intervals[0]]
    return_inds = [0]   
    r1 = intervals[0][1]
     
    for i in range(1, len(intervals)):
        l1 = intervals[i][0]
        r2 = intervals[i][1]
         
        if l1 > r1:
            return_intervals.append(intervals[i])
            return_inds.append(i)
            r1 = r2
    return return_intervals, return_inds
