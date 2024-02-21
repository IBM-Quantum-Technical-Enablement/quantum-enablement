from copy import deepcopy

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass


class Local2qAmplifier(TransformationPass):
    def __init__(self, noise_factor: int, two_q_gate_name: str):
        self.noise_factor = noise_factor
        self.two_q_gate_name = two_q_gate_name
        super().__init__()
    
    def run(self, dag: DAGCircuit):
        for run in dag.collect_runs([self.two_q_gate_name]):
            for node in run:
                fold_dag = DAGCircuit()
                qreg = QuantumRegister(2)
                fold_dag.add_qreg(qreg=qreg)
                for _ in range(self.noise_factor):
                    fold_dag.apply_operation_back(node.op, [qreg[0], qreg[1]])
                    fold_dag.apply_operation_back(node.op, [qreg[0], qreg[1]])
                
                dag.substitute_node_with_dag(node, fold_dag)

        return dag


class GlobalAmplifier(TransformationPass):
    def __init__(self, noise_factor: int):
        self.noise_factor = noise_factor
        super().__init__()
    
    def run(self, dag: DAGCircuit):
        fold_dag = deepcopy(dag)

        for _ in range(self.noise_factor//2):
            fold_dag.compose(dag.reverse_ops())
            fold_dag.compose(dag)

        return fold_dag
    