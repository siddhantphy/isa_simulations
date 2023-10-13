from netsquid.qubits.qubitapi import apply_pauli_noise, operate
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.components.instructions import IGate
from netsquid.qubits.operators import Operator
import numpy as np
from numpy import cos,sin



class DephasingNoiseModelCarbon(QuantumErrorModel):
    """
    A noise model that effectively applies multiple applications
    of dephasing in Z with a fixed probability.
    """

    def __init__(self, prob_of_dephasing):
        self.prob_of_dephasing = prob_of_dephasing

    def noise_operation(self, qubits, number_of_applications):
        # The probability of not-dephasing after a single
        # application of dephasing ...
        prob_single = 1. - self.prob_of_dephasing

        # ... and after multiple applications:
        prob_multiple = (1. + (2. * prob_single - 1) ** number_of_applications) / 2.

        # apply dephasing in Z
        for qubit in qubits:
            apply_pauli_noise(qubit, (prob_multiple, 0., 0., 1. - prob_multiple))
    
    def noise_operation_own(self,qubits,number_of_applications,diamond):
        parameter =0
        alpha_A = 0.03
        p_deph = (1 - alpha_A) / 2 * (1 - np.exp(-(parameter) ** 2 / 2))

        decoherence_value = self.network.noise_parameters["T2_carbon"]
        carbon_detuning_decoherence_based = 0 if self.network.noiseless == True else 1/(2*np.pi*decoherence_value)
        duration = number_of_applications # still wrong
        first_matrix_value = (cos(-2*np.pi*(A_parr_c+carbon_detuning_decoherence_based)*duration/2)+1j*sin(-2*np.pi*(A_parr_c+carbon_detuning_decoherence_based)*duration/2))

        second_matrix_value = (cos(-2*np.pi*(carbon_detuning_decoherence_based)*duration/2)+1j*sin(-2*np.pi*(carbon_detuning_decoherence_based)*duration/2))
        operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,second_matrix_value,0],[0,0,0,np.conj(second_matrix_value)]),dtype=np.complex_)

        op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_decoherence")

        INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)
        for i, qubit in enumerate(qubits):
            A_parr_c = diamond.mem_positions[i+1].properties["A_parr"]

            qubit.operate(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,i+1])	
