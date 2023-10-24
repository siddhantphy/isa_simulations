from netsquid.qubits.qubitapi import apply_pauli_noise, operate
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.components.instructions import IGate
from netsquid.qubits.operators import Operator
from netsquid.qubits.qubitapi import operate, create_qubits, assign_qstate,reduced_dm
import numpy as np
import random
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
        # parameter =0
        alpha_A = 0.03
        # p_deph = (1 - alpha_A) / 2 * (1 - np.exp(-(parameter) ** 2 / 2))
        # network = diamond.supercomponent.supercomponent
        # print('the value is')
        # print(decoherence_value)
        # duration = number_of_applications # still wrong
        sample_values = np.arange(0,600e-9, 10e-9)
        tc = 300e-9
        exponantional_decay = 1-np.exp(-sample_values/tc)
        prob_dist_exp = [0]+[exponantional_decay[i+1]-exponantional_decay[i] for i in range(len(exponantional_decay)-1) ]
        time_duration_until_excitation = 0
        time_duration_after_excitation = 0
        for i in range(number_of_applications):
            current_time_until_excitation = random.choices(sample_values,prob_dist_exp)[0]
            # print(current_time_until_excitation)
            time_duration_until_excitation += current_time_until_excitation
            time_duration_after_excitation += 600e-9-current_time_until_excitation
        # t = duration
        # tc = 300e-9
        # exponantional_decay = 1-np.exp(-t/tc)
        Beta_A = np.sqrt(1-alpha_A**2)
        q0 = create_qubits(1)
        qubit_state = np.array([alpha_A,Beta_A])
        assign_qstate(q0,qubit_state)
        # electron = diamond.peek(0)[0]
        self._noise_execution(q0[0],qubits,diamond,time_duration_until_excitation)
        qubit_state = np.array([1,0])
        assign_qstate(q0,qubit_state)
        self._noise_execution(q0[0],qubits,diamond,time_duration_after_excitation)

        
    def _noise_execution(self,control_qubit,qubits,diamond,duration):
        network = diamond.supercomponent.supercomponent
        decoherence_value = network.noise_parameters["T2_carbon"]
        print("am i here?")
        carbon_detuning_decoherence_based = 0 if (network.noiseless == True or decoherence_value == 0) else 1/(2*np.pi*decoherence_value)
        for i, qubit in enumerate(qubits):
            A_parr_c = diamond.mem_positions[i+1].properties["A_parr"]

            first_matrix_value = (cos(-2*np.pi*(A_parr_c+carbon_detuning_decoherence_based)*duration/2)+1j*sin(-2*np.pi*(A_parr_c+carbon_detuning_decoherence_based)*duration/2))

            second_matrix_value = (cos(-2*np.pi*(carbon_detuning_decoherence_based)*duration/2)+1j*sin(-2*np.pi*(carbon_detuning_decoherence_based)*duration/2))
            operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,second_matrix_value,0],[0,0,0,np.conj(second_matrix_value)]),dtype=np.complex_)

            op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_decoherence")

            # INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)
            # print(f"qubit value {qubit}")
            # print(f"electron is {electron}")
            # qubit[0].operate(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,i+1])	
            # print(control_qubit)
            # print(qubit[0])
            operate([control_qubit,qubit[0]],op)
            # last thing to add is an operation which counter acts the phase that has been gained