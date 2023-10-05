
# Math imports
from os import stat
from platform import node
import numpy as np

# utilities import
from itertools import product
import copy

# Plotting imports
import matplotlib.pyplot as plt

# Netsquid imports
import netsquid as ns
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor, NVSingleClickMagicDistributor
from netsquid.nodes import Node
from netsquid.qubits.ketstates import BellIndex
import netsquid.components.instructions as instr
from netsquid_nv.move_circuits import reverse_move_using_CXDirections # Note that this results into a Hadamrd being applied on the electron, so there is a change of basis
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid.qubits.qubitapi import reduced_dm, assign_qstate

# Local imports
# from q_programs import *
# from native_gates_and_parameters import add_native_gates
# from network_model import *

import time
timestr = time.strftime("%Y%m%d-%H%M%S")



KET_0 = np.array([[1], [0]])
KET_1 = np.array([[0], [1]])

KET_PLUS = (KET_0 + KET_1)/np.sqrt(2)
KET_MINUS = (KET_0 - KET_1)/np.sqrt(2)

KET_i_PLUS = (KET_0 + 1j * KET_1)/np.sqrt(2)
KET_i_MINUS = (KET_0 - 1j * KET_1)/np.sqrt(2)

IDENTITY = np.array([[1, 0],
                [0, 1]])
PAULI_X = np.array([[0, 1],
                [1, 0]])
PAULI_Y = np.array([[0, -1j],
                [1j, 0]])
PAULI_Z = np.array([[1, 0],
                [0, -1]])

PAULI_CONFIGS = {"I": IDENTITY, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}






def create_cardinal_states_distance_2():
    """ Create the vectors for logical states and matrices for logical Pauli operators for the distance-2 code. """

    ket_0L = (np.kron(np.kron(KET_0, KET_0) , np.kron(KET_0, KET_0)) + np.kron(np.kron(KET_1, KET_1) , np.kron(KET_1, KET_1)))/np.sqrt(2)
    ket_1L = (np.kron(np.kron(KET_0, KET_1) , np.kron(KET_0, KET_1)) + np.kron(np.kron(KET_1, KET_0) , np.kron(KET_1, KET_0)))/np.sqrt(2)

    ket_plus_L = (ket_0L + ket_1L)/np.sqrt(2)
    ket_minus_L = (ket_0L - ket_1L)/np.sqrt(2)

    ket_iplus_L = (ket_0L + 1j * ket_1L)/np.sqrt(2)
    ket_iminus_L = (ket_0L - 1j * ket_1L)/np.sqrt(2)

    i_L = np.outer(ket_0L, ket_0L) + np.outer(ket_1L, ket_1L)
    z_L = np.outer(ket_0L, ket_0L) - np.outer(ket_1L, ket_1L)
    x_l = np.outer(ket_0L, ket_1L) + np.outer(ket_1L, ket_0L)
    y_L = -1j * np.outer(ket_0L, ket_1L) + 1j* np.outer(ket_1L, ket_0L)

    return [ket_0L, ket_1L, ket_plus_L, ket_minus_L, ket_iplus_L, ket_iminus_L], [i_L, x_l, y_L, z_L]
_, logical_states_dist_2 = create_cardinal_states_distance_2()






def create_analytical_logical_PTM(node_A: Node, node_B, operation: str = "NA", iterations: int = 10, post_select: bool = True):
    """ Construct the Logical Pauli Transfer matrix (LPTM 4 X 4 matrix) using state tomography techniques in the codespace! """

    p_0 = get_analytical_logical_PTM_entries(node_A=node_A, node_B=node_B, input_state="0_L", operation=operation, iterations=iterations, post_select=post_select)
    p_1 = get_analytical_logical_PTM_entries(node_A=node_A, node_B=node_B, input_state="1_L", operation=operation, iterations=iterations, post_select=post_select)
    p_plus = get_analytical_logical_PTM_entries(node_A=node_A, node_B=node_B, input_state="+_L", operation=operation, iterations=iterations, post_select=post_select)
    p_minus = get_analytical_logical_PTM_entries(node_A=node_A, node_B=node_B, input_state="-_L", operation=operation, iterations=iterations, post_select=post_select)
    p_i_plus = get_analytical_logical_PTM_entries(node_A=node_A, node_B=node_B, input_state="+i_L", operation=operation, iterations=iterations, post_select=post_select)
    p_i_minus = get_analytical_logical_PTM_entries(node_A=node_A, node_B=node_B, input_state="-i_L", operation=operation, iterations=iterations, post_select=post_select)

    lptm = np.identity(4)

    lptm[0, 0] = 1
    lptm[0, 1] = lptm[0, 2] = lptm[0, 3] = 0
    lptm[1, 1] = 0.5 * (p_plus[0] - p_minus[0])
    lptm[2, 2] = 0.5 * (p_i_plus[1] - p_i_minus[1])
    lptm[3, 3] = 0.5 * (p_0[2] - p_1[2])
    lptm[1, 0] = 0.5 * (p_0[0] + p_1[0])
    lptm[2, 0] = 0.5 * (p_0[1] + p_1[1])
    lptm[3, 0] = 0.5 * (p_0[2] + p_1[2])
    lptm[2, 1] = 0.5 * (p_plus[1] - p_minus[1])
    lptm[3, 1] = 0.5 * (p_plus[2] - p_minus[2])
    lptm[1, 2] = 0.5 * (p_i_plus[0] - p_i_minus[0])
    lptm[3, 2] = 0.5 * (p_i_plus[2] - p_i_minus[2])
    lptm[1, 3] = 0.5 * (p_0[0] - p_1[0])
    lptm[2, 3] = 0.5 * (p_0[1] - p_1[1])

    return lptm
    
    
    
    
    
    
    
def get_analytical_logical_PTM_entries(node_A: Node, node_B, input_state: str = "NA", operation: str = "NA", iterations: int = 10, post_select: bool = True):
    """ Get the average entries for LPTM by doing muktiple ierations. Functionality for post-selection whe needed! """
    p = [0, 0, 0]
    trashed = 0
    for trial in range(iterations):
        ### already performed in txt file
        # perform_first_stabilizer_measurements(node_A=node_A, node_B=node_B, state=input_state)
        # apply_logical_operation(node_A=node_A, node_B=node_B, operation=operation)
        # meas_res = perform_all_stabilizers(node_A=node_A, node_B=node_B)
        ### end of stabilizer, already done in txt file
        
        meas_res = 0 #values needed from the txt file
        if post_select == True:
            if (meas_res[0]==0 and meas_res[1]==0) and (meas_res[2]==meas_res[3]):
                r_logical = get_analytical_logical_expectation_values(node_A=node_A, node_B=node_B)
                p[0] += r_logical[0]
                p[1] += r_logical[1]
                p[2] += r_logical[2]
            else:
                trashed += 1
        else:
            r_logical = get_analytical_logical_expectation_values(node_A=node_A, node_B=node_B)
            p[0] += r_logical[0]
            p[1] += r_logical[1]
            p[2] += r_logical[2]
    p[0] = p[0]/(iterations - trashed)
    p[1] = p[1]/(iterations - trashed)
    p[2] = p[2]/(iterations - trashed)

    return p





def get_analytical_logical_expectation_values(node_A: Node, node_B: Node):
    """ To calculate all the expectation values {I_L, X_L, Y_L, Z_L} logical Pauli operatos. """

    r_logical = [0, 0, 0]
    #rho logical will be taken from our txt file
    rho_logical = 0
    # rho_logical = get_instantaneous_data_qubit_density_matrix([node_A, node_B])

    # print(np.real(np.trace(logical_states_dist_2[0] @ rho_logical)))
    r_logical[0] = np.trace(logical_states_dist_2[1] @ rho_logical)
    r_logical[1] = np.trace(logical_states_dist_2[2] @ rho_logical)
    r_logical[2] = np.trace(logical_states_dist_2[3] @ rho_logical)

    return r_logical









