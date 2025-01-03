
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
    x_L = np.outer(ket_0L, ket_1L) + np.outer(ket_1L, ket_0L)
    y_L = -1j * np.outer(ket_0L, ket_1L) + 1j* np.outer(ket_1L, ket_0L)

    return [ket_0L, ket_1L, ket_plus_L, ket_minus_L, ket_iplus_L, ket_iminus_L], [i_L, x_L, y_L, z_L]


def get_analytical_logical_expectation_values(qubit_matrix = None,operation_space = "logical"):
    """ To calculate all the expectation values {I_L, X_L, Y_L, Z_L} logical Pauli operatos. """

    r_logical = [0, 0, 0]
    #rho logical will be taken from our txt file
    rho_logical = qubit_matrix
    # rho_logical = get_instantaneous_data_qubit_density_matrix([node_A, node_B])
    I = [[1,0],[0,1]]
    X = [[0,1],[1,0]]
    Y = [[0,-1j],[1j,0]]
    Z = [[1,0],[0,-1]]


    if operation_space == "physical":
        logical_states_dist_2 = [I,X,Y,Z]
    else:
        _, logical_states_dist_2 = create_cardinal_states_distance_2()

    # print(np.real(np.trace(logical_states_dist_2[0] @ rho_logical)))
    r_logical[0] = np.trace(logical_states_dist_2[1] @ rho_logical)
    r_logical[1] = np.trace(logical_states_dist_2[2] @ rho_logical)
    r_logical[2] = np.trace(logical_states_dist_2[3] @ rho_logical)

    return r_logical









