import netsquid as ns
from netsquid.qubits.qubitapi import *
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *
from netsquid.qubits.operators import *
from math import exp, cos, sin
import numpy as np
# a = np.array([[0,1],[1,0]])
# b = np.array([[1],[0]])
# answer = np.matmul(a,b)
# t = 1
# omega_1 = 0
# omega_L = np.pi/2
# zero_rotation = 8
# Omega = 0
# phi = 1.57
# # check = [t*-1j,0,0,0]
# np.set_printoptions(precision=1)

# first_matrix_value = (cos(-(omega_L-omega_1)*t/2)+1j*sin(-(omega_L-omega_1)*t/2))
# first_matrix_value_2 = (cos(-(omega_L-omega_1)*t/2)+1j*sin(-(omega_L-omega_1)*t/2))

# # first_value = np.array(([exp((omega_L-omega_1)*t*-1j/2),0,0,0]), dtype = np.complex_)
# # matrix_operator = [[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]] #a matrix operator which works like a cnot, but if the control qubit is in 0 state
# matrix_operator = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,cos(Omega*t/2),-(1j*cos(phi)+sin(phi))*(sin(Omega*t/2))],[0,0,-(1j*cos(phi)-sin(phi))*(sin(Omega*t/2)),cos(Omega*t/2)]),dtype=np.complex_)
# op = Operator(name = "Cnot_own", matrix = matrix_operator, description="Cnot_inversecontrol")
# q0,q1 = create_qubits(2)
# operate(q1,H)
# operate([q0,q1],op)
# print(matrix_operator)
# print(q0.qstate.qrepr)
# print(ns.qubits.reduced_dm(q0))
# print(ns.qubits.reduced_dm(q1))

upper_line = [0.5]+[0]*14+[0.5]
middle_line = [0]*16
upper_line_1 = [0.5]+[0]*14+[-0.5]
upper_line_2 = [-0.5]+[0]*14+[0.5]

dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
dz_perfect_1 = np.array([upper_line_1,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line_2]).astype(np.float64)
   

q0,q1,q2,q3 = create_qubits(4)
q_real1,q_real2,q_real3,q_real4 = create_qubits(4)

assign_qstate([q0,q1,q2,q3],dz_perfect_0)
assign_qstate([q_real1,q_real2,q_real3,q_real4],dz_perfect_1)
fidelity_value_modicum = fidelity([q_real1,q_real2,q_real3,q_real4],q0.qstate.qrepr)
print(fidelity_value_modicum)

