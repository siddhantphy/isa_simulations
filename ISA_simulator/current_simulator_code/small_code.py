# from netsquid.qubits.qubitapi import *
# import netsquid as ns
# from netsquid.components import Component
# from netsquid.qubits import operators as ops

# from netsquid.qubits.operators import *
# from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
# import numpy as np
# from netsquid.protocols import NodeProtocol, Protocol
# import matplotlib.pyplot as plt
import netsquid as ns
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.components.models.qerrormodels import *

# set_qstate_formalism(QFormalism.DM)

# q1, q2, q3, q4 = create_qubits(num_qubits = 4)
# # assign_qstate(q2, np.diag([0.5,0.5]))
# # assign_qstate(q1, np.diag([0.5,0.5]))
# # operate(q1,H)
# # operate([q1, q2], CNOT)
# # print(q1.qstate.qrepr.reduced_dm())
# # operate(q2,H)
# # print(q2.qstate.qrepr)
# # operate()
# operate(q1,H)
# operate([q1,q2],CNOT)
# # operate([q2,q3],CNOT)
# operate(q4,H)
# operate([q4,q3],CNOT)
# combine_qubits([q2,q3])
# print(q1.qstate.qrepr)
# print(reduced_dm([q2,q3]))
# print(reduced_dm(q2))
# print(q2.qstate.qrepr.reduced_dm(1))
# result_2, prob = measure(q2, keep_combined=True)
# result_3, prob = measure(q3, keep_combined=True)
# print(q1.qstate.qrepr)
# print(f"results are {result_2,result_3}")
# if result_3 == 0 and result_2 == 1:
# 	operate(q3,H)
# 	operate(q2,H)
# if result_2 == 0 and result_3 == 1:
# 	operate(q2,H)
# 	operate(q3,H)
# print(q1.qstate.qrepr)
# measure(q2)
# measure(q3)
# # discard(q2)
# # discard(q3)
# # measure(q3)
# # print(q3.qstate.qrepr)

# # print(q2.qstate.qrepr.reduced_dm())
# # print(q3.qstate.qrepr.reduced_dm())
# print(q1.qstate.qrepr.reduced_dm())
# print(q4.qstate.qrepr.reduced_dm())
# print(q1.qstate.qrepr)

# q2.frequency = 100
# qubit = create_qubits(1)
# print(q2.qstate.qrepr)
# print("split")
# print(q2.qstate.qrepr.reduced_dm())

# # print(q1.qstate.qrepr)
# print(q2.frequency)
# q2.properties = {"frequency": 100}
# # print(q2.properties)
# q3 = create_qubits(1)
# # print(dir(q3))
# class PhotodetectorProtocol(Protocol):
# 	def __init__(self, photodetector = None, p=0.95, between_nodes = None):
# 		super().__init__()


# qubit = PhotodetectorProtocol()
# print(dir(qubit))

# printerlist = [0, 0, 0, 8, 45, 100, 41, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 51, 100, 50, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# x = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500]
# y = [2988, 2111, 1266, 570, 154, 0, 141, 616, 1218, 2050, 3079, 3993, 4884, 5569, 5896, 6100, 5875, 5524, 4858, 4052, 3057, 2003, 1288, 579, 160, 0, 144, 553, 1247, 2100, 3070, 4040, 4826, 5522, 5888, 6098, 5855, 5474, 4871, 4003, 1494]
# printer = []
# for item in y:
#     printer.append(1-item/6000)
#     # y.append(item[0])
# plt.plot(x,printer)
# plt.ylabel('probablity')
# plt.xlabel('time')
# plt.title('The probability of measuring the 1 state versus the time')

# # plt.show()
# name = "Magnetic_bias_2peaks_newwest.png"
# plt.savefig(name)
# print(f"a plot of your graph has been saved as {name}")

# qproc = QuantumProcessor(
#     "TestQPD", num_positions=4,
#     mem_noise_models=[DepolarNoiseModel(500)] * 4, phys_instructions=[
#         PhysicalInstruction(INSTR_INIT, duration=2.5, parallel=True),
#         PhysicalInstruction(INSTR_X, duration=1, topology=[1,2,3],
#                             quantum_noise_model=T1T2NoiseModel(T1=0.5)),
#         PhysicalInstruction(INSTR_CNOT, duration=2, topology=[(2, 1), (1, 3)]),
#         PhysicalInstruction(INSTR_MEASURE, duration=3, parallel=True)])

# # First program
# prog1 = QuantumProgram(num_qubits=2)
# q1, q2 = prog1.get_qubit_indices(2)
# prog1.apply(INSTR_INIT, q1)
# prog1.apply(INSTR_INIT, q2)
# prog1.apply(INSTR_X, q1)
# # prog1.apply(INSTR_CNOT, [q1, q2])
# # prog1.apply(INSTR_SIGNAL, physical=False)
# prog1.apply(INSTR_MEASURE, q1, output_key="m1")
# # prog1.apply(INSTR_MEASURE, q2, output_key="m2")

# # print(qproc.qmemory.peek(0)[0].qstate.qrepr)
# qproc.execute_program(prog1, qubit_mapping=[1, 3])
# ns.sim_run()
# print(prog1.output) 
# # print(q1.qstate.qrepr,q2.qstate.qrepr)
# with open('test_input.txt') as f:
#     lines = f.readlines()

import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.optimize import curve_fit
# reading = lines[0].split()
# reading_2 = lines[2].split()
# print(reading[1][1:])
# print(reading_2)
# print(reading_2[0],reading_2[1])
# tmodel = T1T2NoiseModel()
# print(dir(tmodel._random_dephasing_noise))
from numpy import exp, pi, sqrt
from lmfit import Model
with open('magnetic_bias_store_1peak_10000_real.json', 'r') as f:
    data = json.load(f)


import math

# print(data)
def objective(x, a, b, c, d, e, f):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

# def gaussian(x, amp, cen, wid):
#     """1-d gaussian: gaussian(x, amp, cen, wid)"""
#     return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

def gaussian(x, amp, cen, wid):
    return amp * exp(-(x-cen)**2 / wid)
# gmodel = Model(gaussian)
def gaus(x,H,A,x0,sigma):
    return H+A*np.exp(-(x-x0)**2/(2*sigma**2))
# pars = gmodel.make_params(amp = 5, cen = 5, wid =1)
# init_vals = [1,0,1]
y = data["photon count"]
x = data["frequency"]
x = np.array(x)
n = len(x)
mean = sum(x*y)/sum(y)
sigma = sqrt(sum(y*(x-mean)**2)/sum(y))
popt,covar = curve_fit(gaus, x,y, p0 = [min(y),max(y),mean,sigma])
# result = gmodel.fit(y,pars,x=x)
# print(result.fit_report())
yerr = []
for i in range(len(y)):
    yerr.append(math.erfc(x[i]))
# popt, _ = curve_fit(objective,x,y)
# a,b,c,d,e,f = popt
# x_line = np.arange(min(x),max(x),1)
# y_line = gaussian(x_line, best_vals)
# print(x,y)
# theta = np.polyfit(x,y,2)
# y_line = theta[2] + theta[1] * pow(x,1)+theta[0]*pow(x,2)
plt.plot(x,y,'.', label = "original_data")
plt.ylabel("Photon count")
plt.xlabel("Frequency")
plt.title("The photon count versus the frequency")
# plt.errorbar(x,y,yerr = yerr, label = 'error bars')
plt.plot(x,gaus(x,*popt),'--', color = "red", label = "fitted data")
plt.legend()
plt.savefig("first_test")
# optimizedParameters,pcov = opt.curve_fit()

