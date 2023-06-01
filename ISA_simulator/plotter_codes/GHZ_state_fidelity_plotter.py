import matplotlib.pyplot as plt
import json
import numpy as np
from netsquid.qubits.qubitapi import *
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm
# with open('-_initialisation_surface7_storage.json', 'r') as f:
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

set_qstate_formalism(QFormalism.DM)
fidelity_modicum = []
fidelity_modicum_new_program = []
fidelity_plain = []
fidelity_plain_new_program = []
frequencies = []
for i in range(25):
    frequency = i*250

    with open('test_GHZ_modicum_'+str(frequency)+'hz_withFidelity_test.json', 'r') as f:
        data_modicum = json.load(f)
    with open('test_GHZ_modicum_'+str(frequency)+'hz_withFidelity_new_program_fixed.json', 'r') as f:
        data_modicum_new_program = json.load(f)
    with open('test_GHZ_plain_'+str(frequency)+'hz_withFidelity_fixed.json', 'r') as f:
        data_plain = json.load(f)
    with open('test_GHZ_plain_'+str(frequency)+'hz_withFidelity_new_program_fixed.json', 'r') as f:
        data_plain_new_program = json.load(f)
    
    z_size_modicum = np.array(data_modicum["state"]).astype(np.float64)
    z_size_modicum = np.multiply(z_size_modicum,1/1000)
    z_size_modicum_new_program = np.array(data_modicum_new_program["state"]).astype(np.float64)
    z_size_modicum_new_program = np.multiply(z_size_modicum_new_program,1/1000)

    z_size_plain = np.array(data_plain["state"]).astype(np.float64)
    z_size_plain = np.multiply(z_size_plain,1/1000)
    z_size_plain_new_program = np.array(data_plain_new_program["state"]).astype(np.float64)
    z_size_plain_new_program = np.multiply(z_size_plain_new_program,1/1000)

    upper_line = [0.5]+[0]*14+[0.5]
    middle_line = [0]*16
    upper_line_1 = [0]*5+[0.5]+[0]*4+[0.5]+[0]*5

    dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
   
    signs_modicum = np.array(data_modicum["signs"]).astype(np.float64)
    signs_modicum_new_program = np.array(data_modicum_new_program["signs"]).astype(np.float64)
    state_modicum = np.multiply(z_size_modicum,signs_modicum)
    state_modicum_new_program = np.multiply(z_size_modicum_new_program,signs_modicum_new_program)

    signs_plain = np.array(data_plain["signs"]).astype(np.float64)
    signs_plain_new_program = np.array(data_plain_new_program["signs"]).astype(np.float64)
    state_plain = np.multiply(z_size_plain,signs_plain)
    state_plain_new_program = np.multiply(z_size_plain_new_program,signs_plain_new_program)

    q0,q1,q2,q3 = create_qubits(4)
    q_real1,q_real2,q_real3,q_real4 = create_qubits(4)
    q_real1_new_program,q_real2_new_program,q_real3_new_program,q_real4_new_program = create_qubits(4)
    q_plain,q_plain_2,q_plain_3,q_plain_4 = create_qubits(4)
    q_plain_new_program,q_plain_2_new_program,q_plain_3_new_program,q_plain_4_new_program = create_qubits(4)


    assign_qstate([q0,q1,q2,q3],dz_perfect_0)
    matrix_modicum = state_modicum / np.trace(state_modicum)
    matrix_modicum_new_program = state_modicum_new_program / np.trace(state_modicum_new_program)

    matrix_plain = state_plain / np.trace(state_plain)
    matrix_plain_new_program = state_plain_new_program / np.trace(state_plain_new_program)

    assign_qstate([q_real1,q_real2,q_real3,q_real4],matrix_modicum)
    assign_qstate([q_real1_new_program,q_real2_new_program,q_real3_new_program,q_real4_new_program],matrix_modicum_new_program)

    assign_qstate([q_plain,q_plain_2,q_plain_3,q_plain_4],matrix_plain)
    assign_qstate([q_plain_new_program,q_plain_2_new_program,q_plain_3_new_program,q_plain_4_new_program],matrix_plain_new_program)

    fidelity_value_modicum = fidelity([q_real1,q_real2,q_real3,q_real4],q0.qstate.qrepr)
    fidelity_value_modicum_new_program = fidelity([q_real1_new_program,q_real2_new_program,q_real3_new_program,q_real4_new_program],q0.qstate.qrepr)
    fidelity_value_plain = fidelity([q_plain,q_plain_2,q_plain_3,q_plain_4],q0.qstate.qrepr)
    fidelity_value_plain_new_program = fidelity([q_plain_new_program,q_plain_2_new_program,q_plain_3_new_program,q_plain_4_new_program],q0.qstate.qrepr)
    
    fidelity_modicum.append(fidelity_value_modicum)
    fidelity_modicum_new_program.append(fidelity_value_modicum_new_program)
    fidelity_plain.append(fidelity_value_plain)
    fidelity_plain_new_program.append(fidelity_value_plain_new_program)
    frequencies.append(frequency)
    print(frequency)

print('am i before the plots?')
print(fidelity_modicum)
print(fidelity_modicum_new_program)
print(fidelity_plain)
print(fidelity_plain_new_program)

plt.plot(frequencies,fidelity_modicum,'.', label = "modicum")
plt.plot(frequencies,fidelity_modicum_new_program,'.', label = "modicum_new_program")
plt.plot(frequencies,fidelity_plain,'.', label = "plain")
plt.plot(frequencies,fidelity_plain_new_program,'.', label = "plain_new_program")
print('am i after the plots?')
plt.ylabel("Fidelity")
plt.xlabel("Detuning (Hz)")
plt.title("The GHZ state fidelity vs the detuning frequency")

plt.legend()
plt.savefig("GHZ_fidelity_new_testing")
