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
error_bars_modicum = []
error_bars_modicum_new_program = []
error_bars_plain = []
error_bars_plain_new_program = []

for i in range(25):
    frequency = i*250

    with open("test_GHZ_modicum_"+str(frequency)+"hz_withFidelity_test.json", 'r') as f: #'test_GHZ_modicum_'+str(frequency)+'hz_withFidelity_fixed.json'
        data_modicum = json.load(f)
    with open('test_GHZ_modicum_'+str(frequency)+'hz_withFidelity_new_program_fixed.json', 'r') as f:
        data_modicum_new_program = json.load(f)
    with open('test_GHZ_plain_'+str(frequency)+'hz_withFidelity_fixed.json', 'r') as f:
        data_plain = json.load(f)
    with open('test_GHZ_plain_'+str(frequency)+'hz_withFidelity_new_program_fixed.json', 'r') as f:
        data_plain_new_program = json.load(f)
    
    
    fidelity_value_modicum = sum(data_modicum["fidelity"])/100
    fidelity_value_modicum_new_program = sum(data_modicum_new_program["fidelity"])/1000
    fidelity_value_plain = sum(data_plain["fidelity"])/1000
    fidelity_value_plain_new_program = sum(data_plain_new_program["fidelity"])/1000
    sum_value_modicum = 0
    sum_value_modicum_new_program = 0
    sum_value_plain = 0
    sum_value_plain_new_program = 0

    for k in range(len(data_modicum["fidelity"])):
        difference_squared_modicum = (data_modicum["fidelity"][k]-fidelity_value_modicum)**2
        sum_value_modicum += difference_squared_modicum
        difference_squared_modicum_new_program = (data_modicum_new_program["fidelity"][k]-fidelity_value_modicum_new_program)**2
        sum_value_modicum_new_program += difference_squared_modicum_new_program
        difference_squared_plain = (data_plain["fidelity"][k]-fidelity_value_plain)**2
        sum_value_plain += difference_squared_plain
        difference_squared_plain_new_program = (data_plain_new_program["fidelity"][k]-fidelity_value_plain_new_program)**2
        sum_value_plain_new_program += difference_squared_plain_new_program
    
    sigma_modicum = np.sqrt(sum_value_modicum/100)
    sigma_modicum_new_program = np.sqrt(sum_value_modicum_new_program/1000)
    sigma_plain = np.sqrt(sum_value_plain/1000)
    sigma_plain_new_program = np.sqrt(sum_value_modicum_new_program/1000)
    z_95 = 1.959
    error_bar_modicum = sigma_modicum/np.sqrt(100)*z_95
    error_bar_modicum_new_program = sigma_modicum_new_program/np.sqrt(1000)*z_95
    error_bar_plain = sigma_plain/np.sqrt(1000)*z_95
    error_bar_plain_new_program = sigma_plain_new_program/np.sqrt(1000)*z_95

    error_bars_modicum.append(error_bar_modicum)
    error_bars_modicum_new_program.append(error_bar_modicum_new_program)
    error_bars_plain.append(error_bar_plain)
    error_bars_plain_new_program.append(error_bar_plain_new_program)

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

# plt.plot(frequencies,fidelity_modicum,'.', label = "modicum")
# plt.plot(frequencies,fidelity_modicum_new_program,'.', label = "modicum_new_program")
# plt.plot(frequencies,fidelity_plain,'.', label = "plain")
# plt.plot(frequencies,fidelity_plain_new_program,'.', label = "plain_new_program")
plt.errorbar(frequencies,fidelity_modicum, yerr = error_bars_modicum,fmt = '.',label = 'modicum')
plt.errorbar(frequencies,fidelity_modicum_new_program, yerr = error_bars_modicum_new_program,fmt = '.',label = 'modicum new program')
plt.errorbar(frequencies,fidelity_plain, yerr = error_bars_plain,fmt = '.',label = 'plain')
plt.errorbar(frequencies,fidelity_plain_new_program, yerr = error_bars_plain_new_program,fmt = '.',label = 'plain new program')

print('am i after the plots?')
plt.ylabel("Fidelity")
plt.xlabel("Detuning (Hz)")
plt.title("The GHZ state fidelity vs the detuning frequency")

plt.legend()
plt.savefig("GHZ_fidelity_fixed_by_fidelity_test")
