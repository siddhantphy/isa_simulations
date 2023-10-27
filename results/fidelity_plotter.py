import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from logical_gate_fidelity import fidelity_calculator_function

carbon_values = [5000.0,50000.0,500000.0,5000000.0]
electron_values = [0,500.0,5000.0]
carbon_values = np.arange(1e3,1e6,1e5)
electron_values = np.arange(1e2,5e3,10e2)


operation_space = ["physical",""]
fidelity_values = []
for electron_measurements in electron_values:
    fidelity_values_per_electron = []
    for items in carbon_values:
        fidelity_values_per_electron.append(fidelity_calculator_function(items,electron_measurements))
    fidelity_values.append(fidelity_values_per_electron)

fidelity_values_second = []

for electron_measurements in electron_values:
    fidelity_values_per_electron = []
    for items in carbon_values:
        fidelity_values_per_electron.append(fidelity_calculator_function(items,electron_measurements,operation_space=operation_space[0]))
    fidelity_values_second.append(fidelity_values_per_electron)
    
# for i, fidelity_plotter_values in enumerate(fidelity_values):
#     plt.plot(carbon_values,fidelity_plotter_values,'--', label = f'T2e = {electron_values[i]} L')
# plt.plot(carbon_values,fidelity_values,'o')
for i, fidelity_plotter_values in enumerate(fidelity_values_second):
    plt.plot(carbon_values,fidelity_plotter_values,'--', label = f'T2e = {electron_values[i]} P')


plt.ylabel("Fidelity", fontsize = 18)
plt.xlabel("Decoherence carbon [ns]", fontsize = 18)
# plt.xticks(fontsize = 8)
plt.ylim([0,1.1]) 
plt.title("physical")
plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.15),
    ncol = 5,
    fontsize = 6,
    frameon=False,
)
sys.path.append(os.path.abspath('../Figures'))

# dir = dir.replace('json_data_storage', 'Figures')
plt.tight_layout()
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('results', 'Figures')
plt.savefig(dir+"/Fidelity_calculations_decoherence_physical.pdf")
# plt.title(["physical"])

# plt.savefig(dir+"/ramsey_theory_test_2.pdf")
