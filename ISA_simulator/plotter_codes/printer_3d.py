import matplotlib.pyplot as plt
import json
import numpy as np
from netsquid.qubits.qubitapi import *
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm
# with open('-_initialisation_surface7_storage.json', 'r') as f:
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism

set_qstate_formalism(QFormalism.DM)
import os
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'json_data_storage')
# save_direct = d.chdir("..")'
filename = 'Surface-7_first_state_Siddhant.json'
fileopener = dir+"/" +filename
with open(fileopener, 'r') as f:
    data = json.load(f)


upper_line = [0.5]+[0]*14+[0.5]
middle_lines = [0]*16
x_pos = []
y_pos = []
for i in range(16):
    x_pos.append(np.arange(0, 16, 1).astype(np.int64))
    y_pos.append(np.arange(0, 16, 1).astype(np.int64))
# x_pos = np.arange(0, 16, 1).astype(np.int64)
# y_pos = np.arange(0, 16, 1).astype(np.int64)

print(x_pos)
print(len(x_pos[0]))
# x_pos = [["0000"],["0001"],["0010"],["0011"],["0100"],["0101"],["0110"],["0111"],["1000"],["1001"],["1010"],["1011"],["1100"],["1101"],["1110"],["1111"]]
# y_pos = [["0000"],["0001"],["0010"],["0011"],["0100"],["0101"],["0110"],["0111"],["1000"],["1001"],["1010"],["1011"],["1100"],["1101"],["1110"],["1111"]]
z_pos = np.zeros((16,16))
x_size = np.ones((16,16))
y_size = np.ones((16,16))
# print(z_pos)
# print(x_size)
# print(data["state"][0])
# z_size = random.random.sample(xrange(20), 16*16)
z_size = np.array(data["state"]).astype(np.float64)
# max_value = max(z_size)
# z_size = np.multiply(z_size,1/1000)
# print(len(z_size[0]))
fig = plt.figure()
ax = plt.axes(projection='3d')
# print(type(z_size[0][0]))
# print(type(x_size[0]))
# print(type(y_size[0]))
# print(type(x_pos[0]))
# print(z_size)

ax.xaxis.set_major_formatter(StrMethodFormatter("|{x:04b}>"))
ax.yaxis.set_major_formatter(StrMethodFormatter("|{x:04b}>"))
ax.xaxis.set_ticks(np.arange(0, 16, 3))
ax.yaxis.set_ticks(np.arange(0, 16, 3))
xpos = [range(z_size.shape[0])]
ypos = [range(z_size.shape[1])]
xpos, ypos = np.meshgrid(xpos, ypos)
# print(xpos)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
# ax.tick_params(axis='x', labelsize=19)
# ax.tick_params(axis='y', labelsize=24)
zpos = np.zeros_like(xpos)
# print(xpos)
dx = np.ones_like(zpos)
dy = dx.copy()
dz = z_size.flatten()
signs = np.array(data["signs"]).astype(np.float64).flatten()
# print(signs)
# cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
cmap = cm.get_cmap('rainbow')
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
# rgba = [cmap((k-min_height)/max_height) for k in dz] 
rgba = [cmap(k) for k in signs]
upper_line = [0.5]+[0]*14+[0.5]
middle_line = [0]*16
upper_line_1 = [0]*5+[0.5]+[0]*4+[0.5]+[0]*5
# print(dz)
dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
dz_perfect_1 = np.array([middle_line,middle_line,middle_line,middle_line,middle_line,upper_line_1,middle_line,middle_line,middle_line,middle_line,upper_line_1,middle_line,middle_line,middle_line,middle_line,middle_line]).astype(np.float64)
# print(dz_perfect)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz,edgecolor = 'black', color=rgba)
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz_perfect_1.flatten(), color=rgba, alpha = 0.5)

# q0,q1,q2,q3 = create_qubits(4)
# q_real1,q_real2,q_real3,q_real4 = create_qubits(4)
# assign_qstate([q0,q1,q2,q3],dz_perfect_0)
# signs = np.array(data["signs"]).astype(np.float64)

# matrix = np.multiply(z_size / np.trace(z_size),signs)
# print(z_size)
# print(np.trace(z_size))
# print(np.trace(matrix))
# print(len(matrix))
# assign_qstate([q_real1,q_real2,q_real3,q_real4],matrix)
# print(q0.qstate.qrepr)
# print(q_real1.qstate.qrepr)
# fidelity_value = fidelity([q_real1,q_real2,q_real3,q_real4],q0.qstate.qrepr)
# print(f"the fidelity value is {fidelity_value}")
# ax.set_title("The logical initialized state")
# ax.set_xlabel("The states")
# ax.set_ylabel("The states")
ax.set_zlabel(r"|$\rho$|",fontsize = 18)
# plt.legend(
#     loc='upper center', 
#     bbox_to_anchor=(0.5, 1.15),
#     fontsize = 18,
#     frameon=False,
# )
# plt.legend(loc = 3,fontsize = 18)
dir = dir.replace('json_data_storage', 'Figures')
plt.tight_layout()
plt.savefig(dir+"/Entangled_state_Surface-7_+     .pdf")


# print(z_size)
# print(data["signs"])