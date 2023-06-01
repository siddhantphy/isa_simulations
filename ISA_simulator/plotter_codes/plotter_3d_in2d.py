import matplotlib.pyplot as plt
import json
import numpy as np
from netsquid.qubits.qubitapi import *
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm
# with open('-_initialisation_surface7_storage.json', 'r') as f:
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
clist = [(0.1, 0.6, 1.0), (0.05, 0.05, 0.05), (0.8, 0.5, 0.1)]
bublor_cmap = LinearSegmentedColormap.from_list("bublor", clist)
set_qstate_formalism(QFormalism.DM)
import os
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'json_data_storage')
filename = 'test_GHZ_modicum_0hz_withFidelity.json'
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

z_pos = np.zeros((16,16))
x_size = np.ones((16,16))
y_size = np.ones((16,16))

z_size = np.array(data["state"]).astype(np.float64)
print(z_size)
z_size = np.multiply(z_size,1/1000)
fig = plt.figure()
ax = plt.axes()


dz = z_size.flatten()
signs = np.array(data["signs"]).astype(np.float64).flatten()
ax.xaxis.set_major_formatter(StrMethodFormatter("|{x:04b}>"))
ax.yaxis.set_major_formatter(StrMethodFormatter("|{x:04b}>"))
x_axis_values = np.arange(0,16,5)
y_axis_values = np.arange(0,16,5)
print(x_axis_values)
ax.xaxis.set_ticks(np.arange(0, 16, 5))
ax.yaxis.set_ticks(np.arange(0, 16, 5))

ax.tick_params(axis='x', labelsize=19)
ax.tick_params(axis='y', labelsize=24)
cmap = cm.get_cmap('rainbow')

im = ax.imshow(z_size, cmap=cmap)
cbar = fig.colorbar(mappable = im, cmap=cmap)
cbar.ax.tick_params(labelsize=19)
cbar.ax.set_ylim(0,0.5)
cbar.set_label(r"|$\rho$|",size = 19)


fig.tight_layout()
ax.tick_params(axis='y', labelsize=19)


# y  = np.arange(0, 16, 3)
# y = y[::-1]
# ax.yaxis.set_ticks(y)

# print(y)


rgba = [cmap(k) for k in signs]
# fig, axes = plt.figure()
# fm.FontProperties.set_size(fig, size = 200)

# ax.set_zlabel(r"|$\rho$|",fontsize = 18)
# ax.set_xlabel("Potential qubit states",fontsize = 24)
# ax.set_ylabel("Potential qubit states",fontsize = 24)
#  ax.set_zlabel(r"|$\rho$|",fontsize = 18)


dir = dir.replace('json_data_storage', 'Figures')
fig.gca().invert_yaxis()


fig.savefig(dir+"/3dplot_modicum_init_test_2_paper.pdf")

