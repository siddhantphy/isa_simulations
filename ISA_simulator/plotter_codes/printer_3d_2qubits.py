import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm
import os
# with open('-_initialisation_surface7_storage.json', 'r') as f:
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'json_data_storage')
# save_direct = d.chdir("..")'
filename = 'Matti_test_small.json'
fileopener = dir+"/" +filename
with open(fileopener, 'r') as f:
    data = json.load(f)

upper_line = [0.5]+[0]*14+[0.5]
middle_lines = [0]*16
x_pos = []
y_pos = []
for i in range(2):
    x_pos.append(np.arange(0, 2, 1).astype(np.int64))
    y_pos.append(np.arange(0, 2, 1).astype(np.int64))
# x_pos = np.arange(0, 16, 1).astype(np.int64)
# y_pos = np.arange(0, 16, 1).astype(np.int64)

print(x_pos)
print(len(x_pos[0]))
# x_pos = [["0000"],["0001"],["0010"],["0011"],["0100"],["0101"],["0110"],["0111"],["1000"],["1001"],["1010"],["1011"],["1100"],["1101"],["1110"],["1111"]]
# y_pos = [["0000"],["0001"],["0010"],["0011"],["0100"],["0101"],["0110"],["0111"],["1000"],["1001"],["1010"],["1011"],["1100"],["1101"],["1110"],["1111"]]
z_pos = np.zeros((2,2))
x_size = np.ones((2,2))
y_size = np.ones((2,2))
print(z_pos)
print(x_size)
print(data["state"][0])
# z_size = random.random.sample(xrange(20), 16*16)
z_size = np.array(data["state"]).astype(np.float64)
print(len(z_size[0]))
fig = plt.figure()
ax = plt.axes(projection='3d')
print(type(z_size[0][0]))
print(type(x_size[0]))
print(type(y_size[0]))
print(type(x_pos[0]))
print(z_size)

ax.xaxis.set_major_formatter(StrMethodFormatter("{x:2b}"))
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:2b}"))
ax.xaxis.set_ticks([0,1])
ax.yaxis.set_ticks([0,1])
xpos = [range(z_size.shape[0])]
ypos = [range(z_size.shape[1])]
xpos, ypos = np.meshgrid(xpos, ypos)
print(xpos)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)
print(xpos)
dx = np.ones_like(zpos)
dy = dx.copy()
dz = z_size.flatten()
signs = np.array(data["signs"]).astype(np.float64).flatten()
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
print(xpos)
print(ypos)
print(zpos)
print(dx)
print(dy)
print(dz)
# ax.grid(linewidth = 2)

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, edgecolor = 'black', color=rgba, shade = True)
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz_perfect_1.flatten(), color=rgba, alpha = 0.5)

# ax.set_title("The state after initialisation")
# ax.set_xlabel("The states",fontsize = 18)
# ax.set_ylabel("The states", fontsize = 18)
ax.set_zlabel(r"|$\rho$|", fontsize = 18)
dir = dir.replace('json_data_storage', 'Figures')
plt.tight_layout()
plt.savefig(dir+"/Hadamard_3d_carbon.pdf")


print(z_size)
print(data["signs"])