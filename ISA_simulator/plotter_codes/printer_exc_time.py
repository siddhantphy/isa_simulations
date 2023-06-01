
from cProfile import label
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy
from scipy.optimize import curve_fit

from math import sqrt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm
import os
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'server_json_data/json_data_storage')

def objective(x, a,b,c):
	return a*c**x+b
counter = 0
error_bar_store_4 = []
# data_photoncount = []
data_time_false = []
for k in range(11):
    p = k+2
    with open(dir+'/duration_time_surface-7_new'+str(p)+'single_inst_false.json', 'r') as f:
        data_4 = json.load(f)
    # average_measure = sum(data_4["time:"])/(100)
    # sum_value = 0
    # for i in range(len(data_4["time:"])):
    #     difference_squared = (data_4["time:"][i]-average_measure)**2
        
    #     sum_value += difference_squared
    # sigma = np.sqrt(sum_value/100)
    # z_95 = 1.959
    # error_bar = sigma/np.sqrt(100)*z_95
    # error_bar_store_4.append(error_bar)
    # data_photoncount.append(sum(data_4["photonCount"]))
    data_time_false.append(data_4["time:"][-1])

data_time_true = []
for k in range(11):
    p = k+2
    with open(dir+'/duration_time_surface-7_new'+str(p)+'single_inst_true.json', 'r') as f:
        data_4 = json.load(f)

    data_time_true.append(data_4["time:"][-1])

data_time_entangle = []
for k in range(11):
    p = k+2
    with open(dir+'/duration_time_surface-7_new'+str(p)+'single_inst_true_withentangle.json', 'r') as f:
        data_4 = json.load(f)

    data_time_entangle.append(data_4["time:"][-1])
    # if p > 15:
    #     data_time.append(data_4["time:"][0])
    # else:
        # data_time.append(sum(data_4["time:"])/100)
    # data_time.append(average_measure)

x = np.arange(start=2, stop=13, step=1)
# popt,covar = curve_fit(objective, x,data_time)

print(x)
# print(data_time)
print(len(x))
# print(len(data_time))
# print(error_bar_store_4)
# print(max(error_bar_store_4))
error_bar_store_4.reverse()
# plt.errorbar(x,np.log(np.flip(data_time)),yerr = error_bar_store_4,fmt = '.')
# plt.plot(x,np.log10(data_time),'.', label = "original_data")
plt.semilogy(x, data_time_false, '.' ,label = 'Crosstalk on')
plt.semilogy(x, data_time_true, '.',label = 'Crosstalk off')
plt.semilogy(x, data_time_entangle, '.', label = 'Crosstalk off with entangled instructions' )

# plt.plot(x,data_time,'.', label = "original_data")

# plt.plot(x,objective(x,*popt),'--', color = "red", label = "fitted data")

# plt.legend()

# plt.ylim([0,1])
plt.ylabel("Execution time [s]")

plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.15),
    ncol = 3,
    fontsize = 9,
    frameon=False,
)
plt.xlabel("Qubit number")
# plt.title("The photon count as a function of photon loss probability")
# plt.plot(rotation_angle/(2*np.pi)*360,P, label = "perfect_data")
# plt.plot(rotation_angle/(2*np.pi)*360,succes_prob_func(rotation_angle,params[0]), label = "after fitting")
plt.savefig("Time_VS_qubitnumbers_server.pdf")
