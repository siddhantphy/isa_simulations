
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
dir = dir.replace('plotter_codes', 'json_data_storage')

def objective(x, a,b,c):
	return a*c**x+b
counter = 0
error_bar_store_4 = []
# data_photoncount = []
data_time = []
x = []
for k in range(1,100):
    p = k*1e6
    with open(dir+'/clk_'+str(p)+'_test.json', 'r') as f:
        data_4 = json.load(f)
    average_measure = sum(data_4["result"])/(1000)
    sum_value = 0
    for i in range(len(data_4["result"])):
        difference_squared = (data_4["result"][i]-average_measure)**2
        
        sum_value += difference_squared
    sigma = np.sqrt(sum_value/1000)
    z_95 = 1.959
    error_bar = sigma/np.sqrt(1000)*z_95
    error_bar_store_4.append(error_bar)
    # data_photoncount.append(sum(data_4["photonCount"]))
    data_time.append(sum(data_4["result"])/1000)
    x.append(p)
    # if p > 15:
    #     data_time.append(data_4["time:"][0])
    # else:
        # data_time.append(sum(data_4["time:"])/100)
    # data_time.append(average_measure)

# x = np.arange(start=2, stop=28, step=1)
# popt,covar = curve_fit(objective, x,data_time)
print(p)
# print(x)
# print(data_time)
# print(len(x))
# print(len(data_time))
# print(error_bar_store_4)
# print(max(error_bar_store_4))
# error_bar_store_4.reverse()
print(len(x))
print(x)
print(len(error_bar_store_4))
# plt.errorbar(x,error_bar_store_4,yerr = error_bar_store_4,fmt = '.', label = "before fitting")
plt.ylabel("average measurement value")

plt.xlabel("clk frequency")
# plt.title("The post selection rate (P) as a function off the rotation angle")
# plt.plot(rotation_angle/(2*np.pi)*360,P, label = "perfect_data")
plt.plot(x,data_time, linestyle = 'dashed', label = "expected value")
# plt.plot(np.multiply(rotation_angle_4,1/(2*np.pi)*360),succes_prob_func(rotation_angle_4,params[0], params[1]), label = "after fitting")
# plt.legend(loc="lower right")
dir = dir.replace('json_data_storage', 'Figures')
plt.tight_layout()
plt.savefig(dir+"/clk_vs_measurement.pdf")



plt.xlabel("Qubit number")
# plt.title("The photon count as a function of photon loss probability")
# plt.plot(rotation_angle/(2*np.pi)*360,P, label = "perfect_data")
# plt.plot(rotation_angle/(2*np.pi)*360,succes_prob_func(rotation_angle,params[0]), label = "after fitting")
plt.savefig("Time_vs_qubits_qgatee_new.pdf")
