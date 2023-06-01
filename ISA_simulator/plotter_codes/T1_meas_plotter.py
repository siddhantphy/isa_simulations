import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.optimize import curve_fit
from numpy import exp, pi, sqrt

import os
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'json_data_storage')
# save_direct = d.chdir("..")'
# filename = 'test_rabi_error_lower_res.json'
dec = '900'
# filename = 'T1_meas_carb2_68e6.json'
filename = 'T1_meas_elec5e3.json'

# filename = 'rabi_decoherence_test_T2=1s_B=4.5mT.json'
fileopener = dir+"/" +filename
with open(fileopener, 'r') as f:
    data = json.load(f)

import scipy
import math
from scipy.stats import norm
from scipy.optimize import leastsq

y = data["result"]
x = data["time"]
x = np.array(x)
x = x[:-1]
# print(x,y)
t = x
def test_func(x, a, b,p,c):
    return a * np.sin(b * x+p)+c
t = x
# data_fit = hi*np.sin(w*t+p)+c
# print(result.fit_report())
yerr = []
error_bars = []
y_new = []
# for i in range(len(y)):
#     yerr.append(math.erfc(x[i]))
# popt, _ = curve_fit(objective,x,y)
# a,b,c,d,e,f = popt
# x_line = np.arange(min(x),max(x),1)
# y_line = gaussian(x_line, best_vals)
# print(x,y)
# import seaborn as sns
# theta = np.polyfit(x,y,2)
print(data["result"][0])
j=0
MeasureAmount = 200
print(len(data["result"])/MeasureAmount)
for i in range(int(len(data["result"])/MeasureAmount)):
    print(data['result'][i])
    print(j)
    result_sum = sum(data["result"][j:j+MeasureAmount])/MeasureAmount
    sum_value = 0
    for k in range(MeasureAmount):
        difference_squared_modicum = (data["result"][k+j]-result_sum)**2
        sum_value += difference_squared_modicum
    sigma_modicum = np.sqrt(sum_value/MeasureAmount)
    z_95 = 1.959
    error_bar = sigma_modicum/np.sqrt(MeasureAmount)*z_95
    error_bars.append(error_bar)
    y_new.append(result_sum)
    j = j+MeasureAmount
# params, params_covariance = scipy.optimize.curve_fit(test_func, x, y_new)
print(len(x))
print(len(data['result']))
print(len(data['result'])/len(x))
print(len(y_new))
# plt.errorbar(x,y_new, yerr = error_bars,fmt = '.',label = 'original data')
def T1_noise(x, T1):
    return 2*(1-exp(-x/T1))-1

plt.plot(x,y_new,'o', label = 'data')
T1 = 5e3
plt.plot(x, T1_noise(x,T1), '--', color = 'red', label = 'expected value')

# plt.rcParams.update({'font.size': 18})
plt.ylabel("Measurement result", fontsize = 18)
plt.xlabel("Duration [ns]", fontsize = 18)

# plt.plot(x,test_func(x, params[0],params[1],params[2],params[3]), label = "fitted data")
# plt.tick_params(axis="x", labelsize=8)
plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.15),
    ncol = 2,
    fontsize = 18,
    frameon=False,
)
# plt.legend(loc = 3,fontsize = 18)
dir = dir.replace('json_data_storage', 'Figures')
plt.tight_layout()
plt.savefig(dir+"/T1_meas_experiment_elec5e3_with_expected.pdf")
# optimizedParameters,pcov = opt.curve_fit()

