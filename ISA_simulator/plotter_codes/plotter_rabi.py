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
# from lmfit import Model

import os
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'json_data_storage')
# save_direct = d.chdir("..")'
filename = 'test_rabi_error_lower_res.json'
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
for i in range(len(data["result"])-2):
    
    result_sum = sum(data["result"][i])/1000
    sum_value = 0
    for k in range(len(data["result"][i])):
        difference_squared_modicum = (data["result"][i][k]-result_sum)**2
        sum_value += difference_squared_modicum
    sigma_modicum = np.sqrt(sum_value/1000)
    z_95 = 1.959
    error_bar = sigma_modicum/np.sqrt(1000)*z_95
    error_bars.append(error_bar)
    y_new.append(result_sum)
params, params_covariance = scipy.optimize.curve_fit(test_func, x, y_new)
print(y_new)
print(x)
# mu, std = norm.fit([x,y])
# xmin,xmax = plt.xlim()
# p = norm.pdf(x,mu,std)
# y_line = theta[2] + theta[1] * pow(x,1)+theta[0]*pow(x,2)
# y = scipy.stats.distributions.norm.pdf(x,mean,sigma**2)
# plt.plot(x,y_new,'.', label = "original_data")
plt.errorbar(x,y_new, yerr = error_bars,fmt = '.',label = 'original data')
# plt.rcParams.update({'font.size': 18})
plt.ylabel("Measurement result", fontsize = 18)
plt.xlabel("Duration [s]", fontsize = 18)
# plt.title("The measurement result versus the microwave duration")
# plt.errorbar(x,y,yerr = yerr, label = 'error bars')
# plt.plot(x,gaus(x,*popt),'--', color = "red", label = "fitted data")
# plt.plot(x,p,'--', color = "red", label = "fitted data")
# sns.distplot([x,y])
plt.plot(x,test_func(x, params[0],params[1],params[2],params[3]), label = "fitted data")
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
plt.savefig(dir+"/rabi_check_paper.pdf")
# optimizedParameters,pcov = opt.curve_fit()

