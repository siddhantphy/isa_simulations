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
from lmfit import Model
import os
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'json_data_storage')
# save_direct = d.chdir("..")'
filename = 'test_magnetic_bias_10ns_50000.json'
fileopener = dir+"/" +filename
with open(fileopener, 'r') as f:
    data = json.load(f)

import scipy
import math
from scipy.stats import norm

# print(data)
def objective(x, a, b, c, d, e, f):
	return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

# def gaussian(x, amp, cen, wid):
#     """1-d gaussian: gaussian(x, amp, cen, wid)"""
#     return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

def gaussian(x, amp, cen, wid):
    return amp * exp(-(x-cen)**2 / wid)
# gmodel = Model(gaussian)
def gaus(x,H,A,x0,sigma):
    return H+A*np.exp(-(x-x0)**2/(2*sigma**2))

def gauss3(x,H1,A1,x1,sigma1,H2,A2,x2,sigma2,H3,A3,x3,sigma3):
    return H1+A1*np.exp(-(x-x1)**2/(2*sigma1**2))+H2+A2*np.exp(-(x-x2)**2/(2*sigma2**2))+H3+A3*np.exp(-(x-x3)**2/(2*sigma3**2))
# pars = gmodel.make_params(amp = 5, cen = 5, wid =1)
# init_vals = [1,0,1]
# y = data["photon count"]
y = data["photon"]

x = data["frequency"]
x = np.array(x)
n = len(x)
mean = sum(x*y)/sum(y)
sigma = sqrt(sum(y*(x-mean)**2)/sum(y))
popt,covar = curve_fit(gaus, x,y, p0 = [min(y),max(y),mean,sigma])
# result = gmodel.fit(y,pars,x=x)
# print(result.fit_report())
yerr = []

for i in range(len(y)):
    yerr.append(math.erfc(x[i]))
# popt, _ = curve_fit(objective,x,y)
# a,b,c,d,e,f = popt
# x_line = np.arange(min(x),max(x),1)
# y_line = gaussian(x_line, best_vals)
# print(x,y)
import seaborn as sns
# theta = np.polyfit(x,y,2)
mu, std = norm.fit([x,y])
xmin,xmax = plt.xlim()
p = norm.pdf(x,mu,std)
# y_line = theta[2] + theta[1] * pow(x,1)+theta[0]*pow(x,2)
# y = scipy.stats.distributions.norm.pdf(x,mean,sigma**2)
plt.plot(x,y,'.', label = "original_data")
plt.ylabel("Photon count",fontsize=18)
plt.xlabel("Frequency [Hz]",fontsize=18)
# plt.title("The photon count versus the frequency")
# plt.errorbar(x,y,yerr = yerr, label = 'error bars')
plt.plot(x,gaus(x,*popt),'--', color = "red", label = "fitted data")
# plt.plot(x,p,'--', color = "red", label = "fitted data")
# sns.distplot([x,y])
# plt.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
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
plt.savefig(dir+"/magnetic_bias_paper.pdf")
# optimizedParameters,pcov = opt.curve_fit()

