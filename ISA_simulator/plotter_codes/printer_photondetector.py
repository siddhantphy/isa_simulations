
from cProfile import label
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy
from scipy.optimize import curve_fit

from math import sqrt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm


counter = 0
error_bar_store_4 = []
data_photoncount = []
for k in range(51):
    p = k/50
    with open('photoncount_p'+str(p)+'.json', 'r') as f:
        data_4 = json.load(f)
    average_measure = sum(data_4["photonCount"])/(1000)
    sum_value = 0
    for i in range(len(data_4["photonCount"])):
        difference_squared = (data_4["photonCount"][i]-average_measure)**2
        counter +=1
        
        sum_value += difference_squared
    sigma = np.sqrt(sum_value/1000)
    z_95 = 1.959
    error_bar = sigma/np.sqrt(1000)*z_95
    error_bar_store_4.append(error_bar)
    data_photoncount.append(sum(data_4["photonCount"]))

x = np.arange(start=0, stop=1.02, step=0.02)
print(x)
print(data_photoncount)
print(len(x))
print(len(data_photoncount))
print(error_bar_store_4)
print(max(error_bar_store_4))
error_bar_store_4.reverse()
plt.errorbar(x,np.flip(data_photoncount),yerr = error_bar_store_4,fmt = '.')

# plt.legend()
# plt.ylim([0,1])
plt.ylabel("Photon count")
plt.xlabel("Photon loss probability")
# plt.title("The photon count as a function of photon loss probability")
# plt.plot(rotation_angle/(2*np.pi)*360,P, label = "perfect_data")
# plt.plot(rotation_angle/(2*np.pi)*360,succes_prob_func(rotation_angle,params[0]), label = "after fitting")
plt.savefig("Photon_count_vs_prob.pdf")
