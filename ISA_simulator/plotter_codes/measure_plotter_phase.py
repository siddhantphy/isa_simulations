from cProfile import label
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy
from scipy.optimize import curve_fit

from math import sqrt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm
from scipy.stats import norm
import seaborn as sns


with open('XL_measurement_values_phase_nocrosstalk_real_old.json', 'r') as f:
    data = json.load(f)

with open('YL_measurement_values_phase_nocrosstalk_real_old.json', 'r') as f:
    data_2 = json.load(f)

with open('ZL_measurement_values_phase_nocrosstalk_real_old.json', 'r') as f:
    data_3 = json.load(f)

with open('YL_measurement_values_phase_nocrosstalk_real_old.json', 'r') as f:
    data_4 = json.load(f)


def gaus(x,H,A,x0,sigma):
    return H+A*np.exp(-(x-x0)**2/(2*sigma**2))

def sigmoid(x, L ,x0, k, b):
    return -L / (1 + np.exp(-k*(x-x0)))+b
    


def test_func(x, a, b,p,c):
    return a * np.sin(b * x+p)+c


# data = data_2+data_2
def succes_prob_func(x,alpha,h0):
    return_data = h0+alpha*((np.cos(np.array(x)/2)**4)+(np.sqrt(1-(np.cos(np.array(x)/2))**2))**4)
    # return_data = 0.5*((np.cos(np.array(x)/2)**4)+(np.sqrt(1-(np.cos(np.array(x)/2)**2))))

    return return_data

# def func(x):
time = data["angle"]
rotation_angle = time
x_sin = rotation_angle
expected_value = (np.sin(x_sin))
counter = 0
error_bar_store = []
for k in range(len(data["total_measurment_amount_per_sweep"])):
    # average_measure = data["total_measurment_amount_per_sweep"][k]/data["total_memory_count"][k]
    average_measure = expected_value[k]
    sum_value = 0
    for i in range(int(data["total_memory_count"][k])):
        difference_squared = (data["measure_values_per_measure"][counter]-average_measure)**2
        counter +=1

        sum_value += difference_squared
    sigma = np.sqrt(sum_value/int(data["total_memory_count"][k]))
    z_95 = 1.959
    error_bar = sigma/np.sqrt(int(data["total_memory_count"][k]))*z_95
    error_bar_store.append(error_bar)
expected_value = (np.sin(np.array(x_sin)-np.pi/2))
counter = 0
error_bar_store_2 = []
for k in range(len(data_2["total_measurment_amount_per_sweep"])):
    # average_measure = data_2["total_measurment_amount_per_sweep"][k]/data_2["total_memory_count"][k]
    average_measure = expected_value[k]
    sum_value = 0
    for i in range(int(data_2["total_memory_count"][k])):
        difference_squared = (data_2["measure_values_per_measure"][counter]-average_measure)**2
        counter +=1
        
        sum_value += difference_squared
    sigma = np.sqrt(sum_value/int(data["total_memory_count"][k]))
    z_95 = 1.959
    error_bar = sigma/np.sqrt(int(data["total_memory_count"][k]))*z_95
    error_bar_store_2.append(error_bar)

counter = 0
error_bar_store_3 = []
for k in range(len(data_3["total_measurment_amount_per_sweep"])):
    # average_measure = data_3["total_measurment_amount_per_sweep"][k]/data_3["total_memory_count"][k]
    sum_value = 0
    for i in range(int(data_3["total_memory_count"][k])):
        difference_squared = (data_3["measure_values_per_measure"][counter]-0)**2
        counter +=1
        
        sum_value += difference_squared
    sigma = np.sqrt(sum_value/int(data["total_memory_count"][k]))
    z_95 = 1.959
    error_bar = sigma/np.sqrt(int(data["total_memory_count"][k]))*z_95
    error_bar_store_3.append(error_bar)

counter = 0
error_bar_store_4 = []

for k in range(len(data_4["total_measurment_amount_per_sweep"])):
    average_measure = data_4["total_memory_count"][k]/data_4["measure_amount_total"][0]
    sum_value = 0
    for i in range(int(data_4["measure_amount_total"][0])):
        difference_squared = (data_4["counter_values_per_measure"][counter]-average_measure)**2
        counter +=1
        
        sum_value += difference_squared
    sigma = np.sqrt(sum_value/1000)
    z_95 = 1.959
    error_bar = sigma/np.sqrt(1000)*z_95
    error_bar_store_4.append(error_bar)
print("check")
print(data_3["measure_amount_total"][0])
print(len(data_3["total_measurment_amount_per_sweep"]))
print(data_3["total_memory_count"])
print(data_3["measure_amount_total"][0])
print(sum(data_3["counter_values_per_measure"][0:int(data_3["total_memory_count"][k])]))
print(data_3["total_measurment_amount_per_sweep"][0])
Succescount = data["total_memory_count"]

Succescount_2 = data_2["total_memory_count"]

Succescount_3 = data_3["total_memory_count"]
Succescount_4 = data_4["total_memory_count"]

Z_V = data["total_measurment_amount_per_sweep"]

Z_V_2 = data_2["total_measurment_amount_per_sweep"]

Z_V_3 = data_3["total_measurment_amount_per_sweep"]
Z_V_4 = np.divide(Succescount_4,data_4["measure_amount_total"][0])

print(Succescount)
# Succescount = np.multiply(Succescount, 1/250)
Z_V = np.divide(Z_V,Succescount)

Z_V_2 = np.divide(Z_V_2,Succescount_2)

Z_V_3 = np.divide(Z_V_3,Succescount_3)
print(Succescount)
time = data["angle"]
time_2 = data_2["angle"]
time_3 = data_3["angle"]

time_4 = data_4["angle"]

rotation_angle = time
rotation_angle_2 = time_2

rotation_angle_3 =time_3
rotation_angle_4 = time_4

# C1 = rotation_angle/(np.pi)
# C2 = sqrt(1-C1^2)
# P = succes_prob_func(rotation_angle)
# print(f"P next {P}")
# P = succes_prob_func(rotation_angle,1)klkl

# mu, std = norm.fit(Z_V)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 1000)
# p = norm.pdf(x, mu, std)
# mean, var  = scipy.stats.distributions.norm.fit(Z_V)
# fitted_data = scipy.stats.distributions.norm.pdf(np.multiply(rotation_angle_4,360/(2*np.pi)), mean, var)
# plt.plot(np.multiply(rotation_angle_4,360/(2*np.pi)),fitted_data,'r-')

x = np.multiply(rotation_angle,360/(2*np.pi))
y = Z_V
y_2 = Z_V_2
# y_3 = Z_V_3
# mean = sum(x*y)/sum(y)
# sigma = sqrt(sum(y*(x-mean)**2)/sum(y))
# popt,covar = curve_fit(gaus, x,y, p0 = [min(y),max(y),mean,sigma])

# p0 = [max(y_3), np.median(x),1,min(y_3)] # this is an mandatory initial guess
# popt_2, pcov = curve_fit(sigmoid, x, y_3,p0, method='dogbox')

# plt.plot(x, gaus(x,*popt), c='b', linestyle = 'dashed', alpha = 0.4)
plt.plot(x, [0]*len(x), c='g', linestyle = 'dashed', alpha = 0.4)
# plt.plot(x, sigmoid(x,*popt_2), c='g', linestyle = 'dashed', alpha = 0.4)
def test(x, a, b,p,c):
    return a * np.sin(b * x+p)+c
 
# curve_fit() function takes the test-function
# x-data and y-data as argument and returns
# the coefficients a and b in param and
# the estimated covariance of param in param_cov
param, param_cov = curve_fit(test, x, y, p0 = [1,1/(2.5*len(x)),0,0])

# print(x)
# print(Z_V)
# params, params_covariance = curve_fit(test_func, x, y)
# print(params)
# ans = (param[0]*(np.sin(param[1]*x+param[2]))+param[3])
ans = (np.sin(x_sin))

plt.plot(x, ans, color ='blue', linestyle = 'dashed', alpha = 0.4)


param, param_cov = curve_fit(test, x, y_2, p0 = [1,1/(2.5*len(x)),np.pi/2,0])

# print(x)
# print(Z_V)
# params, params_covariance = curve_fit(test_func, x, y)
# print(params)
# ans = (param[0]*(np.sin(param[1]*x+param[2]))+param[3])
ans = (np.sin(np.array(x_sin)-np.pi/2))

plt.plot(x, ans, color ='orange', linestyle = 'dashed', alpha = 0.4)

# plt.plot(x,test_func(x, params[0],params[1],params[2],params[3]),c = 'b', linestyle = 'dashed', alpha = 0.4)

# plt.plot(x,test_func(x, params[0],params[1],params[2],params[3]),c = 'b', linestyle = 'dashed', alpha = 0.4)
# plt.plot(x,test_func(x, 1,1/(2.5*len(x)),0,0),c = 'b', linestyle = 'dashed', alpha = 0.4)

# plt.plot(x, p, 'k', linewidth=2)
# params, params_covariance = curve_fit(succes_prob_func, rotation_angle, Succescount)
# print(f"printing succescount and parameter value {Succescount, params}")
# plt.plot(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V,'.', label = "<XL>")
# plt.plot(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_2,'.', label = "<YL>")
# plt.plot(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_3,'.', label = "<ZL>")
plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V,yerr =  error_bar_store,fmt = '.', label = "<XL>")
plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_2,yerr =  error_bar_store_2,fmt = '.', label = "<YL>")
plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_3,yerr =  error_bar_store_3,fmt = '.', label = "<ZL>")
# plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_4,yerr = error_bar_store_4,fmt = '.')
# sns.displot(Z_V)
plt.legend()
# plt.ylim([0,1])

plt.ylabel("<O>")
plt.xlabel(r"Phase Angle, $\phi$")


# plt.title("The average logical measurement as a function off the rotation angle")
# plt.plot(rotation_angle/(2*np.pi)*360,P, label = "perfect_data")
# plt.plot(rotation_angle/(2*np.pi)*360,succes_prob_func(rotation_angle,params[0]), label = "after fitting")
plt.savefig("Measurement_phase_correct_old.pdf")
