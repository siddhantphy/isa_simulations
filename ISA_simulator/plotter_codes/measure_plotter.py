from cProfile import label
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy
from scipy.optimize import curve_fit
import os
from math import sqrt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.cm as cm
from scipy.stats import norm
# import seaborn as sns
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace('plotter_codes', 'json_data_storage')

with open(dir+'/XL_measurement_values_angle_new_macros_no_noise.json', 'r') as f:
    data = json.load(f)

with open(dir+'/XL_measurement_values_angle_new_macros_no_noise.json', 'r') as f:
    data_2 = json.load(f)

with open(dir+'/XL_measurement_values_angle_new_macros_no_noise.json', 'r') as f:
    data_3 = json.load(f)

with open(dir+'/XL_measurement_values_angle_new_macros_no_noise.json', 'r') as f:
    data_4 = json.load(f)

# # data_4["total_measurment_amount_per_sweep"] = data["total_measurment_amount_per_sweep"]+data_2["total_measurment_amount_per_sweep"]+data_3["total_measurment_amount_per_sweep"]
# # data_4["total_memory_count"] = data["total_memory_count"]+data_2["total_memory_count"]+data_3["total_memory_count"]
# # data_4["measure_values_per_measure"] = data["measure_values_per_measure"]+data_2["measure_values_per_measure"]+data_3["measure_values_per_measure"]
# # data_4["measure_amount_total"] = data["measure_amount_total"]+data_2["measure_amount_total"]+data_3["measure_amount_total"]
# data_4["counter_values_per_measure"] = data["counter_values_per_measure"]+data_2["counter_values_per_measure"]+data_3["counter_values_per_measure"]
# # lists_of_lists = [data["total_measurment_amount_per_sweep"],data_2["total_measurment_amount_per_sweep"],data_3["total_measurment_amount_per_sweep"]]
# # data_4["total_measurment_amount_per_sweep"] = [sum(x) for x in zip(*lists_of_lists)]
# lists_of_lists = [data["total_memory_count"],data_2["total_memory_count"],data_3["total_memory_count"]]
# data_4["total_memory_count"] = [sum(x) for x in zip(*lists_of_lists)]
# # lists_of_lists = [data["measure_values_per_measure"],data_2["measure_values_per_measure"],data_3["measure_values_per_measure"]]
# # data_4["measure_values_per_measure"] = [sum(x) for x in zip(*lists_of_lists)]
# lists_of_lists = [data["measure_amount_total"],data_2["measure_amount_total"],data_3["measure_amount_total"]]
# data_4["measure_amount_total"] = [sum(x) for x in zip(*lists_of_lists)]
# # lists_of_lists = [data["counter_values_per_measure"],data_2["counter_values_per_measure"],data_3["counter_values_per_measure"]]
# # data_4["counter_values_per_measure"] = [sum(x) for x in zip(*lists_of_lists)]
# # print(data["total_measurment_amount_per_sweep"])
# print(data_4["total_measurment_amount_per_sweep"])
def gaus(x,H,A,x0,sigma):
    return H+A*np.exp(-(x-x0)**2/(2*sigma**2))

def sigmoid(x, L ,x0, k, b):
    return -L / (1 + np.exp(-k*(x-x0)))+b
    



# data = data_2+data_2
def succes_prob_func(x,alpha,c):
    return_data = c+alpha*((np.cos(np.array(x)/2)**4)+(np.sin(np.array(x)/2))**4)
    # return_data = 0.5*((np.cos(np.array(x)/2)**4)+(np.sqrt(1-(np.cos(np.array(x)/2)**2))))

    return return_data

time = data["angle"]
rotation_angle = time
x_sin_old = rotation_angle
ones_list = [1]*len(x_sin_old)
x_sin = []
for number in x_sin_old:
    x_sin.append(number / 2)

numerator = (np.cos(x_sin)**2+np.sin(x_sin)**2)
denumernator = (np.sqrt(np.cos(x_sin)**4+np.sin(x_sin)**4))

expected_value_X = (np.divide(numerator,denumernator))**2-ones_list
# def func(x):
counter = 0
error_bar_store = []
print(len(data["measure_values_per_measure"]))
for k in range(len(data["total_measurment_amount_per_sweep"])):
    # average_measure = data["total_measurment_amount_per_sweep"][k]/data["total_memory_count"][k]
    average_measure = expected_value_X[k]
    sum_value = 0
    for i in range(int(int(data["total_memory_count"][k])/2-2)):
        print(len(data["measure_values_per_measure"]))
        print(len(data["total_measurment_amount_per_sweep"]))
        print(data["total_memory_count"][k])
        print(counter)
        print(len(data["total_memory_count"]))
        print(sum(data["total_memory_count"])/3)
        print(data["total_measurment_amount_per_sweep"])
        print(sum(data["total_measurment_amount_per_sweep"]))
        print(data["total_measurment_amount_per_sweep"])
        
        difference_squared = (data["measure_values_per_measure"][counter]-average_measure)**2
        counter +=1

        sum_value += difference_squared
    sigma = np.sqrt(sum_value/int(data["total_memory_count"][k]))
    z_95 = 1.959
    error_bar = sigma/np.sqrt(int(data["total_memory_count"][k]))*z_95
    error_bar_store.append(error_bar)

# expected_value_Y = 0
# counter = 0
# error_bar_store_2 = []
# for k in range(len(data_2["total_measurment_amount_per_sweep"])):
#     # average_measure = data_2["total_measurment_amount_per_sweep"][k]/data_2["total_memory_count"][k]
#     average_measure = 0
#     sum_value = 0
#     for i in range(int(data_2["total_memory_count"][k])):
#         difference_squared = (data_2["measure_values_per_measure"][counter]-average_measure)**2
#         counter +=1
        
#         sum_value += difference_squared
#     sigma = np.sqrt(sum_value/int(data["total_memory_count"][k]))
#     z_95 = 1.959
#     error_bar = sigma/np.sqrt(int(data["total_memory_count"][k]))*z_95
#     error_bar_store_2.append(error_bar)

# numerator = (np.cos(x_sin)**2)
# denumernator = (np.sqrt(np.cos(x_sin)**4+np.sin(x_sin)**4))

# expected_value_Z = 2*(np.divide(numerator,denumernator))**2-ones_list
# counter = 0
# error_bar_store_3 = []
# for k in range(len(data_3["total_measurment_amount_per_sweep"])):
#     # average_measure = data_3["total_measurment_amount_per_sweep"][k]/data_3["total_memory_count"][k]
#     average_measure = expected_value_Z[k]
#     sum_value = 0
#     for i in range(int(data_3["total_memory_count"][k])):
#         difference_squared = (data_3["measure_values_per_measure"][counter]-average_measure)**2
#         counter +=1
        
#         sum_value += difference_squared
#     sigma = np.sqrt(sum_value/int(data["total_memory_count"][k]))
#     z_95 = 1.959
#     error_bar = sigma/np.sqrt(int(data["total_memory_count"][k]))*z_95
#     error_bar_store_3.append(error_bar)

counter = 0
error_bar_store_4 = []

for k in range(len(data_4["total_measurment_amount_per_sweep"])):
    average_measure = data_4["total_memory_count"][k]/data_4["measure_amount_total"][0]
    sum_value = 0
    for i in range(int(data_4["measure_amount_total"][0])):
        difference_squared = (data_4["counter_values_per_measure"][counter]-average_measure)**2
        counter +=1
        
        sum_value += difference_squared
    sigma = np.sqrt(sum_value/data_4["measure_amount_total"][0])
    z_95 = 1.959
    error_bar = sigma/np.sqrt(data_4["measure_amount_total"][0])*z_95
    error_bar_store_4.append(error_bar)

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
print(sum(data["counter_values_per_measure"][0:999]))
print(data["measure_amount_total"][0])
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
x = np.multiply(rotation_angle_3,360/(2*np.pi))

print(len(rotation_angle_4))
y = Z_V
y_3 = Z_V_3
mean = sum(x*y)/sum(y)
sigma = sqrt(sum(y*(x-mean)**2)/sum(y))
popt,covar = curve_fit(gaus, x,y, p0 = [min(y),max(y),mean,sigma])

p0 = [max(y_3), np.median(x),1,min(y_3)] # this is an mandatory initial guess
popt_2, pcov = curve_fit(sigmoid, x, y_3,p0, method='dogbox')

# plt.plot(x, gaus(x,*popt), c='b', linestyle = 'dashed', alpha = 0.4)
# plt.plot(x, [0]*len(x), c='orange', linestyle = 'dashed', alpha = 0.4)
# # # plt.plot(x, sigmoid(x,*popt_2), c='g', linestyle = 'dashed', alpha = 0.4)
# plt.plot(x, expected_value_X, color ='b', linestyle = 'dashed', alpha = 0.4)
# plt.plot(x, expected_value_Z, color ='g', linestyle = 'dashed', alpha = 0.4)

print(Z_V)
# plt.plot(x, p, 'k', linewidth=2)
# params, params_covariance = curve_fit(succes_prob_func,rotation_angle_4, Z_V_4)
# print(f"printing succescount and parameter value {Succescount, params}")
# plt.plot(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V,'.', label = "<XL>")
# plt.plot(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_2,'.', label = "<YL>")
# plt.plot(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_3,'.', label = "<ZL>")
# plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V,yerr =  error_bar_store,fmt = '.', label = "<XL>")
# plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_2,yerr =  error_bar_store_2,fmt = '.', label = "<YL>")
# plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_3,yerr =  error_bar_store_3,fmt = '.', label = "<ZL>")
plt.errorbar(np.multiply(rotation_angle_4,360/(2*np.pi)),Z_V_4,yerr = error_bar_store_4,fmt = '.', label = "before fitting")
# sns.displot(Z_V)
# plt.legend()
# plt.ylim([0,1])

# plt.ylabel("<O>")
plt.ylabel("P")

plt.xlabel(r"Rotation Angle, $\theta$")
# plt.title("The post selection rate (P) as a function off the rotation angle")
# plt.plot(rotation_angle/(2*np.pi)*360,P, label = "perfect_data")
plt.plot(np.multiply(rotation_angle_3,1/(2*np.pi)*360),succes_prob_func(rotation_angle_3,1,0), linestyle = 'dashed', label = "expected value")
# plt.plot(np.multiply(rotation_angle_4,1/(2*np.pi)*360),succes_prob_func(rotation_angle_4,params[0], params[1]), label = "after fitting")
plt.legend(loc="lower right")
plt.ylim([0,1])
dir = dir.replace('json_data_storage', 'Figures')
plt.tight_layout()
plt.savefig(dir+"/Surface-7_postselect_nonoise.pdf")
