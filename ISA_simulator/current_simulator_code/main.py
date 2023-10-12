from cmath import inf
import netsquid as ns
from NVNetwork_setup import network_setup
from netsquid.qubits.qubitapi import *
import numpy as np
from NVprotocols import *
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
import json
import os
import time
import itertools
import sys # in order to take command line inputs
from multiprocessing import Pool

set_qstate_formalism(QFormalism.DM)

def main(savename = None, filename = None, node_number = 4,qubit_number = 2, photon_detection_prob = 1, printstates = False, storedata = None,node_distance = 4e-3, photo_distance = 2e-3, noiseless = False, detuning = 0, electron_T2 = 0, electron_T1 = 0, carbon_T1 = 0, carbon_T2 = 0, single_instruction = True, B_osc = 400e-6,no_z_precission = 1, frame = "rotating", wait_detuning = 0, clk = 0, clk_local = 0,B_z = 40e-3,rotation_with_pi = 0,Fault_tolerance_check = False):
	start_time = time.perf_counter()

	print(f"the number of nodes is {node_number} with qubits {qubit_number}")
	print(f"savename is {savename}")
	# Set the filename, which text file needs to be read, this text file should contain QISA specific instructions
	if savename != None:
		storedata = savename
	if filename == None:
		filename = 'Surface-7-logical_X_posselect.txt'
	else:
		filename = filename + '.txt'
	# The text file with proposed filename is opened and the information is stored in parameter 'lines'
	path = os.path.realpath(__file__)
	dir = os.path.dirname(path)
	dir = dir.replace('current_simulator_code', 'Parameter_files')
	if noiseless == True:
		fileopener = dir+"/parameters_noiseless.txt"
	else:
		fileopener = dir+"/parameters_noisy.txt"
	parameter_dict = {}
	with open(fileopener) as f:
		parameters = f.readlines()
	# print(f"the file content is {parameters}")
	# print(f"the filename is {filename}")
	for item in parameters:
		parameter_name = item.split()[0]
		parameter_value = item.split()[1]
		parameter_dict[parameter_name] = parameter_value
	print(parameter_dict)
	# fileopener = dir +"../QISA_schemes/"+filename
	dir = dir.replace('Parameter_files', 'QISA_schemes')
	# save_direct = d.chdir("..")'
	fileopener = dir+"/" +filename
	with open(fileopener) as f:
		lines = f.readlines()
	# Line reader is a empty list, to which all the instructions can be added
	line_reader = []
	if filename == "server_test.txt":
		for i in range(qubit_number-2):
			lines.insert(2*i+2,'Initialize q0')
			lines.insert(2*i+3,'SwapEC q0 ' +str(i+1) )
			lines.insert(2*i+7+2*(i+1),'QgateCC q0 '+str(i+1)+' 0 3.14')
	elif filename == "server_test_2.txt":
		for i in range(qubit_number-2):
			lines.insert(2*i+2,'Initialize q0')
			lines.insert(2*i+3,'SwapEC q0 ' +str(i+1) )
			lines.insert(2*i+4+2*(i+1),'QgateCC q0 '+str(i+1)+' 0 3.14')
		print(lines)
	# Loop through every instruction
	for i in range(len(lines)):
		# Check if the instruction is 'empty' (thus only containing \n)
		if lines[i] != '\n':
			# Check if the instruction starts with a #, this is interpeted as a comment, thus not taken into account as an instruction
			if lines[i][0] != "#":
				# Append every instruction which is not a comment or an empty instruction to the list of instructions which need to be interperted by the global controller.
				line_reader.append(lines[i])
	#read the parameters from the parameter list
	node_number = int(parameter_dict["NodeNumber"])
	qubit_number = int(parameter_dict["QubitNumber"])
	photon_detection_prob = float(parameter_dict["PhotonDetectionProbability"])
	printstates = bool(parameter_dict["printstates"])
	node_distance = float(parameter_dict["NodeDistance"])
	photo_distance = float(parameter_dict["PhotonDistance"])
	detuning = float(parameter_dict["Detuning"])
	electron_T2 = float(parameter_dict["ElectronDecoherence"])
	electron_T1 = float(parameter_dict["ElectronRelaxation"])
	carbon_T2 = float(parameter_dict["CarbonDecoherence"])
	carbon_T1 = float(parameter_dict["CarbonRelaxation"])
	single_instruction = bool(parameter_dict["Crosstalk"])
	no_z_precission = int(parameter_dict["NoZPrecession"])
	frame = str(parameter_dict["frame"])
	wait_detuning = float(parameter_dict["WaitDetuning"])
	clk = float(parameter_dict["clk"])
	clk_local = float(parameter_dict["clkLocal"])
	B_z = float(parameter_dict["StaticField"])
	rotation_with_pi = int(parameter_dict["RotationWithPi"])

	# Setup the network by calling the network setup function
	network = network_setup(node_number = node_number, photon_detection_probability = photon_detection_prob,qubit_number = qubit_number, noiseless = noiseless, node_distance = node_distance, photo_distance = photo_distance, detuning = detuning, electron_T2 = electron_T2, electron_T1 = electron_T1, carbon_T1 = carbon_T1, carbon_T2=carbon_T2, single_instruction=single_instruction, B_osc = B_osc, no_z_precission=no_z_precission, frame = frame, wait_detuning=wait_detuning, clk_local = clk_local)
	network.noiseless = noiseless
	network.noise_parameters = {}
	network.noise_parameters["T2_carbon"] = carbon_T2
	
	# Start the global controller functionality and give the global controller its instructions
	if Fault_tolerance_check == True:
		prog_copy = line_reader
		for i in range(len(line_reader)):
			if line_reader[i][0] in ['BR','LDi','MUL','ADDi','ADD','ST','LABEL','printer','print']:
				continue
			for j in range(qubit_number-2):
				for item in ['x','y','z']:
					prog_copy.insert(i,'fault_inject q0 '+item+' '+str(j)) #add addition for multiple nv centers
					send_protocol = Global_cont_Protocol(network = network, input_prog = line_reader, clk = clk)
					send_protocol.start()
	else:
		send_protocol = Global_cont_Protocol(network = network, input_prog = line_reader, clk = clk)
		send_protocol.start()
	
	#start running 
	print(ns.sim_run()) 
	
	# If you want to print the electron and carbon states, printstates can be set to True
	if printstates == True:
		printer(network)

	
# Save the data by use of the following function
def data_storer(data,store_name):
	path = os.path.realpath(__file__)
	dir = os.path.dirname(path)
	dir = dir.replace('current_simulator_code', 'json_data_storage')
	filename = store_name
	fileopener = dir+"//" +filename
	with open(fileopener, 'w') as file_object:
		json.dump(data, file_object)

def printer(network):
	
	# Set the print precision of the density matrices, this is added for clarity.
	np.set_printoptions(precision=1, suppress = True)
    #gather state data from electron for node0 and node 1
	q_elec = network.get_node("nvnode0").qmemory.peek(0)[0]
	q_elec_2 = network.get_node("nvnode1").qmemory.peek(0)[0]
 
    #print data
	print("electron state 1,2 and paired")
	print(reduced_dm(q_elec))
	print(reduced_dm(q_elec_2))
	print(q_elec.qstate.qrepr)
 
    #gather state data from carbon for node 0
	carbon = network.get_node("nvnode0").qmemory.peek(1)[0]
	carbon_2 = network.get_node("nvnode1").qmemory.peek(1)[0]
	carbon_3 = network.get_node("nvnode0").qmemory.peek(2)[0]
	carbon_4 = network.get_node("nvnode1").qmemory.peek(2)[0]
    #print combined state
	print(reduced_dm([carbon,carbon_2,carbon_3,carbon_4]))




if __name__ == "__main__":
	arguments = []
	duration_list = []
	start_time = time.perf_counter()
	if len(sys.argv) == 1:
		main(node_number = 2,qubit_number=3, printstates=True, detuning = 1e3, electron_T2= 5e3, carbon_T2 = 5e6 ,single_instruction = False, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 10e6)
	elif sys.argv[1] == "noiseless":
		main(node_number = 2,qubit_number=5, printstates=True, detuning = 0, electron_T2= 0, carbon_T2 = 0 ,single_instruction = True, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 0)

	elif sys.argv[1] == "Pool":
		with Pool() as p:
			arg = [(i,None,2,5,0,False,None,4e-3,2e-3,True,0,5e3,0,0,5e6) for i in range(5)]
			p.starmap(main,arg)
