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

def main(savename = None, filename = None, node_number = 4,qubit_number = 2, photon_detection_prob = 1, printstates = False, storedata = None,node_distance = 4e-3, photo_distance = 2e-3, noiseless = True, detuning = 0, electron_T2 = 0, electron_T1 = 0, carbon_T1 = 0, carbon_T2 = 0, single_instruction = True, B_osc = 400e-6,no_z_precission = 1, frame = "rotating", wait_detuning = 0, clk = 0, clk_local = 0):
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
	print(f"the filename is {filename}")
	path = os.path.realpath(__file__)
	dir = os.path.dirname(path)
	dir = dir.replace('current_simulator_code', 'QISA_schemes')
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
	
	# Setup the network by calling the network setup function
	network = network_setup(node_number = node_number, photon_detection_probability = photon_detection_prob,qubit_number = qubit_number, noiseless = noiseless, node_distance = node_distance, photo_distance = photo_distance, detuning = detuning, electron_T2 = electron_T2, electron_T1 = electron_T1, carbon_T1 = carbon_T1, carbon_T2=carbon_T2, single_instruction=single_instruction, B_osc = B_osc, no_z_precission=no_z_precission, frame = frame, wait_detuning=wait_detuning, clk_local = clk_local)
	
	# Start the global controller functionality and give the global controller its instructions
	send_protocol = Global_cont_Protocol(network = network, input_prog = line_reader, clk = clk)
	send_protocol.start()

	print(ns.sim_run()) 
	
	# If you want to print the electron and carbon states, printstates can be set to True
	if printstates == True:
		printer(network)
    #example of hos to store data
	data_storer({"result":0, "photon": 0, "time":0}, "test.json")

	
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
		main(node_number = 2,qubit_number=3, printstates=True, detuning = 0, electron_T2= 0, carbon_T2 = 0 ,single_instruction = True, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 0)

	elif sys.argv[1] == "Pool":
		with Pool() as p:
			arg = [ (1,None,2,3), (2,None,2,3), (3,None,2,3), (4,None,2,3), (5,None,2,3), (6,None,2,3), (7,None,2,3), (8,None,2,3), (9,None,2,3), (10,None,2,3) ]
			p.starmap(main,arg)
