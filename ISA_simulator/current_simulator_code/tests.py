from cmath import inf
import netsquid as ns
from NVNetwork_setup import network_setup
from netsquid.qubits.qubitapi import *
import numpy as np
from NVprotocols import *
from multiprocessing import Process, Queue, Pool, cpu_count, parent_process
from netsquid.qubits.qformalism import QFormalism, set_qstate_formalism
import json
import os
import time
import scipy
import random
import pandas as pd
import itertools
import sys # in order to take command line inputs
from multiprocessing import Pool, Manager
import multiprocessing
set_qstate_formalism(QFormalism.DM)

def main(savename = None, filename = None, node_number = 4,qubit_number = 2, photon_detection_prob = 1, printstates = False, storedata = None,node_distance = 4e-3, photo_distance = 2e-3, noiseless = False, detuning = 0, electron_T2 = 0, electron_T1 = 0, carbon_T1 = 0, carbon_T2 = 0, single_instruction = True, B_osc = 400e-6,no_z_precission = 1, frame = "rotating", wait_detuning = 0, clk = 0, clk_local = 0,B_z = 40e-3,rotation_with_pi = 0,Fault_tolerance_check = False, **kwargs):
	# Set the filename, which text file needs to be read, this text file should contain QISA specific instructions
	# np.random.seed(int(kwargs["mp_process"])+10)
	random.seed(str(os.getpid()) + str(time.time()))
	start_time = time.perf_counter()
	# np.set_printoptions(precision=2, suppress = True)
	# print(f" the arguments are {kwargs}")
	# print(f"the kwargs are {kwargs}")
	if "seed" in kwargs:
		if kwargs["seed"] == True:
			noiseless = True
	# print(f"the number of nodes is {node_number} with qubits {qubit_number}")
	# print(f"savename is {savename}")
	if savename != None:
		storedata = savename
	if filename == None:
		filename = ['test_input_surface7_measurereadout_Z.txt','test_input_surface7_measurereadout_X2.txt','test_input_surface7_logicalreadout.txt','test_input_surface7_newProgram.txt','test_input_surface7_postselectionrate_total.txt','test_input_testing_program.txt']
		# filename = "test_input_own_matrix.txt"
		filename = 'test_input_surface7_measurereadout_X2_phase.txt'
		# filename = "test_input_initialize.txt"
		# filename = 'test_input_measure_new.txt'
		filename = 'test_detectcarbon.txt'
		filename = 'test_detectcarbon_function.txt'
		filename = "phase_testing.txt"
		# filename = 'test_input_GHZ_generation_modicum.txt'
		# filename = 'test_input_excite_mw.txt'
		# filename = 'test_input_GHZ_generation_plain.txt'
		# filename = 'test_input_GHZ_plain_macros.txt'
		# filename = 'test_input_GHZ_plain_macros_new_program.txt'
# 
		# filename = 'test_input_GHZ_modicus_macros.txt'
		# filename = 'test_input_GHZ_modicus_macros_new_program.txt'

		# filename = 'test_input_investigate.txt'
		# filename = 'test_input_GHZsetter.txt'
		# filename = 'test_input_cz.txt'
		# filename = 'test_input_rabi_check.txt'
		filename = "test_input_surface7_measurereadout_X_macros.txt"
		filename = 'Surface-7_1run.txt'
		# filename = 'Surface-7-logical_X_posselect.txt'
		filename = 'Surface-7-logical_X_postselect_copy.txt'
		filename = 'Surface-7-logical_X_postselect_2.txt'
		filename = 'Surface-7-logical_X_postselect_copy - Copy.txt'
		# filename = 'Surface-7_fault_tolerance_1_photonentanglement.txt'
		# filename = 'Surface-7_fault_tolerance_2_photonentanglement.txt'
		# filename = 'Surface-7_fault_tolerance_1_photonentanglement_multiple_rotations.txt'
		filename = 'Surface-7_fault_tolerance_2_photonentanglement_multiple_rotations.txt'

		# filename = 'ramsey_fringe_c13.txt'
		# filename = 'clk_test.txt'
		# filename = "test_input_surface7_measurereadout_old.txt"
		# filename = "test_input_surface7_measurereadout_old_phase.txt"

		# filename = "test_input_surface_3dplot.txt"
		# filename = 'test_input_magnetic_bias.txt'
		# filename = "test_input_memswap1.txt"
		# filename = 'ramsey_fringe_experiment.txt'
		# filename = "Ramsey_hahn_echo_experiment.txt"
		# filename = 'ramsey_fringe_test.txt'
		filename = 'ramsey_fringe_c13.txt'
		# filename = 'T1_measurement_C13.txt'
		# filename = "Matti_test.txt"
		# filename = "Last_matti_test.txt"
		# filename = "server_test.txt"

		# filename = "Extra_test_2.txt"
		# filename = 'test.txt'
		# filename = 'magnetic_bias.txt'
		# filename = "test_input_rabi_check.txt"
		# filename = 'test_input_photondetector.txt'
		# filename = 'Logical_hadamard_gate_fidelity.txt'
		# filename = 'logical_hadamard_gate_fidelity.txt'
	else:
		filename = filename[0] #fix later
		# filename = filename + '.txt'
	# The text file with proposed filename is opened and the information is stored in parameter 'lines'
	# d = os.getcwd().parents[1]
	#parameters are read in the following line
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
	elif filename == 'logical_hadamard_gate_fidelity.txt':
		# for i in range(len(lines)):
			# print('before i am here')
			# print(f'checking lines {lines[i]}')
			# print('i am here')
			# if lines[i][-1] == "iterations":
				# print('i am in here')
		for i in range(len(lines)):
			if lines[i] != '\n':
				if lines[i].split()[0] == "LDi":
					if lines[i].split()[-1] == 'Iterations':
						lines.insert(i,'LDi '+str(kwargs["iterations"])+ ' Iterations')
						lines.pop(i+1)
		# print(lines)


		
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
	printstates = bool(int(parameter_dict["printstates"]))
	node_distance = float(parameter_dict["NodeDistance"])
	photo_distance = float(parameter_dict["PhotonDistance"])
	detuning = float(parameter_dict["Detuning"])
	T2detuning = bool(int(parameter_dict["T2Detuning"]))
	electron_T2 = float(parameter_dict["ElectronDecoherence"])
	electron_T1 = float(parameter_dict["ElectronRelaxation"])
	carbon_T2 = float(parameter_dict["CarbonDecoherence"])
	carbon_T2_on = bool(int(parameter_dict["CarbonDecoherenceEffect"]))
	carbon_T1 = float(parameter_dict["CarbonRelaxation"])
	single_instruction = bool(int(parameter_dict["CrosstalkOff"]))
	no_z_precission = int(parameter_dict["NoZPrecession"])
	frame = str(parameter_dict["frame"])
	wait_detuning = float(parameter_dict["WaitDetuning"])
	clk = float(parameter_dict["clk"])
	clk_local = float(parameter_dict["clkLocal"])
	B_z = float(parameter_dict["StaticField"])
	rotation_with_pi = int(parameter_dict["RotationWithPi"])
	DephasingEntanglementNoise = bool(int(parameter_dict["DephasingEntanglementNoise"]))
	NoisyEntangledState = bool(int(parameter_dict["NoisyEntangledState"]))
	print(f'dephasing parameter in tests {DephasingEntanglementNoise}')
	
	# Setup the network by calling the network setup function
	network = network_setup(node_number = node_number, photon_detection_probability = photon_detection_prob,qubit_number = qubit_number, noiseless = noiseless, node_distance = node_distance, photo_distance = photo_distance, detuning = detuning, electron_T2 = electron_T2, electron_T1 = electron_T1, carbon_T1 = carbon_T1, carbon_T2=carbon_T2, single_instruction=single_instruction, B_osc = B_osc, no_z_precission=no_z_precission, frame = frame, wait_detuning=wait_detuning, clk_local = clk_local,B_z = B_z,rotation_with_pi = rotation_with_pi, carbon_T2_on=carbon_T2_on)
	network.noiseless = noiseless
	network.noise_parameters = {}
	network.noise_parameters["T2_carbon"] = carbon_T2
	network.noise_parameters["DephasingEntanglementNoise"] = DephasingEntanglementNoise
	network.noise_parameters["NoisyEntangledState"] = NoisyEntangledState
	network.noise_parameters["T2Detuning"] = T2detuning
	# prog_copy_for_check = line_reader
	# print('first check in progcopy')
	# print(line_reader)
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
	# Start the simulator run of netsquid
	# start_time = time.time()
	print(ns.sim_run()) 
	# duration = time.time() - start_time
	
	# If the states are wanted to be printed, printstates can be set to True
	if printstates == True:
		printer(network)
	
	# If data is wanted to be stored, the memory names should be addressed and a filename to store them in needs to be given
	# if storedata != None:
	# 	data_to_store = {"duration":duration}
	# 	# for i in range(len(storedata)-1):
	# 	# 	data_to_store.append(network.subcomponents["controller"].memory[storedata[i]])
	# 	data_storer(data_to_store, storedata+'.json')
	timeduration = time.perf_counter() - start_time
	# with open("/home/fwmderonde/virt_env_simulator/ISA_simulator/json_data_storage/duration_time_surface-7-multipleprocessingtool"+str(savename)+".json", 'w') as file_object:
	# 				json.dump({"time:":timeduration}, file_object)
	# carbon = network.get_node("nvnode0").qmemory.peek(1)[0]
	# carbon_2 = network.get_node("nvnode1").qmemory.peek(1)[0]
	# carbon_3 = network.get_node("NVnode0").qmemory.peek(2)[0]
	# carbon_4 = network.get_node("NVnode1").qmemory.peek(2)[0]

	
	# matrix_new = network.qubit_total
	# matrix_new = reduced_dm([carbon,carbon_2,carbon_3,carbon_4])
	# matrix_new = reduced_dm([carbon,carbon_2])
	# matrix_new = reduced_dm(carbon)
 
	# sign_matrix = np.sign(matrix_new)
	
	# fidelities = network.get_node("controller").register_dict["fidelity"]
	memory_values = {}
	if line_reader[-2].split()[0].lower() == "DataStorageName".lower():
		data_stored_name = line_reader[-2].split()[1]

		if len(sys.argv) >1:
			# memory_values["noiseless"] = True
			data_stored_name = line_reader[-2].split()[1]
			data_stored_name = data_stored_name+'_noiseless'
		else:
			# memory_values["noiseless"] = False
			data_stored_name = data_stored_name+'_noisefull'
		memory_values["noiseless"] = noiseless
	if line_reader[-1].split()[0].lower() == "OutputStore".lower():
		for memory in line_reader[-1].split()[1:]:
			memory_values[memory] = (network.get_node("controller").memory[memory.lower()])
		memory_values["parameters"] = parameter_dict
		zeros = [0+0j]*16
		# print(f"qubit store next {network.qubit_store}")
		qubit_state = []
		if network.qubit_store.any() != 0:
			for values in network.qubit_store:
				qubit_intermediate = []
				for items in values:
					qubit_intermediate.append(str(items))
				# print(values)
				# print(qubit_intermediate)
				qubit_state.append(qubit_intermediate)
			# qubit_state = [str(x) for x in network.qubit_store]
			memory_values["qubit_state"] = qubit_state

		# data_storer(memory_values,data_stored_name+".json")
		
	# counter_list = network.get_node("controller").memory['SuccesRegPerMeasure'.lower()]
	# measure_list = network.get_node("controller").memory['MeasureValuePerMeasure'.lower()]
	# measure_amount = network.get_node("controller").memory['MeasureAmountMemoryValue'.lower()]
	# Total_succes_count = network.get_node("controller").memory['MemoryCount'.lower()]
	# sweepAngle = network.get_node("controller").memory['MemoryAngle'.lower()]
	# total_measurement_value = network.get_node("controller").memory['XLMeasureValue'.lower()]
	
	# dec = str(electron_T2)+"B"+str(B_osc)+"detuning"+str(detuning)
	# dec = str(carbon_T2)+"B"+str(B_osc)+"detuning"+str(detuning)
 
	# measure_result = network.get_node("controller").memory['measurevaluemem']
	# time_measure = np.arange(0,50001,100)
	# print(measure_result)
	# time_measure = np.arange(0.0002118,0.00022,0.00000012)
 
 
	# print("printing list values")
	# print(counter_list)
	# print(measure_list)
	# np.set_printoptions(precision=1)
	
	# Get the electron and carbon states
	# print('the new matrix is ')
	# print(matrix_new)
	# data_storer({"state":abs(matrix_new).tolist(), "signs": sign_matrix.real.tolist()}, "Surface-7_first_state_Siddhant.json")
	
 
	# data_storer({"state":abs(matrix_new).tolist(), "signs": sign_matrix.real.tolist(), "fidelity":fidelities }, "fidelity_values_surface_7_2_photon_entanglement_check_decoherence_detuning_dmrep_latest"+str(savename)+"_100meas.json")
	# data_storer({"state":abs(matrix_new).tolist(), "signs": sign_matrix.real.tolist(), "fidelity":fidelities }, "fidelity_values_surface_7_2_photon_entanglement_check_dmrep_latest_100meas_initialisation.json")

	
 
	# data_storer({"result":measure_result, "time":time_measure.tolist()}, "ramsey_fringe_carbon_for_experiment_withwaitdetuning_extra.json")
	
	# savename = "clk_"+str(clk_local)+"_test.json"
	# data_storer({"result":measure_result, "Clk":clk_local}, savename)
 
	
	# data_storer({"result":network.get_node("NVnode0").subcomponents["local_controller"].memAddr_result, "photon": network.get_node("NVnode0").subcomponents["local_controller"].memAddr_photon_count, "time":network.get_node("NVnode0").subcomponents["local_controller"].memAddr_time}, "ramsey_fringe_default.json")
	# data_storer({"result":0, "photon": 0, "time":0}, "test.json")

	# data_storer({"photonCount":network.get_node("controller").memory['PhotonCount']}, 'photoncount_p'+str(photon_detection_prob)+'.json')
	# data_storer({"photon": network.get_node("NVnode0").subcomponents["local_controller"].memAddr_photon_count, "frequency":network.get_node("NVnode0").subcomponents["local_controller"].memAddr_freq}, "test_magnetic_bias_10ns_second_peak.json")
	# print("now printing state within memory")
	# print(network.qubit_total)
	
	# data_storer({'counter_values_per_measure':counter_list,"measure_values_per_measure":measure_list,"measure_amount_total":measure_amount,"total_memory_count":Total_succes_count,"angle":sweepAngle,"total_measurment_amount_per_sweep":total_measurement_value}, "new_surface-7_results-sweep_check_2_photonentanglement_decoherence_detuning_dmrep_latest"+str(savename)+"_100meas.json")
	# data_storer({'counter_values_per_measure':counter_list,"measure_values_per_measure":measure_list,"measure_amount_total":measure_amount,"total_memory_count":Total_succes_count,"angle":sweepAngle,"total_measurment_amount_per_sweep":total_measurement_value}, "new_surface-7_results-sweep_check_2_dmrep_latest_100meas_intialisation.json")
	# return_value = kwargs["mp_queue"].get()
	memory_values["DataStorageName"] = data_stored_name
	if "mp_queue" in kwargs:
		return_value = {}
		
		return_value["values"] = memory_values
		print(f"the following value will be put {return_value}")
		kwargs["mp_queue"].put(return_value)
	else:
		return memory_values

	
	
# Save the data by use of the following function
def data_storer(data,store_name):
	path = os.path.realpath(__file__)
	dir = os.path.dirname(path)
	dir = dir.replace('current_simulator_code', 'json_data_storage')
	# save_direct = d.chdir("..")'
	filename = store_name
	fileopener = dir+"//" +filename
	# fileopener = '/workspaces/Thesis/ISA_simulator/json_data_storage/'+filename
	# fileopener = '/home/fwmderonde/virt_env_simulator/ISA_simulator/json_data_storage/'+filename
	with open(fileopener, 'w') as file_object:
		json.dump(data, file_object)
	# print(f"the data is {data}")
	# data_send = pd.DataFrame(data)
	# data_send.to_json(fileopener)
def printer(network):
	
	# Set the print precision of the density matrices, this is added for clarity.
	np.set_printoptions(precision=2, suppress = True)
	
	# Get the electron and carbon states
	# q_elec = network.get_node("nvnode0").qmemory.peek(0)[0]
	# q_elec_2 = network.get_node("NVnode1").qmemory.peek(0)[0]
	q_elec = network.get_node("nvnode0").qmemory.peek(0)[0]
	q_elec_2 = network.get_node("nvnode1").qmemory.peek(0)[0]

	carbon = network.get_node("nvnode0").qmemory.peek(1)[0]
	# carbon_2 = network.get_node("nvnode1").qmemory.peek(1)[0]
	# carbon_3 = network.get_node("NVnode0").qmemory.peek(2)[0]
	# carbon_4 = network.get_node("NVnode1").qmemory.peek(2)[0]
	# print("carbon state paired and then 1")
	# print(carbon.qstate.qrepr)
	# print(reduced_dm(carbon))
	# print("carbon state 2")
	# print(reduced_dm(carbon_2))
	print("electron state 1,2 and paired")
	print(reduced_dm(q_elec))
	print(reduced_dm(q_elec_2))
	print(q_elec.qstate.qrepr)
	carbon = network.get_node("nvnode0").qmemory.peek(1)[0]
	carbon_2 = network.get_node("nvnode1").qmemory.peek(1)[0]
	carbon_3 = network.get_node("nvnode0").qmemory.peek(2)[0]
	carbon_4 = network.get_node("nvnode1").qmemory.peek(2)[0]
	carbon_5 = network.get_node("nvnode0").qmemory.peek(3)[0]
	carbon_6 = network.get_node("nvnode1").qmemory.peek(3)[0]
	carbon_7 = network.get_node("nvnode0").qmemory.peek(4)[0]
	carbon_8 = network.get_node("nvnode1").qmemory.peek(4)[0]
 

	# print(reduced_dm(q_elec))
	# print(q_elec.qstate.qrepr)
	# print(network.get_node("nvnode0").subcomponents["microwave"].parameters)

	print(carbon.qstate.qrepr)
	# # print(carbon_2.qstate.qrepr)
 
	# print(reduced_dm(carbon))
	# # print(reduced_dm(carbon_2))
	# # print(reduced_dm([carbon,carbon_3,q_elec_2,q_elec_4]))

	# # print("combined carbon state")
	# print(reduced_dm([q_elec,carbon_3]))
	# print(reduced_dm(carbon_3))
	# print('density matrix after measure')
	# print(reduced_dm([q_elec,q_elec_2,carbon,carbon_2]))
	# print(reduced_dm([q_elec,q_elec_2,carbon_3]))
	# print("now printing state of carbon 1 and 2 on nvnode0")
	# print(reduced_dm([carbon,carbon_2]))
	# print("now printing state of carbon 1 and 2 on nvnode1")
	# print(reduced_dm([carbon_2,carbon_4]))
	# print("now printing state of carbon 3  on nvnode0 and 1")
	# print(reduced_dm([carbon_5,carbon_6]))
	# print("now printing state of carbon 1 and 2  on nvnode0 and 1")
	# print(reduced_dm([carbon,carbon_2,carbon_3,carbon_4]))
	# print("now printing state of electron and carbon 1 and 2  on nvnode0 and 1")
	# print(reduced_dm([q_elec,carbon_2,carbon_3,carbon_4]))
	# print("second time printing state of electron and carbon 1 and 2  on nvnode0 and 1")
	# print(reduced_dm([q_elec,q_elec_2,carbon_3,carbon_4]))
	# print("now printing state of carbon 3 and 4 on nvnode0 and 1")
	# print(reduced_dm([carbon_5,carbon_6,carbon_7,carbon_8]))
	# measure(carbon_8)
	# measure(carbon_2)
	# measure(carbon)

 
	# print("now printing state within memory")
	# print(network.qubit_total)
	print('the carbon state is')
	print(carbon.qstate.qrepr)
	print(carbon_2.qstate.qrepr)

	# print(reduced_dm([carbon_5,carbon_7,carbon_6,carbon_8]))
 

 
	
	# print(reduced_dm([carbon,carbon_2,carbon_3,carbon_4]))

def run_multiprocess(
	# code: code_type,
	# decoder: decoder_type,
	# error_rates: dict = {},
	iterations: int = 1,
	QISA_file = None,
	# decode_initial: bool = True,
	# seed: Optional[float] = None,
	processes = None,
	noiseless = None,
	# benchmark: Optional[BenchmarkDecoder] = None,
	**kwargs,
	):
	"""Runs surface code simulation using multiple processes.
	Using the standard module `.multiprocessing` and its `~multiprocessing.Process` class, several processes are created that each runs its on contained simulation using `run`. The ``code`` and ``decoder`` objects are copied such that each process has its own instance. The total number of ``iterations`` are divided for the number of ``processes`` indicated. If no ``processes`` parameter is supplied, the number of available threads is determined via `~multiprocessing.cpu_count` and all threads are utilized.
	If a `.BenchmarkDecoder` object is attached to ``benchmark``, `~multiprocessing.Process` copies the object for each separate thread. Each instance of the the decoder thus have its own benchmark object. The results of the benchmark are appended to a list and addded to the output.
	See `run` for examples on running a simulation.
	Parameters
	----------
	code
		A surface code instance (see initialize).
	decoder
		A decoder instance (see initialize).
	error_rates
		Dictionary for error rates (see `~qsurface.errors`).
	iterations
		Total number of iterations to run.
	decode_initial
		Decode initial code configuration before applying loaded errors.
	seed
		Float to use as the seed for the random number generator.
	processes
		Number of processes to spawn.
	benchmark
		Benchmarks decoder performance and analytics if attached.
	kwargs
		Keyword arguments are passed on to every process of run.
	"""
	# if hasattr(code, "figure"):
	#     raise TypeError("Cannot use surface code with plotting enabled for multiprocess.")
	path = os.path.realpath(__file__)
	dir = os.path.dirname(path)
	dir = dir.replace('current_simulator_code', 'QISA_schemes')
	# save_direct = d.chdir("..")'
	fileopener = dir+"/" +QISA_file[0]
	with open(fileopener) as f:
		lines = f.readlines()
	for i in range(len(lines)):
		if lines[i] != '\n':
			print(lines[i])
			if lines[i].split()[0] == "LDi":
				if lines[i].split()[-1] == 'Iterations':
					print(iterations)
					iterations = int(lines[i].split()[1])
	
	
	if processes is None:
		processes = cpu_count()
		processes = 4
	# processes = 2
	process_iters = iterations // processes
	print(f'get the process values {processes} with process iters {process_iters}')
	if process_iters == 0:
		print("Please select more iterations")
		return

 

    # if decode_initial:
    #     if code.superoperator_enabled:
   #         code.init_superoperator_errors()
   #         code.superoperator_random_errors()
   #     else:
   #         code.random_errors(**error_rates) #Applying random errors on the current code
	#     decoder.decode(**kwargs)    
	#     code.logical_state #Get the current logical state
	# QISA_file = QISA_file

		# lines.insert(2*i+2,'Initialize q0')
	# Initiate processes
	# manager = Manager()
	# return_dict = manager.dict()
	workers = []
	mp_ques = []
	mp_queue = Queue()
	
	print(f"the kwargs are {kwargs}")
	for process in range(processes):
		print('i made changes')
		print(f"noiseless value is {noiseless}")
		mp_ques.append(Queue())
		# time.sleep(1)
		# random.seed(str(os.getpid()) + str(time.time()))
		
		workers.append(
			Process(
				target=main,
				args=(None,QISA_file),
				kwargs={
					"iterations": process_iters,
					# "decode_initial": False,
					"seed": noiseless,
					# "noiseless": noiseless,
					"mp_process": process,
					"mp_queue": mp_queue,
					# "error_rates": error_rates,
					# "benchmark": benchmark,
					**kwargs,
				},
			)
		)
	print("Starting", processes, "workers.")



	# Start and join processes
	# print(f"the workers are {workers}")
	# print(f"with parent {workers.parent}")
	for worker in workers:
		# time.sleep(1)
		worker.start()

	
	# print('this far we get')
	outputs = []
	outputs_second = []

	# for i, worker in enumerate(workers):
	# 	print(f"the worker values are {mp_ques[i].get()}")
	# 	outputs.append(mp_ques[i].get())
	# 	# worker.join()
	# print(f"the output values are {outputs}")
	
	for worker in workers:
		worker.join()
	# print(f" the size is {mp_queue.qsize()}")
	for i, worker in enumerate(workers):
		# print(f"the worker values are {mp_ques[i].get()}")
		values = mp_queue.get()
		# print(f"the values are {values}")
		outputs.append(values)
	print(f"the output values are {outputs}")
	# print(f"the second outputs are {outputs_second}")
	output = {"P_values": []}



	for partial_output in outputs:
		output["P_values"] += partial_output["values"]["P_value"]
	output["Noise_parameters"] = outputs[0]["values"]['parameters']
	output["number_of_cores"] = processes
	output["number_of_iterations"] = iterations
	if "DataStorageName" in outputs[0]["value"]:
	# if outputs[0]["values"].has_key("DataStorageName"):
		output["DataStorageName"] = output[0]["values"]["DataStorageName"]
	# output[]
	# if benchmark:
	#     benchmarks = [partial_output["benchmark"] for partial_output in outputs]



        # if len(benchmarks) == 1:
        #     output["benchmark"] = benchmarks[0]
        # else:
        #     combined_benchmark = {}
        #     stats = defaultdict(lambda: {"mean": [], "std": []})
        #     iterations = []
        #     for benchmark in benchmarks:
        #         print(benchmark)
        #         iterations.append(benchmark["iterations"])
        #         for name, value in benchmark.items():
        #             if name[-4:] == "mean":
        #                 stats[name[:-4]]["mean"].append(value)
        #             elif name[-3:] == "std":
        #                 stats[name[:-3]]["std"].append(value)
        #             else:
        #                 if type(value) in [int, float] and name in combined_benchmark:
        #                     combined_benchmark[name] += value
        #                 else:
        #                     combined_benchmark[name] = value
        #     for name, meanstd in stats.items():
        #         mean, std = _combine_mean_std(meanstd["mean"], meanstd["std"], iterations)
        #         combined_benchmark[f"{name}mean"] = mean
        #         combined_benchmark[f"{name}std"] = std
        #     output["benchmark"] = combined_benchmark
        # output["benchmark"]["seed"] = seed

 

	return output


if __name__ == "__main__":
	multiprocessing.set_start_method('spawn')

	# for i in range(1):
	
	# 	duration_list = []
	# 	for j in range(1):
	# 		start_time = time.time()
	# 		# for i in range(25):
	# 			# detuning = i*250
	# 			# detuning = 0
	# 		qubit_number = 27
	# 		main(node_number = 1,qubit_number=qubit_number, printstates=False, detuning = 0,single_instruction=True)
	# 		duration_list.append(time.time() - start_time)
		# with open("duration_time_qubits_qgatee_new"+str(qubit_number)+".json", 'w') as file_object:
		# 	json.dump({"time:":duration_list}, file_object)
	
 
	# for i in range(1,100):
	# 	start_time = time.perf_counter()
	# 	clk_local = i*1e6
	# 	main(node_number = 1,qubit_number=2, printstates=False, detuning = 0, electron_T2= 0, carbon_T2 = 0 ,single_instruction = True, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=1, clk = 0, clk_local = clk_local)
	# 	print(f"iterator value is {i} for clk_local {clk_local}")
	# 	print(f"elapsed time is {time.perf_counter()-start_time}")
	# main(node_number = 1,qubit_number=2, printstates=True, detuning = 0, electron_T2= 0, carbon_T2 = 0 ,single_instruction = True, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=1, clk = 0, clk_local = 1e8)


	arguments = []
	duration_list = []
	start_time = time.perf_counter()
	if len(sys.argv) == 1:
		results = main(node_number = 2,qubit_number=3, printstates=True, detuning = 1e3, electron_T2= 5e3, carbon_T2 = 5e6 ,single_instruction = False, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 10e6)
		end_time = time.perf_counter()-start_time
		results["time_duration"] = end_time
		if "DataStorageName" in results:
		# if results.has_key("DataStorageName"):
			storagename = str(results.pop("DataStorageName"))
			data_storer(results,storagename+".json")
	elif sys.argv[1] == "noiseless":
		results = main(node_number = 2,qubit_number=5, printstates=True, detuning = 0, electron_T2= 5e3, carbon_T2 = 56e6 ,single_instruction = True, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 0,B_z = 0.1890,rotation_with_pi = 0)
		end_time = time.perf_counter()-start_time
		results["time_duration"] = end_time
		if "DataStorageName" in results:
		# if results.has_key("DataStorageName"):
			storagename = str(results.pop("DataStorageName"))
			data_storer(results,storagename+".json")
		# duration_list.append(time.perf_counter() - start_time)
		#carbon decoherence time is taken from paper sent by nic
		# with open("/home/fwmderonde/virt_env_simulator/ISA_simulator/json_data_storage/duration_time_surface-7_itself_4cores_4tasks.json", 'w') as file_object:
		# 		json.dump({"time:":duration_list}, file_object)
	elif sys.argv[1] == "Pool":
		with Pool() as p:
			# arg = [ (1,None,2,5,0,False,None,4e-3,2e-3,True,0,5e3,5e6), (2,None,2,3), (3,None,2,3), (4,None,2,3), (5,None,2,3), (6,None,2,3), (7,None,2,3), (8,None,2,3), (9,None,2,3), (10,None,2,3) ]
			arg = [(i,None,2,5,0,False,None,4e-3,2e-3,True,0,5e3,0,0,5e6) for i in range(5)]

			p.starmap(main,arg)
	elif sys.argv[1] == "supercomputer":
		iterations = 50
		if len(sys.argv) >2:
			noiseless_arg = True
		else:
			noiseless_arg = False
		QISA_file = ["logical_hadamard_gate_fidelity.txt"]
		results = run_multiprocess(iterations=iterations,QISA_file = QISA_file, noiseless = noiseless_arg)
		end_time = time.perf_counter()-start_time
		results["time_duration"] = end_time
		if "DataStorageName" in results:
		# if results.has_key("DataStorageName"):
			storagename = str(results.pop("DataStorageName"))
			data_storer(results,storagename+".json")
		else:
			data_storer(results,"first_measurement_values_noiseless.json")
		# plot_points[size].append((rate, no_error /iterations))

			# duration_list.append(time.perf_counter() - start_time)
			# with open("/workspaces/Thesis/ISA_simulator/json_data_storage/duration_time_surface-7_itself_4cores_4tasks.json", 'w') as file_object:
			# 		json.dump({"time:":duration_list}, file_object)

		# for i in range(0,int(sys.argv[2])):
		# 	qubit_number = i+2
		# 	main(node_number = 1,qubit_number=qubit_number, printstates=False, detuning = 0, electron_T2= 0, carbon_T2 = 0 ,single_instruction = True, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 0)
		# 	duration_list.append(time.perf_counter() - start_time)
		# 	with open("/home/fwmderonde/virt_env_simulator/ISA_simulator/json_data_storage/duration_time_surface-7_itself.json", 'w') as file_object:
		# 			json.dump({"time:":duration_list}, file_object)
	# else:
	# 	savename = str(sys.argv[1])
	# 	filename = str(sys.argv[2])
	# 	arguments = [savename, filename]
	# 	for i in range(3,len(sys.argv)):
	# 		arguments.append(float(sys.argv[i]))
	# 	main(*arguments)

	# main(node_number = 1,qubit_number=2, printstates=True, detuning = 0, electron_T2= 5e6, carbon_T2 = 5e6,single_instruction = True, no_z_precission=1,B_osc = 400e-3)
	#carbon T1 = 5e3
		# prepare photondetector value

		# for i in range(101):
		# 	p = i/50
		# 	main(node_number = 1, photon_detection_prob = p,qubit_number=1, printstates=False, detuning = 0)
		# 	print(f"p is now {p}")
		# get T2 time as input value, make it today
		

		# for i in range(1):
