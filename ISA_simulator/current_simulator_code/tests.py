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
import pandas as pd
import itertools
import sys # in order to take command line inputs
from multiprocessing import Pool

set_qstate_formalism(QFormalism.DM)

def main(savename = None, filename = None, node_number = 4,qubit_number = 2, photon_detection_prob = 1, printstates = False, storedata = None,node_distance = 4e-3, photo_distance = 2e-3, noiseless = False, detuning = 0, electron_T2 = 0, electron_T1 = 0, carbon_T1 = 0, carbon_T2 = 0, single_instruction = True, B_osc = 400e-6,no_z_precission = 1, frame = "rotating", wait_detuning = 0, clk = 0, clk_local = 0,B_z = 40e-3,rotation_with_pi = 0,Fault_tolerance_check = False):
	# Set the filename, which text file needs to be read, this text file should contain QISA specific instructions
	start_time = time.perf_counter()
	# np.set_printoptions(precision=2, suppress = True)

	print(f"the number of nodes is {node_number} with qubits {qubit_number}")
	print(f"savename is {savename}")
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
		# filename = 'ramsey_fringe_c13.txt'
		# filename = 'T1_measurement_C13.txt'
		# filename = "Matti_test.txt"
		# filename = "Last_matti_test.txt"
		# filename = "server_test.txt"

		# filename = "Extra_test_2.txt"
		# filename = 'test.txt'
		# filename = 'magnetic_bias.txt'
		# filename = "test_input_rabi_check.txt"
		# filename = 'test_input_photondetector.txt'
		filename = 'Logical_hadamard_gate_fidelity.txt'
	else:
		filename = filename + '.txt'
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
	network = network_setup(node_number = node_number, photon_detection_probability = photon_detection_prob,qubit_number = qubit_number, noiseless = noiseless, node_distance = node_distance, photo_distance = photo_distance, detuning = detuning, electron_T2 = electron_T2, electron_T1 = electron_T1, carbon_T1 = carbon_T1, carbon_T2=carbon_T2, single_instruction=single_instruction, B_osc = B_osc, no_z_precission=no_z_precission, frame = frame, wait_detuning=wait_detuning, clk_local = clk_local,B_z = B_z,rotation_with_pi = rotation_with_pi)
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
			memory_values["noiseless"] = True
			data_stored_name = line_reader[-2].split()[1]
			data_stored_name = data_stored_name+'_noiseless'
		else:
			memory_values["noiseless"] = False
			data_stored_name = data_stored_name+'_noisefull'
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

		data_storer(memory_values,data_stored_name+".json")
		
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




if __name__ == "__main__":
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
		main(node_number = 2,qubit_number=3, printstates=True, detuning = 1e3, electron_T2= 5e3, carbon_T2 = 5e6 ,single_instruction = False, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 10e6)
	elif sys.argv[1] == "noiseless":
		main(node_number = 2,qubit_number=5, printstates=True, detuning = 0, electron_T2= 5e3, carbon_T2 = 56e6 ,single_instruction = True, no_z_precission=1,B_osc = 400e-6, frame = "rotating", wait_detuning=0, clk = 0, clk_local = 0,B_z = 0.1890,rotation_with_pi = 0)
		# duration_list.append(time.perf_counter() - start_time)
		#carbon decoherence time is taken from paper sent by nic
		# with open("/home/fwmderonde/virt_env_simulator/ISA_simulator/json_data_storage/duration_time_surface-7_itself_4cores_4tasks.json", 'w') as file_object:
		# 		json.dump({"time:":duration_list}, file_object)
	elif sys.argv[1] == "Pool":
		with Pool() as p:
			# arg = [ (1,None,2,5,0,False,None,4e-3,2e-3,True,0,5e3,5e6), (2,None,2,3), (3,None,2,3), (4,None,2,3), (5,None,2,3), (6,None,2,3), (7,None,2,3), (8,None,2,3), (9,None,2,3), (10,None,2,3) ]
			arg = [(i,None,2,5,0,False,None,4e-3,2e-3,True,0,5e3,0,0,5e6) for i in range(5)]

			p.starmap(main,arg)
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
