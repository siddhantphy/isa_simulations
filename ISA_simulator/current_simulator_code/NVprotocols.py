from logging import raiseExceptions
import netsquid as ns
from netsquid.components.instructions import INSTR_X, INSTR_Y, INSTR_Z
from netsquid.protocols import  Protocol, Signals
import numpy as np
from noise_models import DephasingNoiseModelCarbon
from netsquid.components.instructions import IGate
from netsquid.qubits.qubitapi import dephase
import math
from Hardware_to_logical_computations import get_analytical_logical_expectation_values
from math import pi
from netsquid.qubits.qubitapi import *
from NVPrograms import Programs_microwave
from netsquid.qubits.operators import *
from datetime import datetime, timedelta
# from netsquid_nv.photon_noise import NVPhotonEmissionNoiseModel
from NV_photonemission import NVPhotonEmissionNoiseModel
from pydynaa.core import EventExpression
import sys
sys.setrecursionlimit(20000)
import threading
from netsquid_nv.magic_distributor import NVSingleClickMagicDistributor#, NVDoubleClickMagicDistributor
threading.stack_size(2**30) # Needed to perform magnetic biasing sequence

class PhotodetectorProtocol(Protocol):
	def __init__(self, photodetector, p=0.95):
		super().__init__()
		self.photodetector = photodetector
		self.p = p

	def run(self):
		port_in = self.photodetector.ports[self.photodetector.port_names[0]]
		while True:
			yield self.await_port_input(port_in)
			self.photodetector.photoncountreg = [self.photodetector.photoncountreg[0]+np.random.binomial(1,self.p)]
			
### This protocol is added in order to be able to send data from local controllers to global controllers			
class Global_data_Protocol(Protocol):
	def __init__(self, network,node):
		super().__init__()
		self.node = node
		self.controller = network.subcomponents["controller"]
	
	def run(self):
		# The input port of the global controller is defined
		port_in = self.controller.ports["In_"+self.node+"data"]
		while True:
			yield self.await_port_input(port_in)
			message = port_in.rx_input()
			self.controller.register_dict[message.items[0][0]] = message.items[0][1]
			self.controller.register_dict[message.items[1][0]] = message.items[1][1]


### The next protocol is used to make the Global controller start scheduling the instructions
class Global_cont_Protocol(Protocol):
	def __init__(self, network,input_prog, clk):
		super().__init__()
		self.network = network
		self.controller = network.get_node("controller")
		self.input_prog = input_prog
		self.clk = clk
		zeros = [0+0j]*16
		self.network.qubit_total = np.diag(zeros)
		self.network.qubit_store = np.diag(zeros)
		self.controller.register_dict["fidelity"] = []
		self.clk_cycles = self.controller.clk_cycle_dict
		self.clk_flag = 0
		self.controller.memory["P_value".lower()] = []

		for i in range(len(self.network.nodes)-1):
			self.add_subprotocol(Global_data_Protocol(network,"nvnode"+str(i)), "Global_data_protocol_nvnode"+str(i))

	def run(self):
		self.input_prog = [item_in_list.lower() for item_in_list in self.input_prog]
		for j in range(len(self.network.nodes)-1):
			self.input_prog.insert(j,"init q"+str(j))
		# Check if there is a LABEL statement in the instructionset. If so, remember the increment value of this instruction
		for i, long_items in enumerate(self.input_prog):
			items = long_items.split() #split every input instruction into loose words, so the intruction word can be checked
			if items[0][-1] == ":": # Check instruction word for equality to LABEL
				self.controller.LABEL_dict[items[0][:-1]] = i # Remember the increment value for the LABEL and store it in a dictionary with LABEL name
			elif items[0] == "label": # Check instruction word for equality to LABEL
				self.controller.LABEL_dict[items[1]] = i # Remember the increment value for the LABEL and store it in a dictionary with LABEL name
		# print(f"the amount of nv centers is {len(self.network.nodes)}")

		
		i = -1 # Set first value for i at -1, because i is incremented after the while loop and needs to start at 0
		while i < (len(self.input_prog)-1): # Loop through all the instructions
			i +=1
			items = self.input_prog[i].lower().split() # Split the instructions into loose words, so the instruction word can be checked
			if items[0] == "{":

				message_NV_center_1 = self._decompose_single_instruction(self.input_prog[i+1].lower().split())
				message_NV_center_2 = self._decompose_single_instruction(self.input_prog[i+2].lower().split())
				NV_center_1 = self.input_prog[i+1].lower().split()[1]
				NV_center_2 = self.input_prog[i+2].lower().split()[1]
				if NV_center_1[0] == "q":
					NV_center_number_1 = NV_center_1[1:]
					NV_center_number_2 = NV_center_2[1:]

				else:
					NV_center_number_1 = NV_center_1[6:]
					NV_center_number_2 = NV_center_2[6:]

				port_out_1 = self.controller.ports["Out_nvnode"+str(NV_center_number_1)]
				port_out_2 = self.controller.ports["Out_nvnode"+str(NV_center_number_2)]
				port_out_1.tx_output(message_NV_center_1)
				port_out_2.tx_output(message_NV_center_2)
				evt_wait_port_1 = self.await_port_input(port_out_1)
				evt_wait_port_2 = self.await_port_input(port_out_2)
				yield evt_wait_port_2 and evt_wait_port_1

				# port_out.tx_output(items)
				# yield self.await_port_input(port_out)
				i +=3 #this will result in skipping performing the operations multiple times (because they are already done in parallel)
				if self.input_prog[i].lower().split()[0] != "}":
					raise ValueError("parallel operations have gone wrong, more then two parallel instructions are not supported")
			elif items[0] == "nventangle_magic": # Perform magic entanglement by use of statesampler

				# Get the 2 nodes for which the magic entanglement needs to be implemented
				node1 = self.network.get_node(items[1]) 
				node2 = self.network.get_node(items[2])
				counter = 0
				process_done = False
				while process_done == False:
					# Set values needed for delivery
					self.distance = 0
					electron_pos = 0
					if len(items) == 4:
						other_parameters = {"length_A": 10,
										"length_B": 10,
										"p_loss_length_A": 0.1,
										"p_loss_length_B": 0.1,
										"p_loss_init_A": 0.9,
										"p_loss_init_B": 0.9,
										"detector_efficiency": 0.5,
										'dark_count_probability':0,
										'visibility':0.9}
					else:
						other_parameters = {"length_A": 0,
									"length_B": 0,
									"p_loss_length_A": 0,
									"p_loss_length_B": 0,
									"p_loss_init_A": 0,
									"p_loss_init_B": 0,
									"detector_efficiency": 1,
									'dark_count_probability':0,
									'visibility':1}



					memory_positions = {node1.ID: electron_pos, node2.ID: electron_pos}


					# Get the distributor for the state distribution
					distributor = NVSingleClickMagicDistributor(nodes=[node1, node2], length_A=self.distance, length_B=self.distance)
					
					# Deliver the sampled states
					delivery = distributor.add_delivery(memory_positions=memory_positions, alpha=0.00000001, cycle_time=0.0001, **other_parameters)
					evt_expr = EventExpression(event_id = delivery.any_id) # Get event id, for scheduling purposes
					# print(distributor.get_label(delivery))
					yield evt_expr # Wait for the event expression to be done in order to get the next instruction
					yield self.await_timer(1)
					# print(distributor._last_label[1])
					if str(distributor._last_label[1]) == "BellIndex.B11":
						counter +=1
						# print("this is not correct")
					else:
						process_done = True
						# print('this is correct')
				# print('within nventangle magic')
				# print(f"the counter value is {counter}")
				
			
			elif items[0] == "printstate":
				np.set_printoptions(precision=2, suppress = True)

				# node1 = self.network.get_node(items[1]) 
				# print(f"printing the state of node {items[1]} qubit {items[2]} due to printstate statement")
				# print(node1.qmemory.peek(int(items[2]))[0].qstate.qrepr)
				qubit_store_list = []
					# print(int((len(items)-2)/2)+2)
					# print(range(2,int(len(items)+1),2))
				for i in range(1,int((len(items))),2):
					# print(i)
					# print(f"the value of i is {i} with node nvnode {str(items[i][1:])}")
					qubit_store_list.append(self.network.get_node("nvnode"+str(items[i][1:])).qmemory.peek(int(items[i+1]))[0])
					# i +=1
				qubit_matrix = reduced_dm(qubit_store_list)
				print(qubit_matrix)
			

			elif items[0] == "state_inject_ghz_2":
				# Get the nodes for which the GHZ states needs to be set
				Nodes = [self.network.get_node(items[1]), self.network.get_node(items[2])]

				# Make an empty list of electrons, so the electrons can be added for the nodes
				qubits = []

				# Add electrons of the nodes to the list
				print(items[3])
				qubits.append(Nodes[0].qmemory.peek(int(items[3]))[0])
				qubits.append(Nodes[1].qmemory.peek(int(items[4]))[0])
				# for items_node in Nodes:
				# 	qubits.append(items_node.qmemory.peek(items[3])[0])
				
				# Make the wanted GHZ state
				Zerolist = [0]*2
				checker = [0.5]+Zerolist+[0.5]
				checker_no = [0.5]+Zerolist+[-0.5]
				checker_no_2 = [-0.5]+Zerolist+[0.5]

				zeroline = [0]*4
				electron_GHZ = np.array([checker,zeroline,zeroline,checker])
				electron_GHZ_nogo = np.array([checker_no,zeroline,zeroline,checker_no_2])

				# Assign GHZ state to the electrons
				# print(items[3])
				# if items[3] == "correct":
				assign_qstate(qubits, electron_GHZ)

			elif items[0] == "state_inject_ghz_4":
				# Get the nodes for which the GHZ states needs to be set
				Nodes = [self.network.get_node(items[1]), self.network.get_node(items[2])]

				# Make an empty list of electrons, so the electrons can be added for the nodes
				qubits = []

				# Add electrons of the nodes to the list
				# print(items[3])
				qubits.append(Nodes[0].qmemory.peek(int(items[3]))[0])
				qubits.append(Nodes[1].qmemory.peek(int(items[5]))[0])
				qubits.append(Nodes[0].qmemory.peek(int(items[4]))[0])
				qubits.append(Nodes[1].qmemory.peek(int(items[6]))[0])
				# for items_node in Nodes:
				# 	qubits.append(items_node.qmemory.peek(items[3])[0])
				
				# Make the wanted GHZ state
				Zerolist = [0]*14
				checker = [0.5]+Zerolist+[0.5]
				checker_no = [0.5]+Zerolist+[-0.5]
				checker_no_2 = [-0.5]+Zerolist+[0.5]

				zeroline = [0]*16
				electron_GHZ = np.array([checker,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,zeroline,checker])
				electron_GHZ_nogo = np.array([checker_no,zeroline,zeroline,checker_no_2])
				if items[7] == 'surface':
					rotation_angle = float(items[8]) if items[8][0].isdigit() else self.controller.register_dict[items[8]]
					rotation_phase = float(items[9]) if items[9][0].isdigit() else self.controller.register_dict[items[9]]
					cos_value = np.cos(rotation_angle/2)**2/(np.sqrt(np.cos(rotation_angle/2)**4+np.sin(rotation_angle/2)**4))/(np.sqrt(2))
					sin_value = np.sin(rotation_angle/2)**2*(np.cos(rotation_phase)+1j*np.sin(rotation_phase))/(np.sqrt(np.cos(rotation_angle/2)**4+np.sin(rotation_angle/2)**4))/(np.sqrt(2))
					# upper_line = [0.5]+[0]*14+[0.5]
					# middle_line = [0]*16
					# middle_value_line = [0]
					# upper_line_1 = [0]*5+[0.5]+[0]*4+[0.5]+[0]*5
					perfect_state = np.array([cos_value, 0, 0,0, 0, sin_value,0, 0, 0,0, sin_value,0, 0, 0,0, cos_value])#.astype(np.float64)
					# dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
					q0,q1,q2,q3 = create_qubits(4)
					self.network.qubit_store = perfect_state
					assign_qstate(qubits,perfect_state)
				
				# Assign GHZ state to the electrons
				else:
					assign_qstate(qubits, electron_GHZ)
			elif items[0] == "nventangle_real": #link paper here
				Nodes = [self.network.get_node(items[1]), self.network.get_node(items[2])]
				electrons = []
				
				# Add electrons of the nodes to the list
				for items_node in Nodes:
					electrons.append(items_node.qmemory.peek(0)[0])
				
				# Make the wanted GHZ state
				Zerolist = [0]*2
				checker = [0.5]+Zerolist+[0.5]
				checker_no = [0.5]+Zerolist+[-0.5]
				checker_no_2 = [-0.5]+Zerolist+[0.5]
				# entangled_state = entangle_create()
				# def entangle_create():
				# if one assumes pdet = pdeta, pdetb, pdc = 0, alpha = alpha_a = alpha_b and V = 1, then the fidelity of pab with the closest bell satte is F = 1-alpha and the generation rate is rab = 2alpha pdet
				alpha_A = 0.07
				alpha_B = 0.05
				p_det_A = 3.6e-4
				p_det_B = 4.4e-4
				p_dc = 1.5e-7
				V = 0.9
				plusmin = 0
				counter = 0
				gen_rate = 0

				while ((plusmin != 1) or (gen_rate !=1)):
					plusmin = np.random.binomial(1,0.5)*2-1
					# gen_rate = np.random.binomial(1,2*alpha_A*p_det_A)
					gen_rate = np.random.binomial(1,1/100)
					counter +=1
				# print(f"the counter value is {counter}")
				# while gen_rate !=1:
				# 	gen_rate = np.random.binomial(1,2*alpha_A*p_det_A)
				# 	counter +=1
				charge_state_fail_NV0 = np.random.binomial(1,0)
				charge_state_fail_NV1 = np.random.binomial(1,0)

				# print(f"the charge state fail is {charge_state_fail}")
				# if charge_state_fail:
					# entangle_create()
				# print(f"check plusmin probability {plusmin}")
				## determine entanglement parameters
				#duration time of total entanlgement, max of 2.5 ms
				p00 = alpha_A*alpha_B*(p_det_A+p_det_B+2*p_dc)
				p01 = alpha_A*(1-alpha_B)*(p_det_A+2*p_dc)
				p10 = alpha_B*(1-alpha_A)*(p_det_B+2*p_dc)
				p11 = 2*(1-alpha_A)*(1-alpha_B)*p_dc
				p_tot = p00+p01+p10+p11
				electron_entangled_state = np.array([[p00/p_tot,0,0,0],[0,p01/p_tot,plusmin*np.sqrt(V*p01*p10)/p_tot,0],[0,plusmin*np.sqrt(V*p01*p10)/p_tot,p10/p_tot,0],[0,0,0,p11/p_tot]])
				zeroline = [0]*4

				## calculate dephasing noise
				# dephasing_noise = alpha_A / 2 * (1 - np.exp(-(2 * np.pi * dw * td * 10 ** (-6)) ** 2 / 2))

				if self.network.noiseless:
					electron_GHZ = np.array([checker,zeroline,zeroline,checker])
					# print(f"the qubit state is {electron_GHZ}")
					assign_qstate(electrons, electron_entangled_state)
				else:
					if charge_state_fail_NV0 == 1:
						self.network.get_node(items[1]).subcomponents["nv_center_quantum_processor"].NV_state = "NV0"
					if charge_state_fail_NV1 == 1:
						self.network.get_node(items[2]).subcomponents["nv_center_quantum_processor"].NV_state = "NV0"
					if (charge_state_fail_NV0 == 0 and charge_state_fail_NV1 == 0):
						operation_duration_time = 5555.55*counter #5555.55 comes from 2.5(ms)/450(entanglement attempts) look at the paper
						# print(f"the operation duration time is {operation_duration_time}")
						time_message_to_NV_centers = ['wait', operation_duration_time]
						port_out_1 = self.controller.ports["Out_nvnode0"]
						port_out_2 = self.controller.ports["Out_nvnode1"]
						port_out_1.tx_output(time_message_to_NV_centers)
						port_out_2.tx_output(time_message_to_NV_centers)
						# print('do i see this')
						evt_wait_port_1 = self.await_port_input(port_out_1)
						evt_wait_port_2 = self.await_port_input(port_out_2)
						yield evt_wait_port_2 and evt_wait_port_1
						# print(f"the qubit state is {electron_GHZ}")
						
						assign_qstate(electrons, electron_entangled_state)
						diamond = self.network.get_node(items[1]).subcomponents["nv_center_quantum_processor"]
						# for k in range(len(diamond.mem_positions)-2): #-2 because there is a memory position for the electron and an additional one for a photon emission (only needed for mathmetical perposes)
						# The iterater value skips the value for the electron position
							
							# iterater_value = k+1
						delta_w = 0
						tau_decay = 0
						p_deph = (1 - alpha_A) / 2 * (1 - np.exp(-(delta_w * tau_decay) ** 2 / 2))
						error_model = DephasingNoiseModelCarbon(prob_of_dephasing=p_deph)
						# number_of_attempts = NVSingleClickMagicDistributor._get_number_of_attempts(delivery)
						carbon_qubits = [diamond.peek(carbon_iterator_value) for carbon_iterator_value in range(len(diamond.mem_positions)-2)]
						error_model.noise_operation_own(
							qubits=carbon_qubits,
							number_of_applications=counter,
							diamond = diamond)
						# yield self.await_timer(1)
					# assign_qstate(electrons, electron_GHZ)

					
				
				# electron_GHZ_real = np.array([[p00,0,0,0],[0,p01, plusmin*np.sqrt(V*p01*p10),0],[0, plusmin*np.sqrt(V*p01*p10),p10,0],[0,0,0,p11]])
				# electron_GHZ_real = np.multiply(electron_GHZ_real,1/p_tot) #perfect config is alpha 0.5 and 0.5
				# return electron_GHZ
				
				# yield self.await_port_input(port_out_1)
				# yield self.await_port_input(port_out_2)
				
    
			elif items[0] == "ghz_setter_2n" or items[0] == "nventangle":
				# Get the nodes for which the GHZ states needs to be set
				Nodes = [self.network.get_node(items[1]), self.network.get_node(items[2])]

				# Make an empty list of electrons, so the electrons can be added for the nodes
				electrons = []

				# Add electrons of the nodes to the list
				for items_node in Nodes:
					electrons.append(items_node.qmemory.peek(0)[0])
				
				# Make the wanted GHZ state
				Zerolist = [0]*2
				checker = [0.5]+Zerolist+[0.5]
				checker_no = [0.5]+Zerolist+[-0.5]
				checker_no_2 = [-0.5]+Zerolist+[0.5]

				zeroline = [0]*4
				electron_GHZ = np.array([checker,zeroline,zeroline,checker])
				electron_GHZ_nogo = np.array([checker_no,zeroline,zeroline,checker_no_2])

				# Assign GHZ state to the electrons
				# print(items[3])
				# if items[3] == "correct":
				assign_qstate(electrons, electron_GHZ)
				# else:
				# 	assign_qstate(electrons, electron_GHZ_nogo)
					
			

			# Added in order for the instructions not to crash.
			elif items[0] == "label" or items[0][-1] == ":":
				pass
			elif items[0] == "datastoragename" or items[0] == "outputstore":
				pass
			
			elif items[0] == "fault_inject":
				injection = items[2].lower()
				instruction_dict = {'x':INSTR_X, 'y':INSTR_Y, 'z':INSTR_Z}
				position = float(items[3])
				self.controller.get_node("nvnode"+items[1][-1]).execute_instruction(instruction_dict[injection], [position])
			elif items[0] == "qgatee":
				offset = len(items[1][:-1])
				port_out = self.controller.ports["Out_nvnode"+items[1][offset:]]
				if items[3][0].isdigit() or items[3][0] == '-':
					items[3] = float(items[3])
				else:
					items[3] = self.controller.register_dict[items[3]]
				if items[2][0].isdigit() or items[2][0] == '-':
					items[2] = float(items[3])
				elif items[2][0] == "[":
					checker = items[2].split(',')
					# for i in range(3):
					if checker[0][1:] in self.controller.register_dict:
						checker[0] = self.controller.register_dict[checker[0][1:]]
					else:
						checker[0] = checker[0][1:]
					if checker[1][:] in self.controller.register_dict:
						checker[1] = self.controller.register_dict[checker[1][:]]
					else:
						checker[1] = checker[1]
					if checker[2][:1] in self.controller.register_dict:
						checker[2] = self.controller.register_dict[checker[0][:1]]
					else:
						checker[2] = checker[2][:1]
						
					items[2] = "["+str(checker[0])+","+str(checker[1])+","+str(checker[2])+"]"
				else:
					items[2] = str(self.controller.register_dict[items[2]])
				items.remove(items[1])
				port_out.tx_output(items)
				yield self.await_port_input(port_out)
    
			elif items[0] == "qgateuc" or items[0] == "qgatecc":
				offset = len(items[1][:-1])
				port_out = self.controller.ports["Out_nvnode"+items[1][offset:]]
				if items[4][0].isdigit() or items[4][0] == '-':
					items[4] = float(items[4])
				else:
					items[4] = self.controller.register_dict[items[4]]
				if items[3][0].isdigit() or items[3][0] == '-':
					items[3] = float(items[3])
				elif items[3][0] == "[":
					checker = items[3].split(',')
					# for i in range(3):
					if checker[0][1:] in self.controller.register_dict:
						checker[0] = self.controller.register_dict[checker[0][1:]]
					else:
						checker[0] = checker[0][1:]
					if checker[1][:] in self.controller.register_dict:
						checker[1] = self.controller.register_dict[checker[1][:]]
					else:
						checker[1] = checker[1]
					if checker[2][:1] in self.controller.register_dict:
						checker[2] = self.controller.register_dict[checker[0][:1]]
					else:
						checker[2] = checker[2][:1]
						
					items[3] = "["+str(checker[0])+","+str(checker[1])+","+str(checker[2])+"]"
				else:
					items[3] = str(self.controller.register_dict[items[3]])
				items.remove(items[1])
				port_out.tx_output(items)
				yield self.await_port_input(port_out)
			elif items[0] == "qgatez":
				offset = len(items[1][:-1])
				port_out = self.controller.ports["Out_nvnode"+items[1][offset:]]
				items.remove(items[1])
				port_out.tx_output(items)
				yield self.await_port_input(port_out)
			# Perform set command
			elif items[0] == "set":
				# Read the component to set a parameter value for
				component = items[2]

				# Read the parameter which needs to be set
				parameter = items[3]
				
				# Read the value to which the parameter needs to be set
				value = float(items[4])

				# Check if the component is in the network directly, if not, send the command through to the node
				if component in dict(self.network.subcomponents).keys():
					# Set value for the component in the network
					self.network.subcomponents[component].parameters[parameter] = value
				else:
					# Send the command to the corresponding local controller
					port_out = self.controller.ports["Out_nvnode"+items[1][1:]]
					if items[1][:1] != 'q' or items[1][:1] != 'nvnode':
						raise ValueError("The last input must give the node to perform the instruction on")
					items.remove(items[1])				
					port_out.tx_output(items)
					yield self.await_port_input(port_out)

			
			# Print statement, added to make it easier to see what happens within the microcode
			elif items[0] == "print":
				print(items[1:])

			# Implementation of BR statement	
			elif items[0] == "br":
				# Check if first value is negative
				if items[1][0] == "-":
					items[1] = -1*int(items[1][1])

				elif items[1].isdigit():
					items[1] = int(items[1])
				else:
					# If the value is not a number, it must be a register and the value within the register is needed
					items[1] = self.controller.register_dict[items[1]]
				
				# Check if second numerical value is negative
				if items[3][0] == "-":
					items[3] = -1*int(items[3][1])
				elif items[3].isdigit():
					items[3] = int(items[3])
				else:
					# If the value is not a number, it must be a register and the value within the register is needed
					items[3] = self.controller.register_dict[items[3]]
				if items [2] == "<":					
					if items[1] < items[3]:
						# If the BR statement is true, the code jumps to the LABEL location
						i = self.controller.LABEL_dict[items[4]]
				if items [2] == ">":
					if items[1] > items[3]:
						# If the BR statement is true, the code jumps to the LABEL location
						i = self.controller.LABEL_dict[items[4]]
			
			# Jump to LABEL location
			elif items[0] == "jump":
				i = self.controller.LABEL_dict[items[1]]

			# ADD integer value to the mentioned register	
			elif items[0] == "addi":
				self.controller.register_dict[items[1]]+=int(items[2])
			
			elif items[0] == "subi":
				self.controller.register_dict[items[1]]-=int(items[2])
			
			elif items[0] == "wait":
				if items[2] in self.controller.register_dict:
					items[2] = self.controller.register_dict[items[2]]
				items.remove(items[1])
				port_out.tx_output(items)
				yield self.await_port_input(port_out)
			
			# Additional instruction used for printing memory and register values
			elif items[0] == "printer":
				if items[1] in self.controller.register_dict:
					print(self.controller.register_dict[items[1]])
				else:
					print(self.controller.memory[items[1]])		

			elif items[0] == 'statestore':
				# upper_line = [0.5]+[0]*14+[0.5]
				# middle_line = [0]*16
				# upper_line_1 = [0]*5+[0.5]+[0]*4+[0.5]+[0]*5
				# dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
   
				# carbon = self.network.get_node("nvnode0").qmemory.peek(1)[0]
				# carbon_2 = self.network.get_node("nvnode1").qmemory.peek(1)[0]
				# carbon_3 = self.network.get_node("nvnode2").qmemory.peek(1)[0]
				# carbon_4 = self.network.get_node("nvnode3").qmemory.peek(1)[0]
				# carbon_1 = self.network.get_node("nvnode0").qmemory.peek(3)[0]
				# carbon_2 = self.network.get_node("nvnode1").qmemory.peek(3)[0]
				# carbon_3 = self.network.get_node("nvnode0").qmemory.peek(4)[0]
				# carbon_4 = self.network.get_node("nvnode1").qmemory.peek(4)[0]
				if items[1] == 'surface':
					# rotation_angle = float(items[2])
					# rotation_phase = float(items[3])
					# rotation_angle = float(items[2]) if items[2][0].isdigit() else self.controller.register_dict[items[2]]
					# rotation_phase = float(items[3]) if items[3][0].isdigit() else self.controller.register_dict[items[3]]
					
					# cos_value = np.cos(rotation_angle/2)**2/(np.sqrt(np.cos(rotation_angle/2)**4+np.sin(rotation_angle/2)**4))/(np.sqrt(2))
					# sin_value = np.sin(rotation_angle/2)**2*(np.cos(rotation_phase)+1j*np.sin(rotation_phase))/(np.sqrt(np.cos(rotation_angle/2)**4+np.sin(rotation_angle/2)**4))/(np.sqrt(2))
					# upper_line = [0.5]+[0]*14+[0.5]
					# print(f"the values for the matrix are {cos_value} and {sin_value}")
					# middle_line = [0]*16
					# middle_value_line = [0]
					# upper_line_1 = [0]*5+[0.5]+[0]*4+[0.5]+[0]*5
					# perfect_state = np.array([cos_value, 0, 0,0, 0, sin_value,0, 0, 0,0, sin_value,0, 0, 0,0, cos_value])#.astype(np.float64)
					# dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
					# q0,q1,q2,q3 = create_qubits(4)
					# assign_qstate([q0,q1,q2,q3],perfect_state)
					qubit_store_list = []
					# print(int((len(items)-2)/2)+2)
					# print(range(2,int(len(items)+1),2))
					for i in range(2,int((len(items))),2):
						# print(i)
						# print(f"the value of i is {i} with node nvnode {str(items[i][1:])}")
						qubit_store_list.append(self.network.get_node("nvnode"+str(items[i][1:])).qmemory.peek(int(items[i+1]))[0])
						# i +=1
					# carbon_matrix = reduced_dm([carbon_1,carbon_2, carbon_3, carbon_4])
					# print(qubit_store_list)
					qubit_matrix = reduced_dm(qubit_store_list)
					np.trace(qubit_matrix)
					print('sup')
					# self.network.qubit_total += carbon_matrix
					self.network.qubit_store = qubit_matrix
					# self.network.qubit_total += qubit_matrix 

					# q4,q5,q6,q7 = create_qubits(4)
					# assign_qstate([q4,q5,q6,q7],carbon_matrix)
					# fidelity_value = fidelity([carbon_1,carbon_2, carbon_3, carbon_4],q0.qstate.qrepr)
					# # fidelity_value = fidelity([carbon_1,carbon_2, carbon_3, carbon_4],q0.qstate.qrepr)
					# self.controller.register_dict["fidelity"].append(fidelity_value)
					# print(f"the perfect state is {perfect_state}")
					# print(f"the state of the qubit is {q0.qstate.qrepr}")
					# print(f"the state of the qubit is {qubit_matrix}")
					# print(f"the fidelity value is {fidelity_value}")
			elif items[0] == "logical_analysis":
				#call function create_analytical_logcial_PTM
				#pass on qubit matrix
				# print("hello i have succes")
				qubit_store_list = []
					# print(int((len(items)-2)/2)+2)
					# print(range(2,int(len(items)+1),2))
				for i in range(1,int((len(items))),2):
					# print(i)
					# print(f"the value of i is {i} with node nvnode {str(items[i][1:])}")
					qubit_store_list.append(self.network.get_node("nvnode"+str(items[i][1:])).qmemory.peek(int(items[i+1]))[0])
					# i +=1
				# carbon_matrix = reduced_dm([carbon_1,carbon_2, carbon_3, carbon_4])
				# print(qubit_store_list)
				qubit_matrix = reduced_dm(qubit_store_list)
				np.trace(qubit_matrix)
				# print('sup')
				# self.network.qubit_total += carbon_matrix
				# self.network.qubit_store = qubit_matrix
				# qubit_matrix = self.network.qubit_store
				# p_store = []
				p = np.real(get_analytical_logical_expectation_values(qubit_matrix))
				# self.network.p_value = p
				p = [p[0],p[1],p[2]]
				# self.p_store.append(p)
				# print(p)
				# np.array(p)
				self.controller.memory["P_value".lower()].append(p)

			elif items[0] == 'fidelity_calc':
				upper_line = [0.5]+[0]*14+[0.5]
				middle_line = [0]*16
				upper_line_1 = [0]*5+[0.5]+[0]*4+[0.5]+[0]*5
				dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
   
				# carbon = self.network.get_node("nvnode0").qmemory.peek(1)[0]
				# carbon_2 = self.network.get_node("nvnode1").qmemory.peek(1)[0]
				# carbon_3 = self.network.get_node("nvnode2").qmemory.peek(1)[0]
				# carbon_4 = self.network.get_node("nvnode3").qmemory.peek(1)[0]
				carbon_1 = self.network.get_node("nvnode0").qmemory.peek(3)[0]
				carbon_2 = self.network.get_node("nvnode1").qmemory.peek(3)[0]
				carbon_3 = self.network.get_node("nvnode0").qmemory.peek(4)[0]
				carbon_4 = self.network.get_node("nvnode1").qmemory.peek(4)[0]
				if items[1] == 'surface':
					# rotation_angle = float(items[2])
					# rotation_phase = float(items[3])
					# rotation_angle = float(items[2]) if items[2][0].isdigit() else self.controller.register_dict[items[2]]
					# rotation_phase = float(items[3]) if items[3][0].isdigit() else self.controller.register_dict[items[3]]
					
					# cos_value = np.cos(rotation_angle/2)**2/(np.sqrt(np.cos(rotation_angle/2)**4+np.sin(rotation_angle/2)**4))/(np.sqrt(2))
					# sin_value = np.sin(rotation_angle/2)**2*(np.cos(rotation_phase)+1j*np.sin(rotation_phase))/(np.sqrt(np.cos(rotation_angle/2)**4+np.sin(rotation_angle/2)**4))/(np.sqrt(2))
					# upper_line = [0.5]+[0]*14+[0.5]
					# middle_line = [0]*16
					# middle_value_line = [0]
					# upper_line_1 = [0]*5+[0.5]+[0]*4+[0.5]+[0]*5
					perfect_state = self.network.qubit_store
					# dz_perfect_0 = np.array([upper_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,middle_line,upper_line]).astype(np.float64)
					q0,q1,q2,q3 = create_qubits(4)
					assign_qstate([q0,q1,q2,q3],perfect_state)
					if items[2] == "XL":
						ns.qubits.operate(q0,ns.X)
						ns.qubits.operate(q2,ns.X)
					elif items[2] == "YL":
						ns.qubits.operate(q0,ns.Y)
						ns.qubits.operate(q2,ns.Y)
					elif items[2] == "ZL":
						ns.qubits.operate(q0,ns.Z)
						ns.qubits.operate(q2,ns.Z)
					elif items[2] == "HL":
						ns.qubits.operate(q0,ns.H)
						ns.qubits.operate(q2,ns.H)
					carbon_matrix = reduced_dm([carbon_1,carbon_2, carbon_3, carbon_4])
					self.network.qubit_total += carbon_matrix
					q4,q5,q6,q7 = create_qubits(4)
					assign_qstate([q4,q5,q6,q7],carbon_matrix)
					fidelity_value = fidelity([carbon_1,carbon_2, carbon_3, carbon_4],q0.qstate.qrepr)
					# fidelity_value = fidelity([carbon_1,carbon_2, carbon_3, carbon_4],q0.qstate.qrepr)
					self.controller.register_dict["fidelity"].append(fidelity_value)
					print(f"the perfect state is {perfect_state}")
					print(f"the state of the qubit is {q0.qstate.qrepr}")
					print(f"the state of the qubit is {carbon_matrix}")
					print(f"the fidelity value is {fidelity_value}")


				# q_elec_4 = self.network.get_node("nvnode3").qmemory.peek(0)[0]
				# q_elec_2 = self.network.get_node("nvnode1").qmemory.peek(0)[0]

				
				# q0,q1,q2,q3 = create_qubits(4)
				# assign_qstate([q0,q1,q2,q3],dz_perfect_0)
				# q4,q5,q6,q7 = create_qubits(4)


				# if items[1] == "plain":
				# 	qubit_matrix = reduced_dm([carbon,carbon_3,q_elec_2,q_elec_4])

				# 	self.network.qubit_total += qubit_matrix
				# 	assign_qstate([q4,q5,q6,q7],qubit_matrix)
				# 	# print(q4.qstate.qrepr)
				# 	# print(q0.qstate.qrepr)
				# 	fidelity_value = fidelity([q4,q5,q6,q7],q0.qstate.qrepr)
				# 	self.controller.register_dict["fidelity"].append(fidelity_value)

				# else:
				# 	carbon_matrix = reduced_dm([carbon,carbon_2, carbon_3, carbon_4])
				# 	self.network.qubit_total += carbon_matrix
				# 	assign_qstate([q4,q5,q6,q7],carbon_matrix)
				# 	fidelity_value = fidelity([q4,q5,q6,q7],q0.qstate.qrepr)
				# 	self.controller.register_dict["fidelity"].append(fidelity_value)
				
				# print(f"the state is {q4.qstate.qrepr}")
				# print(f"the qubit matrix state is {qubit_matrix}")
				# print(f"vs the made state {q0.qstate.qrepr}")
				# print(f"the fidelity value is {fidelity_value}")

			
			# Added instruction to make GHZ checking be done easier
			elif items[0] == "determine_GHZ_measurement":
				m_0 = self.network.get_node("nvnode0").subcomponents["local_controller"].ResultRegisters[0]
				m_1 = self.network.get_node("nvnode1").subcomponents["local_controller"].ResultRegisters[0]
				m_2 = self.network.get_node("nvnode2").subcomponents["local_controller"].ResultRegisters[0]
				m_3 = self.network.get_node("nvnode3").subcomponents["local_controller"].ResultRegisters[0]
				measurement_GHZ = m_0*m_1*m_2*m_3
				self.network.subcomponents["controller"].memAddr[0] = measurement_GHZ
			
			# Send excite_mw statement through
			elif items[0] == "excite_mw" or items[0] == "mw" or items[0] == "rf":
				port_out = self.controller.ports["Out_nvnode"+items[1][1:]]
				for j in range(len(items)-2):
					if items[j+2][0].isdigit():
						items[j+2] = float(items[j+2])
					else:
						items[j+2] = self.controller.register_dict[items[j+2]]
				items.remove(items[1])
				port_out.tx_output(items)
				yield self.await_port_input(port_out)




			# Store a value, from register into dictionary
			elif items[0] == "st":
				if items[2] in self.controller.memory:
					self.controller.memory[items[2]].append(self.controller.register_dict[items[1]])
				else:
					# print(self.controller.memory)
					# print(self.controller.register_dict)
					self.controller.memory[items[2]] = [self.controller.register_dict[items[1]]]
			
			# Add a value from one register to another register
			elif items[0] == "add":
				self.controller.register_dict[items[1]] = self.controller.register_dict[items[2]]+ self.controller.register_dict[items[3]]
			
			# Substract a value from one register to another register
			elif items[0] == "sub":
				self.controller.register_dict[items[1]] = self.controller.register_dict[items[2]]- self.controller.register_dict[items[3]]
			
			# Load an integer value into a register
			elif items[0] == "ldi":
				self.controller.register_dict[items[2]] = float(items[1])
			
			# Copy a value from 1 register into another register
			elif items[0] == "mov":
				self.controller.register_dict[items[1]] = self.controller.register_dict[items[2]]
			
			# Multiply the values of 2 registers together and store it in a register
			elif items[0] == "mul":
				self.controller.register_dict[items[1]] = int(self.controller.register_dict[items[2]])*int(self.controller.register_dict[items[3]])
			
			# Send every other command through to the local controller.
			else:
				
				if items[1][:1] != 'q' and items[1][:-1] != 'nvnode':
					raise ValueError(f"The last input must give the node to perform the instruction on, fix instruction {items}")
				offset = len(items[1][:-1])
				
				self.clk_flag = 1
				port_out = self.controller.ports["Out_nvnode"+items[1][offset:]]
				items.remove(items[1])
				port_out.tx_output(items)
				yield self.await_port_input(port_out)
			if self.clk !=0:		
				if self.clk_flag ==1:
					time = self.clk_cycles["rest"]*self.clk
					yield self.await_timer(time)
					self.clk_flag = 0
				else:
					time = self.clk_cycles[items[0]]*self.clk
					yield self.await_timer(time)
	
	def _decompose_single_instruction(self,items):
		if items[0] == "qgatee":
			offset = len(items[1][:-1])
			port_out = self.controller.ports["Out_nvnode"+items[1][offset:]]
			if items[3][0].isdigit() or items[3][0] == '-':
				items[3] = float(items[3])
			else:
				items[3] = self.controller.register_dict[items[3]]
			if items[2][0].isdigit() or items[2][0] == '-':
				items[2] = float(items[3])
			elif items[2][0] == "[":
				checker = items[2].split(',')
				# for i in range(3):
				if checker[0][1:] in self.controller.register_dict:
					checker[0] = self.controller.register_dict[checker[0][1:]]
				else:
					checker[0] = checker[0][1:]
				if checker[1][:] in self.controller.register_dict:
					checker[1] = self.controller.register_dict[checker[1][:]]
				else:
					checker[1] = checker[1]
				if checker[2][:1] in self.controller.register_dict:
					checker[2] = self.controller.register_dict[checker[0][:1]]
				else:
					checker[2] = checker[2][:1]
					
				items[2] = "["+str(checker[0])+","+str(checker[1])+","+str(checker[2])+"]"
			else:
				items[2] = str(self.controller.register_dict[items[2]])
			items.remove(items[1])
			

		elif items[0] == "qgateuc" or items[0] == "qgatecc":
			offset = len(items[1][:-1])
			if items[4][0].isdigit() or items[4][0] == '-':
				items[4] = float(items[4])
			else:
				items[4] = self.controller.register_dict[items[4]]
			if items[3][0].isdigit() or items[3][0] == '-':
				items[3] = float(items[3])
			elif items[3][0] == "[":
				checker = items[3].split(',')
				# for i in range(3):
				if checker[0][1:] in self.controller.register_dict:
					checker[0] = self.controller.register_dict[checker[0][1:]]
				else:
					checker[0] = checker[0][1:]
				if checker[1][:] in self.controller.register_dict:
					checker[1] = self.controller.register_dict[checker[1][:]]
				else:
					checker[1] = checker[1]
				if checker[2][:1] in self.controller.register_dict:
					checker[2] = self.controller.register_dict[checker[0][:1]]
				else:
					checker[2] = checker[2][:1]
					
				items[3] = "["+str(checker[0])+","+str(checker[1])+","+str(checker[2])+"]"
			else:
				items[3] = str(self.controller.register_dict[items[3]])
			items.remove(items[1])
			
		elif items[0] == "qgatez":
			offset = len(items[1][:-1])
			items.remove(items[1])

		elif items[0] == "wait":
				if items[2] in self.controller.register_dict:
					items[2] = self.controller.register_dict[items[2]]
				items.remove(items[1])
		return items
	def start(self):

		super().start()
		self.start_subprotocols()

### Protocol of the beamsplitter, not used in current implementation ###
class Beamsplitter_Protocol(Protocol):
	def __init__(self, beamsplitter):
		super().__init__()
		self.beamsplitter = beamsplitter
		self.checker_value = 0
		
	
	def run(self):
		port_in = self.beamsplitter.ports[self.beamsplitter.ports_in[0]]
		port_in_2 = self.beamsplitter.ports[self.beamsplitter.ports_in[1]]
		
		ports_out = [self.beamsplitter.ports[self.beamsplitter.ports_out[0]],self.beamsplitter.ports[self.beamsplitter.ports_out[1]]]
		while True:
			evt_wait_port_1 = self.await_port_input(port_in)
			evt_wait_port_2 = self.await_port_input(port_in_2)
			expression = yield evt_wait_port_2 | evt_wait_port_1

			expression.reprime()

			self.checker_value +=1
			message = port_in.rx_input()
			message_2 = port_in_2.rx_input()

			if message != None and message_2 != None:
				qubit = message.items[0]
				qubit_2 = message_2.items[0]
				Beamsplitter_send_1 = np.random.binomial(1,0.5)
				Beamsplitter_send_2 = np.random.binomial(1,0.5)
				if Beamsplitter_send_1 == Beamsplitter_send_2:
					ports_out[Beamsplitter_send_1].tx_output([qubit,qubit_2])
				else:
					ports_out[Beamsplitter_send_1].tx_output(qubit)
					ports_out[Beamsplitter_send_2].tx_output(qubit_2)
			elif message != None:
				qubit = message.items[0]
				Beamsplitter_send = np.random.binomial(1,0.5)
				ports_out[Beamsplitter_send].tx_output(qubit)
			elif message_2 != None:
				qubit_2 = message_2.items[0]
				Beamsplitter_send = np.random.binomial(1,0.5)
				ports_out[Beamsplitter_send].tx_output(qubit_2)

	
### Protocol of the diamond, not used in current implementation ###
class Diamond_Protocol(Protocol):
	def __init__(self, nv_center, laser_protocol, no_z_precission, detuning, decay = False, single_instruction = True, frame = "rotating",Dipole_moment = None):
		super().__init__()
		self.nv_center = nv_center
		self.no_z_precission = no_z_precission
		self.detuning = detuning
		self.single_instruction = single_instruction
		self.frame = frame
		self.Dipole_moment = Dipole_moment
		self.laser_protocol = laser_protocol
		self.add_subprotocol(laser_protocol, "laser_protocol")

		# If the decay protocol is added, there is the potential for the NV center to decay into the NV0 state
		if decay == True:
			Decay_protocol = DecayProtocol(self.nv_center, decay_prob = 0 , darkcount_prob = 0)
			self.add_subprotocol(Decay_protocol, "decay_protocol")

	def run(self):
		port = self.nv_center.ports["In_classic_cont"]
		while True:
			# The message is taken from the local controller and handed off to the quantum processor 
			yield self.await_port_input(port)
			message = port.rx_input().items

			yield from self._prog_executer(message)

	def _prog_executer(self, message):
		# Before a program is executed it is made sure that the last program has finished
		for items in message:
			if self.nv_center.busy:
				yield self.await_program(self.nv_center)
			if message[0] == "excite_mw":
				yield self.await_timer(abs(float(items[2])))
			N_state = self.nv_center.peek(len(self.nv_center.mem_positions)-2)[0]
			# The quantum program which coincides with the instruction name is retrieved 
			qprogram = Programs_microwave(items, self.nv_center, no_z = self.no_z_precission, detuning = self.detuning, single_instruction = self.single_instruction, frame = self.frame, N_state = reduced_dm(N_state),Dipole_moment = self.Dipole_moment)

			# The program is executed
			self.nv_center.execute_program(qprogram)
			# After the program has started executing, it is waited until it stopped executing
			if self.nv_center.busy:
				yield self.await_program(self.nv_center)	

		# A signal is sent, the signal is connected to the local controller and the local controller will wait for this signal before sending the next instruction
		self.send_signal(signal_label = Signals.SUCCESS)
		
	def start(self):
		super().start()
		self.start_subprotocols()

# Local controller protocol implementation
class Loc_Cont_Protocol(Protocol):
	def __init__(self,controller,node, photon_detection_probability, absorption_probability,no_z_precission, decay, photon_emission_noise, detuning,single_instruction, frame, wait_detuning,Dipole_moment,rotation_with_pi):
		super().__init__()
		self.controller = controller
		self.node = node
		self.wait_detuning = wait_detuning
		self.laser_protocol = Laser_protocol(switches = node.subcomponents["switches"], diamond = node.subcomponents["nv_center_quantum_processor"], absorption_probability= absorption_probability, photon_emission_noise= photon_emission_noise)
		self.Diamond_protocol = Diamond_Protocol(nv_center = node.subcomponents["nv_center_quantum_processor"], laser_protocol = self.laser_protocol, no_z_precission = no_z_precission, decay = decay, detuning= detuning, single_instruction=single_instruction,frame = frame, Dipole_moment = Dipole_moment)
		self.Photo_protocol = PhotodetectorProtocol(photodetector = node.subcomponents["photodetector"], p = photon_detection_probability)
		self.switch_protocol = SwitchProtocol(switch=node.subcomponents["switches"])
		self.add_subprotocol(self.switch_protocol, 'switch_protocol')
		self.add_subprotocol(self.Diamond_protocol, "diamond_protocol")
		self.add_subprotocol(self.Photo_protocol, "photo_protocol")
		self.controller.tresholdregister[0] = 3
		self.clk_local = self.controller.clk_local
		self.clk_cycles = self.controller.clk_cycle_dict
		self.rotation_with_pi = rotation_with_pi

		# Determine parameters to get the value for the electron resonance frequency
		self.D = D = 2.87e9  #Hz
		self.gamma_c = gamma_c = 1.07e7 #Hz/T (gyromagnetic ratio carbon)
		self.gamma_e = gamma_e = 28.7e9 #Hz/T (gyromagnetic ratio electron)
		self.B_z = B_z = self.node.supercomponent.subcomponents['external_magnet'].parameters["magnetic_field"] # T
		self.B_osc = B_osc = self.node.subcomponents["microwave"].parameters["B_osc"]
		
		# Calculate the resonance frequency for the electron
		self.controller.elec_freq_reg[0] = D - gamma_e*B_z
		self.controller.elec_rabi_reg[0] = B_osc*gamma_e
		omega_L_c = math.copysign(B_z*gamma_c, D-gamma_e*B_z)
		for k in range(len(self.node.qmemory.mem_positions)-3):
					# The iterater value skips the value for the electron position
					iterater_value = k+1

					# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
					A_parr_c = self.node.qmemory.mem_positions[iterater_value].properties["A_parr"]

					# Calculate the larmor frequency
					# omega_L_c = abs(B_z*gamma_c)
					
					# Calculate the rabi frequency and resonance frequencies
					omega_c = B_osc*gamma_c
					omega_c_res = abs(omega_L_c-A_parr_c)
					# print(f" omega_c_res = {omega_c_res}")

					# Add the resonance and rabi frequencies to their respective registers
					self.controller.carbon_frequencies.append(omega_c_res)
					self.controller.carbon_rabi_frequencies.append(omega_c)

	def run(self):
		# Define the in and output ports of the local controller. The local controller has 1 input and output for instruction communication and 1 output for data communication
		self.port_in = port_in = self.controller.ports["In_cont"]
		self.port_out = port_out = self.controller.ports["Out_cont"]
		self.port_out_data = self.controller.ports["Out_cont_data"]
		
		while True:
			# Wait for input
			yield self.await_port_input(port_in)
			# Get the message at the input port. It contains an instruction
			message = port_in.rx_input().items
			# Make an empty list for the new decomposition message which is going to be send on into the system.
			self.message_send = message_send = []

			# The next value is used for scheduling purposes, some instructions have their own manner of registerering that they are done, thus the done value is added.
			done = 0

			# The initialisation command, this is needed, because NetSquid does not give an initial quantum state to the qubits
			# The qubits are initialised in the maximally mixed state, because it is assumed that at the start of the simulation, the qubits have not been touched for a long time
			if message[0] == "init":
				# A qubit is created and given the initial value, afterwards this value is added to the qubits in the diamond memory
				for i in range(len(self.node.qmemory.mem_positions)):
					qubit = create_qubits(1)
					assign_qstate(qubit, np.diag([1,0]))
					if i == 1:
						assign_qstate(qubit, np.diag([0.5,0.5]))
					self.node.qmemory.put(qubit,positions = [i])

				# In order to make the program continue, the done value is set to 1	
				done = 1

			# The excite mw statement is putten through to the quantum processor, with its parameters
			# This is added for easy testing purposes
			elif message[0] == "excite_mw" or message[0] == "mw" or message[0] == "rf":
				message_send.append(message)
				port_out.tx_output(message_send)

			# Prelimenary entanglement statement, instructions still need to be added
			elif message[0] == "entangle":
				pass

			# Start of the initialistaion function
			elif message[0] == "init_real" or message[0] == "initialize":
				yield from self.real_initialisation_func()
				done = 1
			
			# Start of the charge resonance check function
			elif message[0] == "crc":
				yield from self.real_crc_func()
				done = 1

			# Implementation of the magnetic biasing function
			elif message[0] == "magnetic_bias" or message[0] == "magbias":
				# The electron is first initialised
				yield from self.real_initialisation_func()
				# memory addresses are prepaired
				self.controller.memAddr = []
				self.controller.memAddr_photon_count = []
				self.controller.memAddr_freq = []
				
				# Frequency start, step and stop parameters are read from input parameters
				self.controller.sweepStopReg = [float(message[3])]
				self.controller.sweepStepReg = [float(message[2])]
				self.controller.sweepStartReg = [float(message[1])]

				# Time is read, in order to get the prediction algorithm to work
				start_sim_time = datetime.now()

				# A while loop is started, makes sure that every value is looped trough
				while self.controller.sweepStartReg[0] < self.controller.sweepStopReg[0]:

					# Call the sweep bias function, which performs a multitude of measurements for the given frequency
					yield from self.real_sweep_bias_func()

					# Get the photoncount
					self.controller.Registers[0] = self.controller.supercomponent.subcomponents["photodetector"].photoncountreg[0]

					# Store the values
					self.controller.memAddr.append([self.controller.Registers[0],self.controller.sweepStartReg[0]])
					self.controller.memAddr_photon_count.append(self.controller.Registers[0])
					self.controller.memAddr_freq.append(self.controller.sweepStartReg[0])
					
					# Get the new frequency value
					self.controller.sweepStartReg = [self.controller.sweepStartReg[0]+self.controller.sweepStepReg[0]]

					# Perform duration prediction algorithm
					print(f"the current frequency is {self.controller.sweepStartReg[0]}")
					percentage_done = (self.controller.sweepStartReg[0]-float(message[1]))/(float(message[3])-float(message[1]))*100
					print(f"The simulation is at {percentage_done:.2f}% done")
					duration = datetime.now()-start_sim_time
					simulation_time_now = duration.total_seconds()
					print(f"you have been simulating for {simulation_time_now} seconds")
					time_left = 100/(percentage_done)*simulation_time_now*(100-percentage_done)/100
					print(f"it will approximately take you {time_left} seconds to finish the simulation")
					final_time = datetime.now()+timedelta(seconds = time_left)#+timedelta(hours=1)
					print(f"your program will finish approximately by {final_time}")
					print(f"The total simulation time is estimated to be {(final_time-start_sim_time).total_seconds()} in seconds")

				done = 1
			
			# Start carbon detection algorithm
			elif message[0] == "detectcarbon":
				# Read start, step and stop register values from the input parameters
				self.controller.sweepStopReg = [float(message[3])]
				self.controller.sweepStepReg = [float(message[2])]
				self.controller.sweepStartReg = [float(message[1])]
				self.controller.measureamount = [float(message[4])]

				# Prepare memory
				self.controller.memAddr_result = [0]*(int((self.controller.sweepStopReg[0]-self.controller.sweepStartReg[0])/self.controller.sweepStepReg[0]))
				self.controller.memAddr_photon_count = [0]*(int((self.controller.sweepStopReg[0]-self.controller.sweepStartReg[0])/self.controller.sweepStepReg[0]))
				self.controller.memAddr_freq = [0]*(int((self.controller.sweepStopReg[0]-self.controller.sweepStartReg[0])/self.controller.sweepStepReg[0]))

				i = -1
				while self.controller.sweepStartReg[0] < self.controller.sweepStopReg[0]:
					i = i+1
					# Reset the measurement counter before performing the following measurements for the frequenty
					self.controller.measurecounter = 0
					while self.controller.measurecounter < self.controller.measureamount:
						# Initialise the electron
						yield from self.real_initialisation_func()
						# Prepare the message_send parameter as a list so the instructions can be added
						self.message_send = []


						# Perform algorithm needed to detect carbon nuclei
						self.message_send.append(["excite_mw", '0', np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57])
						self.message_send.append(["excite_mw", '0', np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.sweepStartReg[0], 0])
						self.message_send.append(["excite_mw", '0', np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0])
						self.message_send.append(["excite_mw", '0', np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.sweepStartReg[0], 3.14])
						self.message_send.append(["excite_mw", '0', np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0])
						self.message_send.append(["excite_mw", '0', np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 4.71])
      
						# self.message_send.append(["excite_mw", '0', 2.7365789665416314e-07/2, 1722000000, 1.57])
						# self.message_send.append(["excite_mw", '0', 0.00073401697/2, self.controller.sweepStartReg[0], 0])
						# self.message_send.append(["excite_mw", '0', 2.7365789665416314e-07, 1722000000, 0])
						# self.message_send.append(["excite_mw", '0', 0.00073401697/2, self.controller.sweepStartReg[0], 3.14])
						# self.message_send.append(["excite_mw", '0', 2.7365789665416314e-07, 1722000000, 0])
						# self.message_send.append(["excite_mw", '0', 2.7365789665416314e-07/2, 1722000000, 4.71])

						# Send the algorithm to the diamond
						self.port_out.tx_output(self.message_send)

						# Wait for feedback from the diamond
						yield self.await_signal(sender = self.subprotocols["diamond_protocol"], signal_label = Signals.SUCCESS)

						# Measure the electron
						yield from self.real_measurement_func()

						# Store the results
						self.controller.memAddr_result[i]+=self.controller.ResultRegisters[0]
						self.controller.memAddr_photon_count[i] += (self.controller.Registers[0])
						self.controller.memAddr_freq[i] = (self.controller.sweepStartReg[0])	

						# Increment the measure counter	
						self.controller.measurecounter += 1				
					
					# Get a new value for the frequency
					self.controller.sweepStartReg[0] += self.controller.sweepStepReg[0]

				done = 1
			
			# Swap a carbon nucleus to the electron in the given basis
			elif message[0] == "measure_carbon" or message[0] == "swapce":
				# Quantum algorithm for X basis measurement is performed
				if message[1] == "x":
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57, 400e-6])

					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],3.14,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],0,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])

					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 0, 400e-6])

				# Quantum algorithm for Y basis measurement is performed
				elif message[1] == "y":
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57, 400e-6])

					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],4.71,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],1.57,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])

					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2,self.controller.elec_freq_reg[0], 0, 400e-6])

				# Quantum algorithm for Z basis measurement is performed
				elif message[1] == "z":
					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],3.14,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],0,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])

					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57, 400e-6])

					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],1.57,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[int(message[2])]/2, self.controller.carbon_frequencies[int(message[2])],4.71,400e-6])
					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])

					message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 0, 400e-6])
				
				else:
					raise ValueError("The desired basis is not yet supported. Choose between 'x, y or z'")
 
				# The quantum algorithm is send to the diamond
				port_out.tx_output(message_send)

			# The real measurement function is performed
			elif message[0] == "measure_real" or message[0] == "measure" or message[0] == "measuree":
				yield from self.real_measurement_func()
				done = 1

			# The rabi check algorithm is performed
			elif message[0] == "rabi_check" or message[0] == "rabicheck":
				# Values are read from the parameters
				self.controller.startTimeReg = [float(message[1])]
				self.controller.stepTimeReg = [float(message[2])]
				self.controller.MeasureAmountReg = [float(message[4])]
				self.controller.stopTimeReg = [float(message[3])]
				# self.controller.MeasureCounterReg = [0]

				# Memory address is prepared
				# self.controller.memAddr_result = [0]*(int((self.controller.Registers[3]-self.controller.Registers[1])/self.controller.stepTimeReg[0]))
				self.controller.memAddr_result = [[]]
				for i in range(int((self.controller.stopTimeReg[0]-self.controller.startTimeReg[0])/self.controller.stepTimeReg[0])+2):
					self.controller.memAddr_result.append([])
				self.controller.memAddr_photon_count = [0]*(int((self.controller.stopTimeReg[0]-self.controller.startTimeReg[0])/self.controller.stepTimeReg[0])+2)
				self.controller.memAddr_time = [0]*(int((self.controller.stopTimeReg[0]-self.controller.startTimeReg[0])/self.controller.stepTimeReg[0])+2)
				
				# Added value to store the photon count easier
				# self.i = 0
				i = -1
				while self.controller.startTimeReg[0] < self.controller.stopTimeReg[0]:
					i = i+1
					# Reset the measurement counter before performing the following measurements for the frequenty
					self.controller.measurecounter = 0
					while self.controller.measurecounter < self.controller.MeasureAmountReg[0]:
						# Initialise the electron
						yield from self.real_initialisation_func()

						# Perform algorithm needed to detect carbon nuclei
						self.message_send = [["excite_mw", '0', self.controller.startTimeReg[0], 1722000000, 0]]
						
						# Send the algorithm to the diamond
						self.port_out.tx_output(self.message_send)

						# Wait for feedback from the diamond
						yield self.await_signal(sender = self.subprotocols["diamond_protocol"], signal_label = Signals.SUCCESS)

						# Measure the electron
						yield from self.real_measurement_func()

						# Store the results
						# self.controller.memAddr_result[i]+=self.controller.ResultRegisters[0]
						self.controller.memAddr_result[i].append(self.controller.ResultRegisters[0])
						self.controller.memAddr_photon_count[i] += (self.controller.Registers[0])
						self.controller.memAddr_time[i] = (self.controller.startTimeReg[0])	

						# Increment the measure counter	
						self.controller.measurecounter += 1				
					
					# Get a new value for the frequency
					print(self.controller.startTimeReg[0])
					self.controller.startTimeReg[0] += self.controller.stepTimeReg[0]

				done = 1

				# Call the rabi check function
				# yield from self.real_Rabi_check_func()
				# done = 1

			# Perform memory swapping algorithm
			elif message[0] == "memswap_electron_to_carbon" or message[0] == "swapec":
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],3.14,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])

				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 0, 400e-6])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],1.57,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],4.71,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				port_out.tx_output(message_send)

			elif message[0] == "full_swap":
				# message_send.append(["excite_mw","0", 2.7365789665416314e-07/2, "1722000000.0", 1.57, 400e-6])

				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],3.14,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["numerical_shift", int(message[1])+1, -np.pi/2])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 3.14, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 1.57, 400e-6])
 
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],3.14,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])

				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 0, 400e-6])

				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["numerical_shift", int(message[1])+1, np.pi/2])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57, 400e-6])

				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])],3.14,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0, 400e-6])


				port_out.tx_output(message_send)

			elif message[0] == "full_swap_own":
				# message_send.append(["excite_mw","0", 2.7365789665416314e-07/2, "1722000000.0", 1.57, 400e-6])
				#controlled X gate
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 3.1415926535898,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["numerical_shift", 0, 1.57])
				#controlled X gate from carbon to electron
				#subset hadamard electron
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				#subset hadamard carbon
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 1.57,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 1.57,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0], self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0], self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				#controlled X gate
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 3.1415926535898,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])

				message_send.append(["numerical_shift", 0, 1.57])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 1.57,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 1.57,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0], self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0], self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0]/2, self.controller.elec_freq_reg[0], 1.57,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])

				

				# controlled X gate from electron on carbon
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 3.1415926535898,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.carbon_rabi_frequencies[0]/2, self.controller.carbon_frequencies[int(message[1])], 0,400e-6])
				message_send.append(["excite_mw","0", np.pi/self.controller.elec_rabi_reg[0], self.controller.elec_freq_reg[0], 0,400e-6])
				
				message_send.append(["numerical_shift", 0, 1.57])

				port_out.tx_output(message_send)
			
			# QgateE implementation	
			elif message[0] == "qgatee":
				# Read the input parameters from the message
				axis = str(message[1])
				if self.rotation_with_pi == 1:
					angle = np.pi*float(message[2])
				else: 
					angle = float(message[2])

				# Get the value of the oscillating magnetic field
				B_osc = self.node.subcomponents["microwave"].parameters["B_osc"]

				# Calculate the corersponding phase and duration parameters
				# This ifelse statement is added to get out of the division by 0 case
				if axis[0] == "[":
					axis_split = axis.split(',')
					new_axis = [0,0,0]
					new_axis[0] = float(axis_split[0][1:])
					new_axis[1] = float(axis_split[1])
					new_axis[2] = float(axis_split[2][:1])
					
					# Calculate the corresponding phase and duration parameters for the carbon rotations
					phase = pi/2*np.sign(new_axis[1]) if new_axis[0] == 0 else np.arctan(new_axis[1]/new_axis[0])
				# phase = pi/2*np.sign(axis[3]) if float(axis[1]) == 0 else np.arctan(axis[3]/axis[1])
				else:
					if self.rotation_with_pi == 1:
						phase = np.pi*float(axis)
					else: 
						phase = float(axis)
					phase = float(axis)
				# phase = pi/2*np.sign(int(axis[3])) if float(axis[1]) == 0 else np.arctan(float(axis[3])/float(axis[1]))
				duration = angle/(self.gamma_e*B_osc)
				# Make the excite_mw statmement, which is to be sent to the microwave
				message_send.append(["excite_mw","0", duration, self.controller.elec_freq_reg[0], phase])

				# Send the microwave instruction to the microwave
				port_out.tx_output(message_send)

			# Perform direction gate on specified carbon nucleus
			elif message[0] == "qgatedir":
				# Read the input parameters from the message
				axis = str(message[2])
				if axis[0] == "[":
					axis_split = axis.split(',')
					new_axis = [0,0,0]
					new_axis[0] = float(axis_split[0][1:])
					new_axis[1] = float(axis_split[1])
					new_axis[2] = float(axis_split[2][:1])
					
					# Calculate the corresponding phase and duration parameters for the carbon rotations
					phase = pi/2*np.sign(new_axis[1]) if new_axis[0] == 0 else np.arctan(new_axis[1]/new_axis[0])
				# phase = pi/2*np.sign(axis[3]) if float(axis[1]) == 0 else np.arctan(axis[3]/axis[1])
				else:
					if self.rotation_with_pi == 1:
						phase = np.pi*float(axis)
					else: 
						phase = float(axis)
					# phase = float(axis)
				if self.rotation_with_pi == 1:
					angle = np.pi*float(message[3])
				else: 
					angle = float(message[3])
				
				carbon_nuclei = int(message[1])
				direction = int(message[4])


				# Get the value of the oscillating magnetic field
				B_osc = self.node.subcomponents["microwave"].parameters["B_osc"]

				# Calculate the corersponding phase and duration parameters for the carbon rotations
				duration = angle/(self.gamma_c*B_osc)

				# The phase remains the same, but duration needs to be changed for the electron rotation, a pi rotation needs to be done for the electron
				duration_e = np.pi/(self.gamma_e*B_osc)

				# Make the excite_mw statmement, which is to be sent to the microwave
				message_send.append(["excite_mw","0", duration, self.controller.carbon_frequencies[carbon_nuclei], phase+pi/2+pi/2*direction])
				message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0])
				message_send.append(["excite_mw","0", duration, self.controller.carbon_frequencies[carbon_nuclei], phase+pi/2-pi/2*direction])
				message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0])

				# Send the microwave instruction to the microwave
				port_out.tx_output(message_send)
			
			elif message[0] == "qgatez":
				index_qubit = message[1]
				if self.rotation_with_pi == 1:
					phase = np.pi*float(message[2])
				else: 
					phase = float(message[2])
				# phase = message[2]
				message_send.append(["numerical_shift", index_qubit, phase])
				port_out.tx_output(message_send)
    
			# Perform controlled rotation on carbon nucleus
			elif message[0] == "qgatecc":
				# Read the input parameters from the message
				axis = str(message[2])
				if axis[0] == "[":
					axis_split = axis.split(',')
					new_axis = [0,0,0]
					new_axis[0] = float(axis_split[0][1:])
					new_axis[1] = float(axis_split[1])
					new_axis[2] = float(axis_split[2][:1])
					
					# Calculate the corresponding phase and duration parameters for the carbon rotations
					phase = pi/2*np.sign(new_axis[1]) if new_axis[0] == 0 else np.arctan(new_axis[1]/new_axis[0])
				else:
					if self.rotation_with_pi == 1:
						phase = np.pi*float(axis)
					else: 
						phase = float(axis)
					# phase = float(axis)
				# phase = pi/2*np.sign(float(axis[3])) if float(axis[1]) == 0 else np.arctan(float(axis[3])/float(axis[1]))
				if self.rotation_with_pi == 1:
					angle = np.pi*float(message[3])
				else: 
					angle = float(message[3])
				# angle = float(message[3])
				carbon_nuclei = int(message[1])

				# Get the value of the oscillating magnetic field
				B_osc = self.node.subcomponents["microwave"].parameters["B_osc"]

				# Calculate the corersponding phase and duration parameters for the carbon rotations
				duration = angle/(self.gamma_c*B_osc)

				# The phase remains the same, but duration needs to be changed for the electron rotation, a pi rotation needs to be done for the electron
				duration_e = np.pi/(self.gamma_e*B_osc)

				### Make the excite_mw statmement, which is to be sent to the microwave ###
				# Perform uncontional rotation for halve of the time
				message_send.append(["excite_mw","0", duration/2, self.controller.carbon_frequencies[carbon_nuclei], phase,400e-6])
				message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", duration/2, self.controller.carbon_frequencies[carbon_nuclei], phase+pi,400e-6])
				message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0,400e-6])

				# Perform conditional rotation for halve of the time
				message_send.append(["excite_mw","0", duration/2, self.controller.carbon_frequencies[carbon_nuclei], phase,400e-6])
				message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0,400e-6])
				message_send.append(["excite_mw","0", duration/2, self.controller.carbon_frequencies[carbon_nuclei], phase,400e-6])
				message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0,400e-6])

				# Send the microwave instruction to the microwave
				port_out.tx_output(message_send)
			
			elif message[0] == "qgateuc":
				# Read the input parameters from the message
				axis = str(message[2])
				if axis[0] == "[":
					axis_split = axis.split(',')
					new_axis = [0,0,0]
					new_axis[0] = float(axis_split[0][1:])
					new_axis[1] = float(axis_split[1])
					new_axis[2] = float(axis_split[2][:1])
					
					# Calculate the corresponding phase and duration parameters for the carbon rotations
					phase = pi/2*np.sign(new_axis[1]) if new_axis[0] == 0 else np.arctan(new_axis[1]/new_axis[0])
				else:
					if self.rotation_with_pi == 1:
						phase = np.pi*float(axis)
					else: 
						phase = float(axis)
					# phase = float(axis)
				
				# phase = pi/2*np.sign(float(axis[3])) if float(axis[1]) == 0 else np.arctan(float(axis[3])/float(axis[1]))
				if self.rotation_with_pi == 1:
					angle = np.pi*float(message[3])
				else: 
					angle = float(message[3])
				# angle = float(message[3])
				carbon_nuclei = int(message[1])
				preserved = int(message[4])

				# Get the value of the oscillating magnetic field
				B_osc = self.node.subcomponents["microwave"].parameters["B_osc"]

				
				duration = angle/(self.gamma_c*B_osc)

				# The phase remains the same, but duration needs to be changed for the electron rotation, a pi rotation needs to be done for the electron
				duration_e = np.pi/(self.gamma_e*B_osc)
				
				# Check if the electron state is wanted to be preserved, if so, perform DDrf gate
				if preserved == 1:
					# Make the excite_mw statmement, which is to be sent to the microwave
					message_send.append(["excite_mw","0", duration, self.controller.carbon_frequencies[carbon_nuclei], phase,400e-6])
					message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0,400e-6])
					message_send.append(["excite_mw","0", duration, self.controller.carbon_frequencies[carbon_nuclei], phase,400e-6])
					message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0,400e-6])
				else:
					# Electron is not wanted to be preserved, so initialise electron
					yield from self.real_initialisation_func()
					
					# Rotate electron into |1\rangle state, so free control of carbon is possible
					message_send.append(["excite_mw","0", duration_e, self.controller.elec_freq_reg[0], 0,400e-6])

					# Directly control carbon nucleus
					message_send.append(["excite_mw","0", duration, self.controller.carbon_frequencies[carbon_nuclei], phase,400e-6])

				# Send the microwave instruction to the microwave
				port_out.tx_output(message_send)
			
			# Perform set command
			elif message[0] == "set":
				# Get which component to set a value for
				component = message[1]

				# Get which parameter to set a value for
				parameter = message[2]

				# Get what value to set the parameter to
				value = float(message[3])

				# Check if the component is part of the node
				if component in dict(self.node.subcomponents).keys():
					self.node.subcomponents[component].parameters[parameter] = value
				else:
					raise ValueError(f"The component {component} is not known to this node")
				done = 1

			# Make the local controller wait for a certain amount of ns
			elif message[0] == "wait":
				duration = float(message[1])
				if float(message[1]) != 0:
					yield self.await_timer(duration)
				
				if self.wait_detuning == 1:
					### prelimenary implementation of detuning, potential for additional noise source instead of next implementation ####
					message_send.append(["wait",duration])
					port_out.tx_output(message_send)
				else:
					done = 1
			
			

			# Raise error if the instruction is not known
			else:
				raise ValueError(f"instruction {message[0]} is not supported by micro-architecture")
			
			
			# Check if done signal is on, if not, this means that a signal should come from the diamond component, mentioning it is done
			if done == 0:
				yield self.await_signal(sender = self.subprotocols["diamond_protocol"], signal_label = Signals.SUCCESS)
			message_send = []
			if self.clk_local !=0 and message[0] != "init":		
				time = self.clk_cycles[message[0]]/self.clk_local
				# print(f" printing time now within local {time} for instruction {message[0]}")
				if self.wait_detuning == 1:
					message_send.append(["wait",time*1e9])
					port_out.tx_output(message_send)
					yield self.await_signal(sender = self.subprotocols["diamond_protocol"], signal_label = Signals.SUCCESS)
				yield self.await_timer(time)
			# Send ready signal to the global controller, the global controller will send the next instruction back
			port_in.tx_output("Ready")

	def start(self):
		super().start()
		self.start_subprotocols()

	# Real measurement function implementation
	def real_measurement_func(self):
		# Load zero into the photon counter
		self.controller.supercomponent.subcomponents["photodetector"].photoncountreg = [0]

		# Turn on the measurement light
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("pulses")

		# Wait for a certain amount of time
		yield self.await_timer(60)

		# Turn off the measurement light
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("pulses")

		# Wait a bit longer in order to get all the photons measured
		yield self.await_timer(1) 

		# Get the photon count from the photon count register
		self.controller.Registers[0] = self.controller.supercomponent.subcomponents["photodetector"].photoncountreg[0]

		# Compare the outcome of the photoncount with the threshold value and add a value to the resultregister
		self.controller.ResultRegisters = [1] if self.controller.Registers[0] >self.controller.tresholdregister[0] else [-1]

		# Send the result to the global controller, to be stored in memory
		self.port_out_data.tx_output([["measureresultreg"+self.node.name,self.controller.ResultRegisters[0]],["PhotonCount"+self.node.name,self.controller.Registers[0]]])
		# Send the photoncount to the global controller, to be stored in memory
		# self.port_out_data.tx_output(["PhotonCount"+self.node.name,self.controller.Registers[0]])


	# Initialistion function
	def real_initialisation_func(self):
		# Set the photon register value to 0
		self.controller.supercomponent.subcomponents["photodetector"].photoncountreg = [0]

		# Turn on initialisation light
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("red")

		# Wait for a certain amount of time
		yield self.await_timer(60)

		# Turn of initialisation light
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("red")

		# Wait a little bit for all the photons to come in
		yield self.await_timer(1) 

		# Add the photon count value to the register value
		self.controller.Registers[0] = self.controller.supercomponent.subcomponents["photodetector"].photoncountreg[0]

		# Check if any photons are measured, if yes, perform initialisation protocol again
		if self.controller.Registers[0] >0:

			self.controller.supercomponent.subcomponents["photodetector"].photoncountreg[0] = 0

			yield from self.real_initialisation_func()

	# The charge resonance function
	def real_crc_func(self):
		# Set the photoncount to be 0
		self.controller.supercomponent.subcomponents["photodetector"].photoncountreg = [0]

		# Turn on both initialisation and readout lasers
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input(["pulses","red"])

		# Wait for a certain amount of time
		yield self.await_timer(60)

		# Turn off both initialisation and readout lasers
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input(["pulses","red"])

		# Wait for a certain amount of time to make sure all photons arive at the photondetector
		yield self.await_timer(1)

		# Get the value from the photoncounter 
		self.controller.Registers[0] = self.controller.supercomponent.subcomponents["photodetector"].photoncountreg[0]

		# If the photon count is below the treshold, the nv center is charge pumped
		if self.controller.Registers[0] <= self.controller.tresholdregister[0]:
			# Charge pump laser is turned on
			self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("green")

			# Wait for a certain amount of time
			yield self.await_timer(60)

			# Charge pump laser is turned off
			self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("green")

			# Run the charge resonance check function again to check if we are now in the correct charge state
			yield from self.real_crc_func()

	# Sweep bias function, used in magnetic biasing sequence
	def real_sweep_bias_func(self):
		# Set photon counter to 0
		self.controller.supercomponent.subcomponents["photodetector"].photoncountreg = [0]

		# Turn on green readout laser
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("green")

		# Perform the next rotation 1000 times
		for i in range(50000):
			yield self.await_timer(1)
			self.message_send = [["excite_mw", '0', 10e-9, self.controller.sweepStartReg[0], 0, 400e-6]]
			self.port_out.tx_output(self.message_send)

		# The green laser is turned off
		self.controller.supercomponent.subcomponents["switches"].ports["In_switch"].tx_input("green")
		yield self.await_timer(1)
	
	# Rabi check function implementation
	# def real_Rabi_check_func(self):
	# 	# Perform initialisation
	# 	yield from self.real_initialisation_func()
	# 	# Perform rotation for the given time 
	# 	self.message_send = [["excite_mw", '0', self.controller.startTimeReg[0], 1722000000, 0, 400e-6]]
	# 	self.port_out.tx_output(self.message_send)

	# 	# Wait untill the quantum processor is done performing the instruction
	# 	yield self.await_signal(sender = self.subprotocols["diamond_protocol"], signal_label = Signals.SUCCESS)

	# 	# Measure the electron
	# 	yield from self.real_measurement_func()

	# 	# Store results
	# 	# self.controller.memAddr_result[self.i]+=self.controller.ResultRegisters[0]
	# 	self.controller.memAddr_result[self.i].append(self.controller.ResultRegisters[0])
	# 	self.controller.memAddr_photon_count[self.i] += (self.controller.Registers[0])
	# 	self.controller.memAddr_time[self.i] = (self.controller.startTimeReg[0])		

	# 	# Increment the measure counter
	# 	self.controller.MeasureCounterReg[0] += 1

	# 	# Check if the correct amount of measurements are already done. 
	# 	# A multitude of measurements need to be done to get probability values
	# 	if self.controller.MeasureCounterReg[0] < self.controller.MeasureAmountReg[0]:
	# 		yield from self.real_Rabi_check_func()
		
	# 	# Reset measurement counter
	# 	self.controller.MeasureCounterReg[0] = 0
	# 	self.i+=1

	# 	# Check if all the wanted times are already tested, if not, change the time and perform algorithm again
	# 	if self.controller.startTimeReg[0] < self.controller.stopTimeReg[0]:
	# 		self.controller.startTimeReg[0] += self.controller.stepTimeReg[0]
	# 		yield from self.real_Rabi_check_func()

# Laser protocol
class Laser_protocol(Protocol):
	def __init__(self,switches,  diamond, absorption_probability, p_charge_state_pump_succes = 1, photon_emission_noise = False):
		super().__init__()
		self.switches = switches.switches
		self.diamond = diamond
		self.p_charge_state = p_charge_state_pump_succes
		self.p = absorption_probability
		self.photon_emission_noise = photon_emission_noise
		self.lasers = self.diamond.supercomponent.supercomponent.subcomponents["red_laser"]
		power = self.lasers.laser_properties["power"]

	def run(self):
		# Define input port for the protocol
		port_in = self.diamond.ports["In_classic_switch"]
		photon_noise_model = NVPhotonEmissionNoiseModel(delta_w =376.5, tau_decay = 52 )

		while True:
			# Wait on an input value
			yield self.await_port_input(port_in)
			laser_name = port_in.rx_input().items[0]
			power = float(self.diamond.supercomponent.supercomponent.subcomponents[laser_name+"_laser"].laser_properties["power"])
			absorption_rate = 5.8526867e-17*power/1.88*1e9
			# print(f"the absorption rate is {absorption_rate}")
			# Get the electron from the diamond, so it can be measured
			self.electron = self.diamond.peek(0)[0]	

			# The protocol for the measurement laser starts		
			if self.switches["pulses"] == 1:

				# If the initialisation laser is not turned on, this means that the measurement protocol is on right now
				if self.switches["red"] == 0:
					# Check if we are in the correct charge state
					if self.diamond.NV_state == "NV-":
						
						# Create a photon which can be send to the photondetector
						photon, = create_qubits(1)
						
						assign_qstate(photon, np.diag([1,0]))

						# Get the output which is sending the photon to the photondetector
						port_out = self.diamond.ports["qout"]

						while self.switches["pulses"] == 1:
							# If a photon is absorped, the electron is measured
							photon_absorped = np.random.binomial(1,self.p)
							if photon_absorped:

								# The electron is measured, a photon will be emitted if the electron is measured to be in the 0 state
								photon_emitted = measure(self.electron)
								if photon_emitted[0] == 0:
									# Send a photon to the photondetector
									port_out.tx_output(photon)

									# Apply photon emission noise to the photon and carbon nuclei if photon emission noise is turned on
									if self.photon_emission_noise == True:
										photon_noise_model.apply_noise(spin = self.electron, photon = photon, memQubits = [self.diamond.peek(1)[0]], alpha = 0.5)
							yield self.await_timer(absorption_rate)

				# If the initialisation laser is on at the same time as the readout laser,
				# This means that a photon will be emitted if the electron is measured in the 0 or 1 state
				elif self.switches["red"] == 1:
					# Check the charge state again
					if self.diamond.NV_state == "NV-":

						# A photon is created to be send to the photondetector
						photon, = create_qubits(1)
						assign_qstate(photon, np.diag([1,0]))
						port_out = self.diamond.ports["qout"]

						# As long as the lasers are on, every nanosecond a photon will be send to the photondetector
						while self.switches["red"] == 1:
							yield self.await_timer(absorption_rate)
							photon_absorped = np.random.binomial(1,self.p)
							if photon_absorped:
								photon_emitted = np.random.binomial(1,1)
								if photon_emitted:
									port_out.tx_output(photon)
									random_bin = np.random.binomial(1,0.5)
									assign_qstate(self.electron, np.diag([random_bin,1-random_bin]))

									# Apply photon emission noise to the photon and carbon nuclei if photon emission noise is turned on
									if self.photon_emission_noise == True:
										photon_noise_model.apply_noise(spin = self.electron, photon = photon, memQubits = [self.diamond.peek(1)[0]], alpha = 0.5)

			# If only the green laser is turned on, the electron will be excited if it is in the 0 or in the 1 state
			elif self.switches["green"]:
				photon, = create_qubits(1)
				
				assign_qstate(photon, np.diag([1,0]))
				while self.switches["green"] == 1:
					
					# Get the succes of the charge state initialisation
					charge_state_pump = np.random.binomial(1,self.p_charge_state)
					
					# Set charge state if the charge state pump was succesfull
					if charge_state_pump:
						self.diamond.NV_state == "NV-"
					
					# Check if we are in the correct charge state
					if self.diamond.NV_state == "NV-":
						photon_absorped = np.random.binomial(1,self.p) 
						if photon_absorped:
							# The electron is measured and its values are stored
							photon_emitted = measure(self.electron)

							# If the electron is measured in the 0 state, a photon is emitted
							if photon_emitted[0] == 0:
								port_out.tx_output(photon)

								# Apply photon emission noise to the photon and carbon nuclei if photon emission noise is turned on
								if self.photon_emission_noise == True:
									photon_noise_model.apply_noise(spin = self.electron, photon = photon, memQubits = [self.diamond.peek(1)[0]], alpha = 0.5)
							
							# If the electron is measured in the 1 state, a photon is emitted with 70%, in this case the electron is put in the 1 state
							# If the electron is measured in the 1 state and no photon it emitted, the electron is put into the 0 state.
							else:
								decay_into_s = np.random.binomial(1,0.3)
								if decay_into_s:
									assign_qstate(self.electron, np.diag([1,0]))
								else:
									port_out.tx_output(photon)
									assign_qstate(self.electron, np.diag([0,1]))
									# Apply photon emission noise to the photon and carbon nuclei if photon emission noise is turned on
									if self.photon_emission_noise == True:
										photon_noise_model.apply_noise(spin = self.electron, photon = photon, memQubits = [self.diamond.peek(1)[0]], alpha = 0.5)
									
					yield self.await_timer(1)

			# Initialisation laser is turned on
			elif self.switches["red"]:
				# Check if you are in the ocrrect charge state
				if self.diamond.NV_state == "NV-":
					# Create a photon and get the connection to the photondetector
					photon, = create_qubits(1)
					assign_qstate(photon, np.diag([1,0]))
					port_out = self.diamond.ports["qout"]
					counter = 0

					while self.switches["red"] == 1:
						counter += 1
						photon_absorped = np.random.binomial(1,self.p)
						if photon_absorped:
							photon_emitted = measure(self.electron)
							photon_emitted = photon_emitted[0]
       

							# If the electron is measured ot be in the 1 state, a photon is emitted and afterwards the electron is put in the 0 state
							if photon_emitted ==1:
								port_out.tx_output(photon)
								assign_qstate(self.electron, np.diag([1,0]))

								# Apply photon emission noise to the photon and carbon nuclei if photon emission noise is turned on
								if self.photon_emission_noise == True:
									photon_noise_model.apply_noise(spin = self.electron, photon = photon, memQubits = [self.diamond.peek(1)[0]], alpha = 0.5)
						yield self.await_timer(1)

			# Entanglement laser function needs to be added
			elif self.switches["random"]:
				if self.diamond.NV_state == "NV-":
					pass

# The switch protocol
class SwitchProtocol(Protocol):
	def __init__(self, switch):
		super().__init__()
		self.switch = switch

	def run(self):
		# Define in and output ports
		port_in = self.switch.ports["In_switch"]
		port_out = self.switch.ports["Out_switch"]
		while True:
			# Wait for input at the input port
			yield self.await_port_input(port_in)

			# Read the message from the input port
			message = port_in.rx_input().items

			# For every name in the message, flip that switch
			for i in range(len(message)):
				if self.switch.switches[message[i]] == 0:
					self.switch.switches[message[i]] =1
					
					# If switch is turned on, sent a signal to the diamond with the laser name
					port_out.tx_output(message[i])
				else:
					self.switch.switches[message[i]] = 0

#next protocol might be used for all consistent parameters.
class DecayProtocol(Protocol): #this should be renamed to ionization protocol in the future
	def __init__(self, diamond, decay_prob = 0.9999, darkcount_prob = 0.00001, timer_value = 1000000):
		super().__init__()
		self.diamond = diamond
		self.p = decay_prob
		self.darkcount_prob = darkcount_prob
		self.timer_value = timer_value
	
	def run(self):
		while True:
			yield self.await_timer(self.timer_value)
			NV_decay = np.random.binomial(1,self.p)
			dark_count = np.random.binomial(1,self.darkcount_prob)
			
			if NV_decay:
				self.diamond.NV_state = 'NV0'
			
			if dark_count:
				self.diamond.supercomponent.subcomponents["photodetector"].photoncountreg[0] += 1 

			
