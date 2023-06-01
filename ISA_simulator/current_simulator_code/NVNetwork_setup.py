import netsquid as ns
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components.instructions import INSTR_CROT, INSTR_ROT, INSTR_ROT_Z
from netsquid_nv.nv_center import NVQuantumProcessor
from netsquid.components.instructions import IGate
from Components import *
from netsquid.nodes.network import Network

# Setup the network
def network_setup(node_distance = 4e-3,qubit_number = 2, photo_distance = 2e-3, node_number = 2, noiseless = True, CarbonAparr = None,photon_detection_probability=1, absorption_probability = 1, no_z_precission = 1, decay = False, photon_emission_noise = False, detuning = 0, electron_T2 = None, carbon_T2 = None, electron_T1 = None, carbon_T1 = None,single_instruction = True,B_osc = 400e-6, B_z = 40e-3,frame = "rotating", wait_detuning = 1,clk_local = 0):
	NV_node_list = [] # Create an empty list, so the nodes can be appended to this list, creating a list of nodenames
	
	# If values for the parallel hyperfine parameter for the carbon nuclei is given, that values are used, otherwise predifined values are used
	if CarbonAparr == None:
		pre_def_A_parr = [213e3, 20e3,-48e3,110e3,300e3,400e3,500e3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #213e3 7e3# Predefined values for the parallel hyperfine parameter for the carbon nuclei
	else: # [213e3, 20e3,-48e3,36e3,13e3,28e3,100e3
		pre_def_A_parr = CarbonAparr # If values are given, these values are used
	
	# In this for loop the wanted amount of nodes are going to be made, a quantum processor will made and added for every node.
	for i in range(int(node_number)): 
		NV_center_1 = NVQuantumProcessor(num_positions=int(qubit_number), noiseless=noiseless, electron_T2=electron_T2,carbon_T2=carbon_T2, electron_T1 = electron_T1, carbon_T1 = carbon_T1)  # make 3 quantum NV processors# the name of the nv center is 'nv_center_quantum_processor'
		NV_center_1.NV_state = 'NV-'
		for k in range(len(NV_center_1.mem_positions)-2): 
			NV_center_1.mem_positions[k+1].add_property(name = "A_parr", value = pre_def_A_parr[k]) #add the parallel values to the memory positions of the carbon nuclei of every node.
		phys_instruction_adder(NV_center_1) #add the needes physical instructions to the diamond quantum processor
		
		# Port names are added to the nv center
		# The connected components are the local controller, the switches (lasers), and the 2 beamsplitters needed for entanglement
		# An output port for the photon emission is already intrinsicly present in the NVquantumProcessor unit
		# The local controller is needed to be connected, because the quantum processor also entails the microwave generator
		NV_center_1.add_ports(['In_classic_cont', 'In_classic_switch', 'Out_beamsplitter_R', 'Out_beamsplitter_L']) 
		
		# NVnodes are created, the quantum processor of the node is the NV center which has just been made and port names are added to the NV node.
		# The connected components to the node are the beamsplitters as well, the connection to the global controller will be added later on
		NV_node = NVNode(name = "nvnode"+str(i), NV_center = NV_center_1, port_names = ["Out_beamsplitter_L","Out_beamsplitter_R"])
		
		# Every node is added to the list of nodes.
		NV_node_list.append(NV_node)


	# The global controller is made
	controller = Controller(name = 'controller')

	# Make the network in which all the components are going to be added.
	network = Network("contr_NV_centers")

	# Add all the nodes of the list to the network
	network.add_nodes(NV_node_list+[controller])
	network.add_subcomponent(ExternalMagnet(name = "external_magnet",magnetic_field=B_z))

	# Add lasers to the network
	laser_adder(network)

	### need for next for loop 	###
	# Add connections between the global controller and every node
	# Add all the needed components to every NVNode
	# Add all the needed connections for the components of every NVnode.
	###							###
	for i in range(int(node_number)):
		connection = ClassicalConnection(name = "C2"+str(i),length = node_distance)
		connection_2 = ClassicalConnection(name = str(i)+"2C", length = node_distance, right = False)
		network.add_connection(controller, NV_node_list[i], connection = connection_2, label = "classical_"+str(i)+"2C", port_name_node1 = "In_nvnode"+str(i)+"data", port_name_node2 = "Out_nvnode"+str(i)+"data")

		network.add_connection(controller, NV_node_list[i], connection = connection, label = "classical_C2"+str(i), port_name_node1 = "Out_nvnode"+str(i), port_name_node2 = "In_nvnode"+str(i))
		print("before subcomponent adder")
		subcomponent_adder(NV_node_list[i], photon_detection_probability, absorption_probability, no_z_precission, decay, photon_emission_noise, detuning,single_instruction,B_osc,frame, wait_detuning,clk_local)
		print("after subcomponent adder")
		if i <int(node_number)-1:
			entanglement_components_adder(network, i)
			entanglement_component_connection_adder(network, i)
		connection_adder(Node = NV_node_list[i], node_distance = node_distance)
		quantum_connection_adder(NV_node_list[i], photo_distance = photo_distance)
	return network

# Add all the components of the NVNodes to the NVnodes
def subcomponent_adder(Node,photon_detection_probability, absorption_probability, no_z_precission, decay, photon_emission_noise, detuning,single_instruction,B_osc,frame, wait_detuning,clk_local):
	photodetector = Photodetector(name = "photodetector", port_names = ['In_p', 'Out_p'])
	switches = Switches(name = "switches", port_names = ['In_switch', 'Out_switch'])
	Node.add_subcomponent(photodetector)
	Node.add_subcomponent(switches)
	microwave_generator = Microwave_generator(name="microwave", port_names = ["In_microwave"],B_osc = B_osc)
	Node.add_subcomponent(microwave_generator)
	local_controller = Local_controller(name = "local_controller",node = Node, port_names = ['In_cont','Out_cont_data', 'Out_cont', 'Out_cont_micro'], photon_detection_probability=photon_detection_probability, absorption_probability = absorption_probability, no_z_precission = no_z_precission, decay = decay, photon_emission_noise = photon_emission_noise, detuning = detuning,single_instruction = single_instruction, frame = frame,wait_detuning=wait_detuning,clk_local = clk_local)	
	Node.add_subcomponent(local_controller)
	


# Add lasers to the network
def laser_adder(Network):
	red_light_laser = Lasers(name = "red_laser")
	red_light_laser.laser_properties["wavelength"] = 570e-9
	red_light_laser_rand = Lasers(name = "red_laser_rand")
	green_light_laser = Lasers(name = "green_laser")
	green_light_laser.laser_properties["wavelength"] = 515e-9
	pulse_laser = Lasers(name = "pulses_laser")
	pulse_laser.laser_properties["wavelength"] = 619e-9

	Network.add_subcomponent(red_light_laser)
	Network.add_subcomponent(red_light_laser_rand)
	Network.add_subcomponent(green_light_laser)
	Network.add_subcomponent(pulse_laser)


# Add entanglement components to the network
def entanglement_components_adder(Network, iterator):
	name_of_node = "nvnode"+str(iterator)
	name_of_next_node = "nvnode"+str(iterator+1)
	photodetector_left = Photodetector(name = "photodetector_left_of_beamsplitter_of_"+name_of_node, port_names = ['In_beamsplitter_of'+name_of_node, 'Out_p'])
	photodetector_right = Photodetector(name = "photodetector_right_of_beamsplitter_of_"+name_of_node, port_names = ['In_beamsplitter_of'+name_of_node, 'Out_p'])
	beamsplitter = Beamsplitter(name = "beamsplitter_of"+name_of_node, port_names = ['In_NV'+name_of_node, 'In_NV'+name_of_next_node, 'Out_photo_left_of'+ name_of_node, 'Out_photo_right_of'+ name_of_node])
	Network.add_subcomponent(photodetector_left)
	Network.add_subcomponent(photodetector_right)
	Network.add_subcomponent(beamsplitter)


# Add connection between the components of the NVNode
def connection_adder(Node, node_distance = 4e-3):
	classic_con_contr_to_proc = ClassicalConnection(name = "C_in2proc", length = node_distance/6, left= False)
	classic_con_switch_to_dia = ClassicalConnection(name = "switch_to_dia", length = 0, left = False)
	classic_con_cont_to_microwave = ClassicalConnection(name = "cont_to_microwave", length = node_distance/6, left = False)
	Node.add_subcomponent(classic_con_contr_to_proc)
	Node.add_subcomponent(classic_con_switch_to_dia)
	Node.add_subcomponent(classic_con_cont_to_microwave)
	Node.subcomponents["local_controller"].ports['Out_cont'].connect(classic_con_contr_to_proc.ports['A'])
	Node.subcomponents["nv_center_quantum_processor"].ports['In_classic_cont'].connect(classic_con_contr_to_proc.ports['B'])
	Node.subcomponents["switches"].ports['Out_switch'].connect(classic_con_switch_to_dia.ports['A'])
	Node.subcomponents["nv_center_quantum_processor"].ports['In_classic_switch'].connect(classic_con_switch_to_dia.ports['B'])
	Node.ports["In_"+Node.name].forward_input(Node.subcomponents["local_controller"].ports['In_cont'])
	Node.subcomponents["local_controller"].ports["In_cont"].forward_output(Node.ports["In_"+Node.name])
	Node.ports["Out_"+Node.name+"data"].forward_input(Node.subcomponents["local_controller"].ports['Out_cont_data'])
	Node.subcomponents["local_controller"].ports["Out_cont_data"].forward_output(Node.ports["Out_"+Node.name+"data"])
	Node.subcomponents["local_controller"].ports['Out_cont_micro'].connect(classic_con_cont_to_microwave.ports['A'])
	Node.subcomponents["microwave"].ports['In_microwave'].connect(classic_con_cont_to_microwave.ports['B'])


# Add quantum ocnnection to the component of the NVnoe
def quantum_connection_adder(Node, photo_distance = 2e-3):
	Quantum_con_dia_to_photodetec = QuantumConnection(name = "N2P", length = 0)
	Node.add_subcomponent(Quantum_con_dia_to_photodetec)
	Node.subcomponents["nv_center_quantum_processor"].ports['qout'].connect(Quantum_con_dia_to_photodetec.ports['A'])
	Node.subcomponents["photodetector"].ports['In_p'].connect(Quantum_con_dia_to_photodetec.ports['B'])

# Add connections between the entanglement components
def entanglement_component_connection_adder(Network, iterator, photo_distance = 0):
	name_of_node = "nvnode"+str(iterator)
	name_of_next_node = "nvnode"+str(iterator+1)	
	Quantum_con_beamsplitter_to_photodetec_left = QuantumConnection(name = "B2PL", length = photo_distance)
	Network.add_subcomponent(Quantum_con_beamsplitter_to_photodetec_left)
	Network.subcomponents["beamsplitter_of"+name_of_node].ports['Out_photo_left_of'+ name_of_node].connect(Quantum_con_beamsplitter_to_photodetec_left.ports['A'])
	Network.subcomponents["photodetector_left_of_beamsplitter_of_"+name_of_node].ports['In_beamsplitter_of'+name_of_node].connect(Quantum_con_beamsplitter_to_photodetec_left.ports['B'])

	Quantum_con_beamsplitter_to_photodetec_right = QuantumConnection(name = "B2PR", length = photo_distance)
	Network.add_subcomponent(Quantum_con_beamsplitter_to_photodetec_right)
	Network.subcomponents["beamsplitter_of"+name_of_node].ports['Out_photo_right_of'+ name_of_node].connect(Quantum_con_beamsplitter_to_photodetec_right.ports['A'])
	Network.subcomponents["photodetector_right_of_beamsplitter_of_"+name_of_node].ports['In_beamsplitter_of'+name_of_node].connect(Quantum_con_beamsplitter_to_photodetec_right.ports['B'])

	Quantum_con_dia_R_to_beamsplitter = QuantumConnection(name = "D2BR", length = photo_distance)
	Network.add_subcomponent(Quantum_con_dia_R_to_beamsplitter)
	Network.subcomponents[name_of_node].ports['Out_beamsplitter_R'].connect(Quantum_con_dia_R_to_beamsplitter.ports['A'])
	Network.subcomponents["beamsplitter_of"+name_of_node].ports['In_NV'+name_of_node].connect(Quantum_con_dia_R_to_beamsplitter.ports['B'])
	Network.subcomponents[name_of_node].subcomponents["nv_center_quantum_processor"].ports["Out_beamsplitter_R"].forward_output(Network.subcomponents[name_of_node].ports['Out_beamsplitter_R'])

	Quantum_con_dia_L_to_beamsplitter = QuantumConnection(name = "D2BL", length = 0)
	Network.add_subcomponent(Quantum_con_dia_L_to_beamsplitter)
	Network.subcomponents[name_of_next_node].ports['Out_beamsplitter_L'].connect(Quantum_con_dia_L_to_beamsplitter.ports['A'])
	Network.subcomponents["beamsplitter_of"+name_of_node].ports['In_NV'+name_of_next_node].connect(Quantum_con_dia_L_to_beamsplitter.ports['B'])
	Network.subcomponents[name_of_next_node].subcomponents["nv_center_quantum_processor"].ports["Out_beamsplitter_L"].forward_output(Network.subcomponents[name_of_next_node].ports['Out_beamsplitter_L'])


def phys_instruction_adder(Node):
	phys_instructions = [] # Add an empty list, so the needed physical instructions can be appended to this list.
	electron_carbon_topologies = \
            [(Node.electron_position, carbon_pos) for carbon_pos in Node.carbon_positions]
	
	phys_instructions.append(
                PhysicalInstruction(INSTR_ROT,
                                    parallel=False,
                                    topology=[Node.electron_position],
                                    duration = 0))
	phys_instructions.append(
            PhysicalInstruction(INSTR_ROT_Z,
                                parallel=False,
                                topology=[Node.electron_position],
                                duration=0))
	phys_instructions.append(
            PhysicalInstruction(INSTR_ROT_Z,
                                parallel=False,
                                topology=Node.carbon_positions,
                                duration=0))
 
	phys_instructions.append(             
            PhysicalInstruction(INSTR_CROT,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                duration=0)) 

	INSTR_CZXY = IGate(name = "CZXY_gate",num_positions=2)
    
	phys_instructions.append(
            PhysicalInstruction(INSTR_CZXY,
                                parallel=False,
                                topology=[(0,1)],
                                duration = 0))
	for instruction in phys_instructions:
            Node.add_physical_instruction(instruction)