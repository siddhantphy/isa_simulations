import netsquid as ns
from netsquid.components import ClassicalChannel, Component
from netsquid.components.models import FibreDelayModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.connections import Connection
from netsquid.nodes.node import Node
from NVprotocols import *


# All the components which are being made are shown below. They inherit from a node or component class and are given initial values.

class NVNode(Node): #make the NVnode class, to which all the components are going to be added in the NVnetwork file
	def __init__(self, name, NV_center, port_names=None):
		super().__init__(name=name, qmemory=NV_center, port_names=port_names)

class Controller(Node): #the global controller is a node on its own
	def __init__(self, name, port_names = None):
		super().__init__(name = name, port_names = port_names)
		self.Registers = [0]
		self.PhotonRegisters = []		
		self.memAddr = [0]
		self.fractionStore = [0]
		self.LABEL_dict = {}
		self.register_dict = {}
		# self.memory = {"measurevaluepermeasure":[0]}
		self.memory = {}
  
		self.clk_cycle_dict = {
			"label": 1,
			"qgateuc": 1,
			"qgatecc": 1,
			"qgatee": 1,
			"qgatez": 1,
			"set": 1,
			"br": 1,
			"jump": 1,
			"addi": 1,
			"subi": 1,
			"wait": 1,
			"st": 1,
			"add": 1,
			"sub": 1,
			"ldi": 1,
			"mov": 1,
			"mul": 1,
			"sub": 1,
			"rest": 1,
		}
		
class Photodetector(Component): 
	def __init__(self, name, port_names=None):
		super().__init__(name=name, port_names=port_names)
		self.photoncountreg = [0]
		self.port_names = port_names
		
class Switches(Component):
	def __init__(self,name, port_names = None):
		super().__init__(name=name, port_names = port_names)
		self.switches = {
			"red": 0,
			"random": 0,
			"green": 0,
			"pulses": 0
		}

class Beamsplitter(Component):
	def __init__(self,name, port_names = None):
		super().__init__(name=name, port_names = port_names)
		self.ports_in = [port_names[0], port_names[1]]
		self.ports_out = [port_names[2],port_names[3]]

class Microwave_generator(Component):
	def __init__(self,name, port_names = None, B_osc = 400e-6):
		super().__init__(name=name, port_names = port_names)
		self.parameters = {"B_osc":B_osc, "envelope":0}

		

class Local_controller(Component):
	def __init__(self, name, node, no_z_precission, decay,  port_names=None, photon_detection_probability = 1, absorption_probability = 1, photon_emission_noise = False, detuning = 0,single_instruction = True, frame = "rotating", wait_detuning = 1,clk_local = 0, Dipole_moment = None,rotation_with_pi=1):
		super().__init__(name=name, port_names=port_names)
		self.Registers = [0] *33
		self.PhotonRegisters = [0]
		self.ResultRegisters = [0] * 34
		self.dacReg = [0]
		self.tresholdregister = [0]
		self.sweepStartReg = [0]
		self.sweepStepReg = [0]
		self.sweepStopReg = [0]
		self.memAddr = []
		self.carbon_frequencies = []
		self.elec_freq_reg = [0]
		self.elec_rabi_reg = [0]
		self.carbon_rabi_frequencies = []
		self.clk_local = clk_local
		self.clk_cycle_dict = {
			"init": 0,
			"excite_mw": 1,
			"mw": 1,
			"rf": 1,
			"entangle": 1,
			"initialize": 1,
			"init_real": 1,
			"crc": 1,
			"magnetic_bias": 1,
			"magbias": 1,
			"detectcarbon": 1,
			"swapce": 1,
			"measure": 1,
			"rabicheck": 1,
			"swapec": 1,
			"qgatee": 1,
			"qgatedir": 1,
			"qgatez": 1,
			"qgatecc": 1,
			"qgateuc": 1,
			"set": 1,
			"wait": 1,
		}


		## the following lists have only been added for simulation plotting perposes, they are not actual part of the microprocessor
		self.memAddr_photon_count = [] 
		self.memAddr_freq = []
		self.memAddr_time = []

		#start the protocol which is connected to the local controller, thus giving it a functionality. The other protocols for the other components are added as subprotocols of the local controller
		self.loc_cont_prot = Loc_Cont_Protocol(controller = self, node = node, photon_detection_probability=photon_detection_probability, absorption_probability = absorption_probability, no_z_precission = no_z_precission, decay = decay, photon_emission_noise = photon_emission_noise, detuning = detuning,single_instruction = single_instruction, frame = frame, wait_detuning=wait_detuning,Dipole_moment=Dipole_moment,rotation_with_pi=rotation_with_pi)
		self.loc_cont_prot.start()

class Lasers(Component):
	def __init__(self,name,port_names = None):
		super().__init__(name =name)
		self.laser_properties = {
			"wavelength": 0,
			"power": 5e7
		}

class VoltageSource(Component):
	def __init__(self, name, voltage):
		super().__init__(name = name, port_names = ["In_volt, Out_volt"])
		self.voltage = voltage

class ExternalMagnet(Component):
	def __init__(self, name, magnetic_field = 0.1089):
		super().__init__(name = name, port_names = ["In_magnet"])
		self.parameters = {"magnetic_field":magnetic_field}
	
class CurrentSource(Component):
	def __init__(self, name, current):
		super().__init__(name = name, port_names = ["In_volt, Out_volt"])
		self.current = current

class ClassicalConnection(Connection): #make a classical connection between the global controller and the NV centers
	def __init__(self,name, length, right = True, left = True):
	
		super().__init__(name = name)
		
		# Add channels, the two options are given, because 1 of the classical connections only needs to be 1 way. Thus saying right = False, will result in a one way connection from local to global controller.
		if left:
			self.add_subcomponent(ClassicalChannel("channel_ABD2C", length = length, models = {"delay_model": FibreDelayModel()}))
			self.ports['B'].forward_input(self.subcomponents["channel_ABD2C"].ports['send'])
			self.subcomponents["channel_ABD2C"].ports['recv'].forward_output(self.ports['A'])
		
		if right:
			self.add_subcomponent(ClassicalChannel("channel_C2ABD",length = length, models = {"delay_model": FibreDelayModel()}))
			self.ports['A'].forward_input(self.subcomponents["channel_C2ABD"].ports['send'])
			self.subcomponents["channel_C2ABD"].ports['recv'].forward_output(self.ports['B'])


class QuantumConnection(Connection):
	def __init__(self,name,length):
		
		super().__init__(name = name)
		
		self.add_subcomponent(QuantumChannel("qchannel_N_to_photo", length = length, models = {"delay_model": FibreDelayModel()}))
		self.ports['A'].forward_input(self.subcomponents["qchannel_N_to_photo"].ports['send'])
		self.subcomponents["qchannel_N_to_photo"].ports['recv'].forward_output(self.ports['B'])

