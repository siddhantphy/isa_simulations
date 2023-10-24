from netsquid.components import QuantumProgram, INSTR_ROT, INSTR_CROT, INSTR_ROT_Z
from math import atan, sin, cos
import numpy as np
from operator import add
import math
from netsquid.qubits.operators import Operator
from netsquid.components.instructions import IGate

class Programs_microwave(QuantumProgram):
	def __init__(self, prog_name, diamond, no_z = 1, detuning = 0,single_instruction = True, frame = "rotating", jitter = 0,N_state = None, Dipole_moment = None,N_noise_effect = None):
		super().__init__()
		self.prog_name = prog_name # In this program name, the instruction which needs to be performed is stored
		self.diamond = diamond # The diamond is needed, because properties of the qubits are needed to know how far the rotations need to be done 
		self.network = network = diamond.supercomponent.supercomponent
		# print('the network is printed next')
		# print(self.network)
		self.ext_magnet = network.subcomponents["external_magnet"] # The external magnetic field is needed in order to calculate the resonance frequency
		self.no_z = no_z # This is a parameter which determines if a z spin precission is wanted (this can be used to turn off unwanted rotations for testing purposes)
		self.microwave = diamond.supercomponent.subcomponents["microwave"] # The microwave component is needed, because it contains information regarding the microwave shape and oscillating magnitude
		self.detuning = detuning
		self.single_instruction = single_instruction
		self.frame = frame
		self.jitter = jitter
		self.N_state = N_state
		self.Dipole_moment = Dipole_moment
		self.mem_positions = len(diamond.mem_positions)
		self.N_noise_effect = N_noise_effect

	def program(self):
		### First all the parameters are determined ###
		# Zero field splitting of the electron ms = -1,1 states with regards to the ms = 0 state. 
		# This is the resonance frequency when there is no magnetic field present
		D = 2.87e9  #Hz
		decoherence_value = self.network.noise_parameters["T2_carbon"]
		T2detuning = self.network.noise_parameters["T2Detuning"]
		carbon_detuning_decoherence_based = 0 if (T2detuning == True or decoherence_value == 0) else 1/(2*np.pi*decoherence_value) #2*np.pi*
		carbon_jitter_detuning = np.random.rand(len(self.diamond.supercomponent.subcomponents["local_controller"].carbon_frequencies))*carbon_detuning_decoherence_based-carbon_detuning_decoherence_based/2 #use this value in order to perform the waiting operation with detuning
		N_state_p0 = abs(self.N_state[0][0])
		# print(decoherence_value)
		N_state_p1 = abs(self.N_state[1][1])
		N_state_effect = np.random.binomial(1,N_state_p1)
  
		jitter = np.random.rand()*self.jitter-self.jitter/2
		#read the value for the z-direction of the external magnetic field
		B_z = self.ext_magnet.parameters["magnetic_field"] 

		#read the oscillating magnetic field strenght value
		B_osc = self.microwave.parameters["B_osc"]

		#gyromagnetic ratio for the electron
		gamma_e = 28.7e9 #Hz/T

		# Calculate the resonance frequency for the ms = -1 electron state
		# omega_0 = D-gamma_e*B_z 
		A_parr_N = self.diamond.mem_positions[self.mem_positions-2].properties["A_parr"]
		if self.N_noise_effect != None:
			omega_0 = abs(D-gamma_e*B_z -N_state_effect*A_parr_N)
		else:
			omega_0 = abs(D-gamma_e*B_z)
		# print(f"check values for old omega {omega_0} and new omega {omega_2}, with hyperfine {A_parr_N}")




		# Calculate the resonance frequency for the ms = 1 electron state
		omega_0_plus1 = abs(D+gamma_e*B_z) 

		# Calculate the optimal evolution over time 
		Omega_e = B_osc*gamma_e

		### Next the values for the carbon nuclei are given ###
		gamma_c = 1.07e7 #gyromagnetic ratio carbon (Hz/T)
		omega_L_c = gamma_c*B_z # Larmor frequency carbon
		omega_L_c = math.copysign(omega_L_c, D-gamma_e*B_z)
		# print(f"check omega first {D-gamma_e*B_z}")
		Omega_c = B_osc*gamma_c #optimal evolution over time

		
		if self.prog_name[0] == "excite_mw": # This is the only instruction important to the microwave generator
			# The frequency of the microwave/radiofrequency wave
			# Maybe change to + detuning instead of - detuning. Then accept detuning to be positive and negative
			omega_rf = abs(float(self.prog_name[3])-self.detuning)
			# print(self.prog_name[2])
			
		
			if self.frame == "rotating":
				if self.microwave.parameters["envelope"] == 0: # The envelope value is checked, this is currently the only supported envelope 
					
					# The difference between the resonance frequency and the microwave frequency is calculated
					# There are 2 values, because this is a triplet state, effective 2-level system is achieved by a large field splitting
					delta_omega = abs(omega_0-omega_rf)
					delta_omega_plus1 = abs(omega_0_plus1-omega_rf)
					
					# Determine detuning angle
					alpha = np.pi/2 if self.detuning == 0 else atan(Omega_e/self.detuning)

					# Determine the damping based on the off resonance frequency
					damping = 1 if delta_omega == 0 else sin(atan(Omega_e/delta_omega)) #formula will be changed in the future
					damping_plus1 = 1 if delta_omega_plus1 == 0 else sin(atan(Omega_e/delta_omega_plus1))
				
					# The phase of the microwave is read, this is needed in order to dermine the axis of rotation
					phase = float(self.prog_name[4])
					# The axis of rotation are determined based on the phase
					axis = [cos(phase),sin(phase),0]					
					
					# The time duration of the microwave is read
					duration = float(self.prog_name[2]) +jitter

					# Calculated rotation on basis of the magnetic field strength, external magnetic field and duration
					# The rotation angle is determined by use of the evolution value multiplied by the time, however there is also a damping factor 
					# This damping factor comes due to the off resonance frequency.
					# When the frequency of the microwave is on resonance, the damping factor is 1, resulting in no damping.
					# Max damping is used, because only 1 electron matrix is given and the rotation needs to be performed if the microwave frequency is close to resonant to 1 of the Larmor frequencies of the electrons.

					damping = 0 if damping < 0.01 else damping
					damping_plus1 = 0 if damping_plus1 < 0.01 else damping_plus1
					# Omega_effective = Omega_e*damping
					rotation_angle_e = Omega_e*duration*max(damping,damping_plus1) #prelimenary implementation of /2*pi. 
					Omega_detuning = np.sqrt((Omega_e*damping)**2+self.detuning**2) # please remember that the detuning here is due to the detuning with the LO with its own reference qubit.
					Omega_detuning_without_damping = np.sqrt((Omega_e)**2+self.detuning**2) #important for rotations on the qubits in their own reference frame. Use minimal frequency indicator to check which reference frame we are in
					#next rotation angle is experimental for detuning coding
					# print(f"detuning omega {Omega_detuning} Omega_e {Omega_e} detuning {self.detuning}")
					# print(f"alpha is {alpha}")
					elec_freq = self.diamond.supercomponent.subcomponents["local_controller"].elec_freq_reg
					carbon_frequencies = self.diamond.supercomponent.subcomponents["local_controller"].carbon_frequencies
					# print(f"the length of the two lists are {len(carbon_jitter_detuning)} and {len(carbon_frequencies)}")
					carbon_frequencies = list(map(add,carbon_frequencies,carbon_jitter_detuning))
					total_freq = elec_freq+carbon_frequencies
					# print(f"the carbon frequencies are {total_freq}")
					total_substracted = [np.abs(np.abs(value) - omega_rf) for value in total_freq]
					min_carbon_value = np.min(total_substracted)
					index_min = total_substracted.index(min_carbon_value)
     
					rotation_angle_e_detuning = Omega_detuning*duration#*max(damping,damping_plus1) #the end is commented out because it is already taken into account by the dampening in Omega
					rotation_angle_e_other_frame = Omega_detuning_without_damping*duration#*max(damping,damping_plus1) #the end is commented out because it is already taken into account by the dampening in Omega
					alpha_damped = np.pi/2 if self.detuning == 0 else atan(Omega_detuning/self.detuning)
					axis_damped_detuning = [cos(phase)*cos(np.pi/2-alpha_damped),sin(phase)*cos(np.pi/2-alpha_damped),cos(alpha_damped)]
					axis_detuning_old = [cos(phase)*cos(np.pi/2-alpha),sin(phase)*cos(np.pi/2-alpha),cos(alpha)] #the squares are a test
					if damping != 0 and index_min !=0:
						axis_detuning = [cos(phase),sin(phase),0] # These values get normalized by NetSquid
						# print(f"check the axis {[axis_detuning[0], axis_detuning[1], axis_detuning[2]]}")
						# axis_detuning = np.multiply(axis_detuning,1/np.sqrt(axis_detuning[0]**2+axis_detuning[1]**2+axis_detuning[2]**2))
						# print(f"check the axis {axis_detuning[0]}")
					elif damping != 0 and index_min == 0:
						axis_detuning = [cos(phase)*cos(np.pi/2-alpha),sin(phase)*cos(np.pi/2-alpha),cos(alpha)] # These values get normalized by NetSquid
					else:
						axis_detuning = [1,0,0]
						rotation_angle_e_detuning = 0
					# print(f"old axis are {axis_detuning_old}")
					# print(f"rotation angle {rotation_angle_detuning} with detuning axis {axis_detuning}")
					# print(f" the sum of the axis is {sum(axis_detuning)}, axis squared are {np.sqrt(axis_detuning[0]**2+axis_detuning[1]**2+axis_detuning[2]**2)}")
					# Apply the rotation to the electron based on the calculated parameters.
					
					# needed_freq = total_freq[index_min]
					if self.single_instruction == True:
						if index_min == 0:
							self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e_detuning, axis = axis_detuning)
					else:
						if index_min == 0:
							self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e_detuning, axis = axis_detuning)
						else:
							self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e_other_frame, axis = axis_damped_detuning)

					### Perform a loop of calculations for every carbon nucleus ###
					for k in range(len(self.diamond.mem_positions)-3):
						# The iterater value skips the value for the electron position
						iterater_value = k+1

						# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
						A_parr_c = self.diamond.mem_positions[iterater_value].properties["A_parr"]

						# Calculate the resonance frequency for the carbon nucleus.
						omega_1_c = abs(omega_L_c-A_parr_c)

						# Calculate the difference between the resonance frequency and the radio frequency wave.
						delta_omega_c = abs(omega_1_c-omega_rf)
						
						# Potential to add an extra parameter for detuning, namely for carbon atoms
						# alpha = np.pi/2 if self.detuning == 0 else atan(Omega_c/self.detuning)
						alpha = np.pi/2 if self.detuning == 0 else atan(Omega_c/self.detuning)

						# print(f"omega_L_c = {omega_L_c}")
						# Calculate the damping factor due to the off resonance frequency
						damping = 1 if delta_omega_c == 0 else sin(atan(Omega_c/delta_omega_c))
						# print(f"delta omega = {delta_omega_c} with omega_1 = {omega_1_c} and rf {omega_rf}")
						# print(f"resulting in damping {damping}")
						damping = 0 if damping < 0.01 else damping
						# Calculate the rotation angle for the carbon nucleus, based on the evolution, duration and damping. 
						rotation_angle_c = Omega_c*duration*damping
						# Omega_c_detuning = np.sqrt((Omega_c*damping)**2+self.detuning**2)
						Omega_c_detuning = np.sqrt((Omega_c*damping)**2+self.detuning**2) #take into account taht this detuning is the detuning of the reference frame qubit with its own oscillator. At this moment the detuning is a single value, but should be changed to a list of detuned oscillators.
						alpha_damped = np.pi/2 if self.detuning == 0 else atan(Omega_c_detuning/self.detuning)
						axis_damped_detuning = [cos(phase)*cos(np.pi/2-alpha_damped),sin(phase)*cos(np.pi/2-alpha_damped),cos(alpha_damped)]
						Omega_c_other_frame = np.sqrt((Omega_c)**2+self.detuning**2)
						rotation_angle_c_detuning = Omega_c_detuning*duration#*damping
						rotation_c_other_frame = Omega_c_other_frame*duration
						# print(f"detuning omega_c {Omega_c_detuning} Omega_c {Omega_c}")
						# print(f"rotatin angle is {rotation_angle_c} with damping {damping}")

						if damping != 0:
							axis_detuning = [cos(phase)*cos(np.pi/2-alpha),sin(phase)*cos(np.pi/2-alpha),cos(alpha)] # These values get normalized by NetSquid
							# print(f"check the axis {[axis_detuning[0], axis_detuning[1], axis_detuning[2]]}")
							axis_detuning = np.multiply(axis_detuning,1/np.sqrt(axis_detuning[0]**2+axis_detuning[1]**2+axis_detuning[2]**2))
							# print(f"check the axis {axis_detuning[0]}")
						else:
							axis_detuning = [1,0,0]
							rotation_angle_c_detuning = 0

						# axis_detuning = [cos(phase)*sin(alpha),sin(phase)*sin(alpha),cos(alpha)]
						# Check if z influence is wanted, this option is added for algorithm testing purposes
						if self.single_instruction == True:
							if iterater_value == index_min:
								if self.no_z == 0:
									# Make the operation matrix needed for the control of the carbon nuclei
									first_matrix_value = (cos(-(omega_L_c-omega_rf)*duration/2)+1j*sin(-(omega_L_c-omega_rf)*duration/2))
									z_matrix_value = (cos(-(omega_1_c-omega_rf)*duration/2)+1j*sin(-(omega_1_c-omega_rf)*duration/2))
									operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,cos(Omega_c*duration/2)*damping,-(1j*cos(phase)+sin(phase))*(sin(Omega_c*duration/2)*damping)],[0,0,-(1j*cos(phase)-sin(phase))*(sin(Omega_c*duration/2))*damping,cos(Omega_c*duration/2)*damping]),dtype=np.complex_)

									#mathmetically describe this matrix as well with damping factor. Without damping factor, works for sure
									operation_matrix_detuning = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,cos(Omega_c*duration/2)*sin(alpha)+z_matrix_value*cos(alpha),-(1j*cos(phase)+sin(phase))*(sin(Omega_c*duration/2)*sin(alpha))],[0,0,-(1j*cos(phase)-sin(phase))*(sin(Omega_c*duration/2))*sin(alpha),cos(Omega_c*duration/2)*sin(alpha)+np.conj(z_matrix_value)*cos(alpha)]),dtype=np.complex_)
								
									# Add the matrix to an operator
									op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_behaviour")
									
									# Add the operator to an instruction
									INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)

									# Apply a rotation to every nucleus based on the parameters, the spin precision around the z axis is also included.
									self.apply(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,iterater_value])
								else:
									# print(f"actually performing rotation with {rotation_angle_c}")
									# Apply a rotation to every nucleus based on the parameters, the axis are not redifined, for the axis are the same as for the electron rotation.
									self.apply(instruction = INSTR_CROT, qubit_indices=[0,iterater_value], angle = rotation_c_other_frame, axis = axis_detuning)
						### Dipole coupling implementation

						else:
							if self.no_z == 0:
								# Make the operation matrix needed for the control of the carbon nuclei
								first_matrix_value = (cos(-(omega_L_c-omega_rf)*duration/2)+1j*sin(-(omega_L_c-omega_rf)*duration/2))
								operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,cos(Omega_c*duration/2)*damping,-(1j*cos(phase)+sin(phase))*(sin(Omega_c*duration/2)*damping)],[0,0,-(1j*cos(phase)-sin(phase))*(sin(Omega_c*duration/2))*damping,cos(Omega_c*duration/2)*damping]),dtype=np.complex_)
								
								# Add the matrix to an operator
								op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_behaviour")
								
								# Add the operator to an instruction
								INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)

								# Apply a rotation to every nucleus based on the parameters, the spin precision around the z axis is also included.
								self.apply(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,iterater_value])
							else:
								# Apply a rotation to every nucleus based on the parameters, the axis are not redifined, for the axis are the same as for the electron rotation.
								if iterater_value == index_min:
									self.apply(instruction = INSTR_CROT, qubit_indices=[0,iterater_value], angle = rotation_c_other_frame, axis = axis_detuning)
								else:
									self.apply(instruction = INSTR_CROT, qubit_indices=[0,iterater_value], angle = rotation_angle_c_detuning, axis = axis_damped_detuning)

						if self.Dipole_moment != None:
							print('are we in dipole influence?')
							for i in range(k): #for i<j
								J = self.Dipole_moment
								value = (cos(-(J)*duration/2)+1j*sin(-(J)*duration/2))
								matrix = [[value,0,0,0],[0,value,0,0],[0,0,value,0],[0,0,0,np.conj(value)]]
								op = Operator(name = "CZXY_gate", matrix = matrix, description="carbon_behaviour")
								INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)
								self.apply(instruction = INSTR_CZXY,operator = op, qubit_indices = [i,k])
			elif self.frame == "lab":
				if self.microwave.parameters["envelope"] == 0: # The envelope value is checked, this is currently the only supported envelope 
			
					# The difference between the resonance frequency and the microwave frequency is calculated
					# There are 2 values, because this is a triplet state, effective 2-level system is achieved by a large field splitting
					delta_omega = abs(omega_0-omega_rf)
					delta_omega_plus1 = abs(omega_0_plus1-omega_rf)
					
					# Determine the damping based on the off resonance frequency
					damping = 1 if delta_omega == 0 else sin(atan(Omega_e/delta_omega))
					damping_plus1 = 1 if delta_omega_plus1 == 0 else sin(atan(Omega_e/delta_omega_plus1))
				
					# The phase of the microwave is read, this is needed in order to dermine the axis of rotation
					phase = float(self.prog_name[4])
					# print(f"the phase is {phase}")
					# The axis of rotation are determined based on the phase
					axis = [cos(phase),sin(phase),0]					
					
					# The time duration of the microwave is read
					duration = float(self.prog_name[2])  +jitter

					# Calculated rotation on basis of the magnetic field strength, external magnetic field and duration
					# The rotation angle is determined by use of the evolution value multiplied by the time, however there is also a damping factor 
					# This damping factor comes due to the off resonance frequency.
					# When the frequency of the microwave is on resonance, the damping factor is 1, resulting in no damping.
					# Max damping is used, because only 1 electron matrix is given and the rotation needs to be performed if the microwave frequency is close to resonant to 1 of the Larmor frequencies of the electrons.

					damping = 0 if damping < 0.01 else damping
					damping_plus1 = 0 if damping_plus1 < 0.01 else damping_plus1
					rotation_angle_e = Omega_e*duration*max(damping,damping_plus1) 
					# print(f"the rotating angle is {rotation_angle_e}")
					# Apply the rotation to the electron based on the calculated parameters.
					if self.single_instruction == True:
						if damping >0.1:
							self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e, axis = axis)
					else:
						self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e, axis = axis)

					### Perform a loop of calculations for every carbon nucleus ###
					for k in range(len(self.diamond.mem_positions)-3): #-2 because there is a memory position for the electron and an additional one for a photon emission (only needed for mathmetical perposes)
						# The iterater value skips the value for the electron position
						iterater_value = k+1

						# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
						A_parr_c = self.diamond.mem_positions[iterater_value].properties["A_parr"]

						# Calculate the resonance frequency for the carbon nucleus.
						omega_1_c = abs(omega_L_c-A_parr_c)
						# print(f"omega_1 = {omega_1_c} with larmor {omega_L_c} and A_parr {A_parr_c}")
						
						# Calculate the difference between the resonance frequency and the radio frequency wave.
						delta_omega_c = abs(omega_1_c-omega_rf)
						# print(f"check delta_omega {delta_omega_c}")
						# Calculate the damping factor due to the off resonance frequency
						damping = 1 if delta_omega_c == 0 else sin(atan(Omega_c/delta_omega_c))
						# print(f"Omega_c {Omega_c}")
						# Calculate the rotation angle for the carbon nucleus, based on the evolution, duration and damping. 
						rotation_angle_c = Omega_c*duration*damping
						# print(f"the rotation angle of the carbon is {rotation_angle_c}")
						# print(f"the axis are {axis}")
						# Check if z influence is wanted, this option is added for algorithm testing purposes
						if self.single_instruction == True:
							if damping > 0.3:
								if self.no_z == 0:
									# Make the operation matrix needed for the control of the carbon nuclei
									first_matrix_value = (cos(-(omega_L_c)*duration/2)+1j*sin(-(omega_L_c)*duration/2))
									operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,cos(Omega_c*duration/2)*damping,-(1j*cos(phase)+sin(phase))*(sin(Omega_c*duration/2)*damping)],[0,0,-(1j*cos(phase)-sin(phase))*(sin(Omega_c*duration/2))*damping,cos(Omega_c*duration/2)*damping]),dtype=np.complex_)
									
									# Add the matrix to an operator
									op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_behaviour")
									
									# Add the operator to an instruction
									INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)

									# Apply a rotation to every nucleus based on the parameters, the spin precision around the z axis is also included.
									self.apply(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,iterater_value])
								else:
									# Apply a rotation to every nucleus based on the parameters, the axis are not redifined, for the axis are the same as for the electron rotation.
									self.apply(instruction = INSTR_CROT, qubit_indices=[0,iterater_value], angle = rotation_angle_c, axis = axis)

						else:
							if self.no_z == 0:
								# Make the operation matrix needed for the control of the carbon nuclei
								first_matrix_value = (cos(-(omega_L_c)*duration/2)+1j*sin(-(omega_L_c)*duration/2))
								operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,cos(Omega_c*duration/2)*damping,-(1j*cos(phase)+sin(phase))*(sin(Omega_c*duration/2)*damping)],[0,0,-(1j*cos(phase)-sin(phase))*(sin(Omega_c*duration/2))*damping,cos(Omega_c*duration/2)*damping]),dtype=np.complex_)
								
								# Add the matrix to an operator
								op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_behaviour")
								
								# Add the operator to an instruction
								INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)

								# Apply a rotation to every nucleus based on the parameters, the spin precision around the z axis is also included.
								self.apply(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,iterater_value])
							else:
								# Apply a rotation to every nucleus based on the parameters, the axis are not redifined, for the axis are the same as for the electron rotation.
								self.apply(instruction = INSTR_CROT, qubit_indices=[0,iterater_value], angle = rotation_angle_c, axis = axis)
			else:
				raise ValueError("This frame is not supported, choose either: 'lab' or 'rotating'")
	

		elif self.prog_name[0] == "wait":
			duration = self.prog_name[1]*1e-9

			if self.frame == "lab":
				angle = omega_0*duration*(2*np.pi)
				# Apply a rotation to every nucleus based on the parameters, the spin precision around the z axis is also included.
				self.apply(instruction = INSTR_ROT_Z, qubit_indices = [0],angle = angle)
				for k in range(len(self.diamond.mem_positions)-3): #-2 because there is a memory position for the electron and an additional one for a photon emission (only needed for mathmetical perposes)
					# The iterater value skips the value for the electron position
					iterater_value = k+1

					# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
					A_parr_c = self.diamond.mem_positions[iterater_value].properties["A_parr"]
					A_perp_c = self.diamond.mem_positions[iterater_value].properties["A_perp"]


					# Calculate the resonance frequency for the carbon nucleus.
					omega_1_c = np.sqrt(A_perp_c**2+(abs(omega_L_c-A_parr_c))**2)

					# Make the operation matrix needed for the control of the carbon nuclei
					first_matrix_value = (cos(-np.pi*2*(abs(omega_L_c))*duration/2)+1j*sin(-np.pi*2*(abs(omega_L_c))*duration/2))
					second_matrix_value = (cos(-np.pi*2*(omega_1_c)*duration/2)+1j*sin(-np.pi*2*(omega_1_c)*duration/2))
					
					operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,second_matrix_value,0],[0,0,0,np.conj(second_matrix_value)]),dtype=np.complex_)
					# print(f"the operations matrix is {operation_matrix}")
					# print(f"with duration {duration}")
					# Add the matrix to an operator
					op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_behaviour")
					
					# Add the operator to an instruction
					INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)
					# Apply a rotation to every nucleus based on the parameters, the spin precision around the z axis is also included.
					self.apply(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,iterater_value])		
			elif self.frame == "rotating":
			
				# omega_mw = omega_0-self.detuning
				angle = self.detuning*duration*(2*np.pi)
				self.apply(instruction = INSTR_ROT_Z, qubit_indices = [0],angle = angle)
				for k in range(len(self.diamond.mem_positions)-2): #-2 because there is a memory position for the electron and an additional one for a photon emission (only needed for mathmetical perposes)
					# The iterater value skips the value for the electron position
					iterater_value = k+1

					# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
					A_parr_c = self.diamond.mem_positions[iterater_value].properties["A_parr"]
					# print(f"check A_parr {A_parr_c}")
					# Calculate the resonance frequency for the carbon nucleus.
					omega_1_c = omega_L_c-A_parr_c
					# print(f"duration is {duration}")
					# print(f"rotation is {A_parr_c*duration*2*np.pi}")
					# in case of -self.detuning, this results in +detuning.
					# Make the operation matrix needed for the control of the carbon nuclei
					first_matrix_value = (cos(-2*np.pi*(A_parr_c+self.detuning)*duration/2)+1j*sin(-2*np.pi*(A_parr_c+self.detuning)*duration/2))
					# first_matrix_value = (cos(-1*(2*np.pi*(self.detuning)+A_parr_c)*duration/2)+1j*sin(-1*(2*np.pi*(self.detuning)+A_parr_c)*duration/2))

					second_matrix_value = (cos(-2*np.pi*(self.detuning)*duration/2)+1j*sin(-2*np.pi*(self.detuning)*duration/2))
					# first_matrix_value = (1+1j)
					# second_matrix_value = (cos(-(self.detuning)*duration/2)+1j*sin(-(self.detuning)*duration/2))

					
					operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,second_matrix_value,0],[0,0,0,np.conj(second_matrix_value)]),dtype=np.complex_)
					# print(f"the operation matrix is {operation_matrix}")
					# Add the matrix to an operator
					# print(operation_matrix)
					op = Operator(name = "CZXY_gate", matrix = operation_matrix, description="carbon_behaviour")
					
					# Add the operator to an instruction
					INSTR_CZXY = IGate(name = op.name, operator = op, num_positions=2)
					# print(f"the iterator value is {iterater_value}")
					# print(f"the operator is {op} with matrix ")
					# Apply a rotation to every nucleus based on the parameters, the spin precision around the z axis is also included.
					self.apply(instruction = INSTR_CZXY,operator = op, qubit_indices = [0,iterater_value])	
			else:
				raise ValueError("This frame is not supported, choose either: 'lab' or 'rotating'")
		elif self.prog_name[0] == "numerical_shift":
			qubit_indices = int(self.prog_name[1])
			angle = float(self.prog_name[2])
			self.apply(instruction = INSTR_ROT_Z, qubit_indices = [qubit_indices],angle = angle)
		else:
			# Raise an error if an unknown program is asked to be implemented.
			raise ValueError("not a known program")
		
		# Ensure that the program waits up untill the moment that the whole code is being simulated
		yield self.run() 

