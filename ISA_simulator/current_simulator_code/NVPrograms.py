from netsquid.components import QuantumProgram, INSTR_ROT, INSTR_CROT, INSTR_ROT_Z
from math import atan, sin, cos
import numpy as np
from netsquid.qubits.operators import Operator
from netsquid.components.instructions import IGate

class Programs_microwave(QuantumProgram):
	def __init__(self, prog_name, diamond, no_z = 1, detuning = 0,single_instruction = True, frame = "rotating", jitter = 0):
		super().__init__()
		self.prog_name = prog_name # In this program name, the instruction which needs to be performed is stored
		self.diamond = diamond # The diamond is needed, because properties of the qubits are needed to know how far the rotations need to be done 
		self.ext_magnet = diamond.supercomponent.supercomponent.subcomponents["external_magnet"] # The external magnetic field is needed in order to calculate the resonance frequency
		self.no_z = no_z # This is a parameter which determines if a z spin precission is wanted (this can be used to turn off unwanted rotations for testing purposes)
		self.microwave = diamond.supercomponent.subcomponents["microwave"] # The microwave component is needed, because it contains information regarding the microwave shape and oscillating magnitude
		self.detuning = detuning
		self.single_instruction = single_instruction
		self.frame = frame
		self.jitter = jitter

	def program(self):
		### First all the parameters are determined ###
		# Zero field splitting of the electron ms = -1,1 states with regards to the ms = 0 state. 
		# This is the resonance frequency when there is no magnetic field present
		D = 2.87e9  #Hz
		
		jitter = np.random.rand()*self.jitter-self.jitter/2
		#read the value for the z-direction of the external magnetic field
		B_z = self.ext_magnet.parameters["magnetic_field"] 

		#read the oscillating magnetic field strenght value
		B_osc = self.microwave.parameters["B_osc"]

		#gyromagnetic ratio for the electron
		gamma_e = 28.7e9 #Hz/T

		# Calculate the resonance frequency for the ms = -1 electron state
		omega_0 = D-gamma_e*B_z 

		# Calculate the resonance frequency for the ms = 1 electron state
		omega_0_plus1 = D+gamma_e*B_z 

		# Calculate the optimal evolution over time 
		Omega_e = B_osc*gamma_e

		### Next the values for the carbon nuclei are given ###
		gamma_c = 1.07e7 #gyromagnetic ratio carbon (Hz/T)
		omega_L_c = gamma_c*B_z # Larmor frequency carbon
		Omega_c = B_osc*gamma_c #optimal evolution over time

		
		if self.prog_name[0] == "excite_mw": # This is the only instruction important to the microwave generator
			# The frequency of the microwave/radiofrequency wave
			# Maybe change to + detuning instead of - detuning. Then accept detuning to be positive and negative
			omega_rf = float(self.prog_name[3])-self.detuning
			
		
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
					Omega_effective = Omega_e*damping
					rotation_angle_e = Omega_e*duration*max(damping,damping_plus1) #prelimenary implementation of /2*pi. 
					Omega_detuning = np.sqrt((Omega_e*damping)**2+self.detuning**2)
					#next rotation angle is experimental for detuning coding
					# print(f"detuning omega {Omega_detuning} Omega_e {Omega_e} detuning {self.detuning}")
					# print(f"alpha is {alpha}")
					rotation_angle_e_detuning = Omega_detuning*duration#*max(damping,damping_plus1) #the end is commented out because it is already taken into account by the dampening in Omega
					axis_detuning_old = [cos(phase)*cos(np.pi/2-alpha),sin(phase)*cos(np.pi/2-alpha),cos(alpha)] #the squares are a test
					if damping != 0:
						axis_detuning = [Omega_effective*cos(phase)*cos(np.pi/2-alpha),Omega_effective*sin(phase)*cos(np.pi/2-alpha),self.detuning*cos(alpha)] # These values get normalized by NetSquid
						# print(f"check the axis {[axis_detuning[0], axis_detuning[1], axis_detuning[2]]}")
						axis_detuning = np.multiply(axis_detuning,1/np.sqrt(axis_detuning[0]**2+axis_detuning[1]**2+axis_detuning[2]**2))
						# print(f"check the axis {axis_detuning[0]}")
					else:
						axis_detuning = [1,0,0]
						rotation_angle_e_detuning = 0
					# print(f"old axis are {axis_detuning_old}")
					# print(f"rotation angle {rotation_angle_detuning} with detuning axis {axis_detuning}")
					# print(f" the sum of the axis is {sum(axis_detuning)}, axis squared are {np.sqrt(axis_detuning[0]**2+axis_detuning[1]**2+axis_detuning[2]**2)}")
					# Apply the rotation to the electron based on the calculated parameters.
					if self.single_instruction == True:
						if damping >0.1:
							self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e_detuning, axis = axis_detuning)
					else:
						self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e_detuning, axis = axis_detuning)

					### Perform a loop of calculations for every carbon nucleus ###
					for k in range(len(self.diamond.mem_positions)-2):
						# The iterater value skips the value for the electron position
						iterater_value = k+1

						# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
						A_parr_c = self.diamond.mem_positions[iterater_value].properties["A_parr"]

						# Calculate the resonance frequency for the carbon nucleus.
						omega_1_c = omega_L_c-A_parr_c

						# Calculate the difference between the resonance frequency and the radio frequency wave.
						delta_omega_c = abs(omega_1_c-omega_rf)
						
						# Potential to add an extra parameter for detuning, namely for carbon atoms
						# alpha = np.pi/2 if self.detuning == 0 else atan(Omega_c/self.detuning)
						alpha = np.pi/2 if self.detuning == 0 else atan(Omega_c/self.detuning)


						# Calculate the damping factor due to the off resonance frequency
						damping = 1 if delta_omega_c == 0 else sin(atan(Omega_c/delta_omega_c))
						damping = 0 if damping < 0.01 else damping
						# Calculate the rotation angle for the carbon nucleus, based on the evolution, duration and damping. 
						rotation_angle_c = Omega_c*duration*damping
						Omega_c_detuning = np.sqrt((Omega_c*damping)**2+self.detuning**2)
						rotation_angle_c_detuning = Omega_c_detuning*duration#*damping
						if damping != 0:
							axis_detuning = [Omega_c_detuning*cos(phase)*cos(np.pi/2-alpha),Omega_c_detuning*sin(phase)*cos(np.pi/2-alpha),self.detuning*cos(alpha)] # These values get normalized by NetSquid
							# print(f"check the axis {[axis_detuning[0], axis_detuning[1], axis_detuning[2]]}")
							axis_detuning = np.multiply(axis_detuning,1/np.sqrt(axis_detuning[0]**2+axis_detuning[1]**2+axis_detuning[2]**2))
							# print(f"check the axis {axis_detuning[0]}")
						else:
							axis_detuning = [1,0,0]
							rotation_angle_c_detuning = 0

						# axis_detuning = [cos(phase)*sin(alpha),sin(phase)*sin(alpha),cos(alpha)]
						# Check if z influence is wanted, this option is added for algorithm testing purposes
						if self.single_instruction == True:
							if damping > 0.3:
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
									# Apply a rotation to every nucleus based on the parameters, the axis are not redifined, for the axis are the same as for the electron rotation.
									self.apply(instruction = INSTR_CROT, qubit_indices=[0,iterater_value], angle = rotation_angle_c, axis = axis)
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

								self.apply(instruction = INSTR_CROT, qubit_indices=[0,iterater_value], angle = rotation_angle_c, axis = axis)
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
					
					# Apply the rotation to the electron based on the calculated parameters.
					if self.single_instruction == True:
						if damping >0.1:
							self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e, axis = axis)
					else:
						self.apply(instruction = INSTR_ROT, qubit_indices = 0, angle = rotation_angle_e, axis = axis)

					### Perform a loop of calculations for every carbon nucleus ###
					for k in range(len(self.diamond.mem_positions)-2): #-2 because there is a memory position for the electron and an additional one for a photon emission (only needed for mathmetical perposes)
						# The iterater value skips the value for the electron position
						iterater_value = k+1

						# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
						A_parr_c = self.diamond.mem_positions[iterater_value].properties["A_parr"]

						# Calculate the resonance frequency for the carbon nucleus.
						omega_1_c = omega_L_c-A_parr_c

						# Calculate the difference between the resonance frequency and the radio frequency wave.
						delta_omega_c = abs(omega_1_c-omega_rf)
						
						# Calculate the damping factor due to the off resonance frequency
						damping = 1 if delta_omega_c == 0 else sin(atan(Omega_c/delta_omega_c))
						
						# Calculate the rotation angle for the carbon nucleus, based on the evolution, duration and damping. 
						rotation_angle_c = Omega_c*duration*damping

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
				for k in range(len(self.diamond.mem_positions)-2): #-2 because there is a memory position for the electron and an additional one for a photon emission (only needed for mathmetical perposes)
					# The iterater value skips the value for the electron position
					iterater_value = k+1

					# Get the parallel hyperfine interaction parameter for every carbon nucleus, this differs per nucleus.
					A_parr_c = self.diamond.mem_positions[iterater_value].properties["A_parr"]

					# Calculate the resonance frequency for the carbon nucleus.
					omega_1_c = omega_L_c-A_parr_c

					# Make the operation matrix needed for the control of the carbon nuclei
					first_matrix_value = (cos(-np.pi*2*(omega_L_c)*duration/2)+1j*sin(-np.pi*2*(omega_L_c)*duration/2))
					second_matrix_value = (cos(-np.pi*2*(omega_1_c)*duration/2)+1j*sin(-np.pi*2*(omega_1_c)*duration/2))
					
					operation_matrix = np.array(([first_matrix_value,0,0,0],[0,np.conj(first_matrix_value),0,0],[0,0,second_matrix_value,0],[0,0,0,np.conj(second_matrix_value)]),dtype=np.complex_)
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

