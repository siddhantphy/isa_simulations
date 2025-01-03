import numpy as np
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.components.models import DepolarNoiseModel, T1T2NoiseModel
from netsquid.components.instructions import INSTR_X, INSTR_Y, INSTR_Z, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_H,\
    INSTR_MEASURE, INSTR_SWAP, INSTR_INIT, INSTR_CXDIR, INSTR_EMIT, INSTR_CX, INSTR_CROT, INSTR_CYDIR, INSTR_ROT, INSTR_CROT_Y
from netsquid.qubits.operators import Operator
from netsquid_nv.delft_nvs.delft_nv_2020_near_term import NVParameterSet2020NearTerm
from netsquid.components.instructions import *


class NVQuantumProcessor(QuantumProcessor):
    """NV center quantum processor capable of emitting entangled photons.

    The default properties correspond to the Hanson group's setup in Delft, last updated in 2020. For more details on
    these values, see :class:`~netsquid_nv.delft_nvs.delft_nv_2020_near_term.NVParameterSet2020NearTerm` and parent
    classes. For details on the modelling, see appendices D-F of https://arxiv.org/abs/2010.12535 as well as Appendix D
    of https://arxiv.org/abs/1903.09778.

    Notes:

    1. The :class:`netsquid.components.qprocessor.QuantumProcessor` is initialized with one more position than what
    is specified. The reason for this is that NetSquid requires an extra position for emission, e.g. if we want to
    emit the qubit held in position 1, its state gets first placed in position emission_position and only then can be
    sent to the outside world. We define the emission position to be the last in the quantum memory, and the only
    instruction associated to it is :class:`netsquid.components.instructions.INSTR_EMIT`. It is highly recommended that
    other instructions with a topology including this position are not added.

    2. The `use_magical_swap` parameter defines whether a SWAP instruction between the electron and a carbon qubit is
    added. The noise is approximated by the square of the depolarizing probability of the two-qubit gate, as the actual
    SWAP circuit contains two such gates, and they dominate the noise (see Fig. 1C, orange box of
    https://arxiv.org/abs/1703.03244). The reason for including this option, despite being less physically accurate,
    is that it does not merge the quantum states, thus keeping the size of the quantum state from blowing up and
    allowing simulation of some otherwise unfeasible scenarios.

    3. We currently set `emission_duration` to be the same as `photon_emission_delay`. One could also get rid of the
    name `photon_emission_delay` entirely, but we choose not to because even though `photon_emission_delay` and
    `emission_duration` take the same value for the model used in this snippet, this is not necessarily the case.
    `photon_emission_delay` refers exclusively to the time a photon takes to be emitted, whereas the `emission_duration`
    could also include e.g. some time taken to get the emitted photon out of the NV. If one ever wants to take this into
    account, it can simply be done by adding it to the `emission_duration` property.

    Parameters
    ----------
    num_positions : int
        Number of qubits in the NV center. One of the positions is an electron, the other are carbons.
    electron_position : int, optional
        Position for electron. Default 0.
    noiseless : bool, optional
        If True, all operations are perfect. If False (default) standard values are used for noise models.
    properties : dict
        Use to manually specify properties.
    """

    OPERATION_DURATIONS = {
        "carbon_init_duration": 310E3,
        "carbon_z_rot_duration": 20E3,
        "electron_init_duration": 2E3,
        "electron_single_qubit_duration": 5,
        "ec_two_qubit_gate_duration": 500E3,
        "measure_duration": 3.7E3,
        "magical_swap_gate_duration": 1.000010E6
    }

    OTHER_PARAMETERS = {
        "use_magical_swap": False
    }

    def __init__(self, num_positions, electron_position=0, noiseless=False, **properties):
        default_parameter_set = NVParameterSet2020NearTerm()
        params = default_parameter_set.to_dict() if not noiseless else default_parameter_set.to_perfect_dict()
        params.update(self.OPERATION_DURATIONS.copy())
        params.update(self.OTHER_PARAMETERS.copy())
        params.update(properties)
        for property, value in params.items():
            self.add_property(name=property, value=value, mutable=False)

        self.emission_duration = self.properties["photon_emission_delay"]
        self.electron_position = electron_position
        super().__init__(name="nv_center_quantum_processor",
                         num_positions=num_positions + 1,
                         mem_noise_models=self._define_memory_noise_models(num_positions))
        # self.electron_freq = 200
        # self.NV_state = "NV-" #NV0, is the other possibility
        print("the noise is next within init")
        # print(self.mem_noise_models)
        # print(mem_noise_models)
        print(self.name)
        # print(self.memory_noise_models)
        # print(dir(self.nv_super_instructions))
        print(self.num_positions)
        # print(self._mem_noise_models)
        print("mem positions next")

        # self.mem_positions[0].add_property(name = "frequency", value = 200)
        print(self.mem_positions[0].properties)
        # print_freq = self.mem_positions[0].properties["frequency"]
        # print(f" the frequency of the electron is {print_freq}")
        self._set_physical_instructions()

    def _define_memory_noise_models(self, num_positions):
        """Defines noise models for the quantum processor's memory positions. We add no memory noise model to the
        emission position.
        """
        print_prop = self.properties["electron_T2"]
        print(f"T2 is {print_prop}")
        
        electron_qubit_noise = T1T2NoiseModel(T1=self.properties["electron_T1"], T2=self.properties["electron_T2"])
        carbon_qubit_noise = T1T2NoiseModel(T1=self.properties["carbon_T1"], T2=self.properties["carbon_T2"])
        print("electron qubit noise next")
        print(dir(electron_qubit_noise))
        self.electron_position = 0  # hardcode electron at 0 (?)
        self.carbon_positions = [pos + 1 for pos in range(num_positions - 1)]
        self.emission_position = num_positions
        mem_noise_models = [electron_qubit_noise] + \
                           [carbon_qubit_noise] * len(self.carbon_positions) + [None]
        print(mem_noise_models)
        return mem_noise_models

    def _define_instruction_noise_models(self):
        """Defines noise models for the quantum processor's instructions.
        """

        electron_init_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["electron_init_depolar_prob"],
                              time_independent=True)

        electron_single_qubit_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["electron_single_qubit_depolar_prob"],
                              time_independent=True)

        carbon_init_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["carbon_init_depolar_prob"],
                              time_independent=True)

        carbon_z_rot_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["carbon_z_rot_depolar_prob"],
                              time_independent=True)

        ec_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["ec_gate_depolar_prob"],
                              time_independent=True)

        magical_swap_gate_depolar_prob = 1 - (1 - self.properties["ec_gate_depolar_prob"]) ** 2
        magic_swap_noise = DepolarNoiseModel(depolar_rate=magical_swap_gate_depolar_prob,
                                             time_independent=True)

        self.models["electron_init_noise"] = electron_init_noise
        self.models["electron_single_qubit_noise"] = electron_single_qubit_noise
        self.models["carbon_init_noise"] = carbon_init_noise
        self.models["carbon_z_rot_noise"] = carbon_z_rot_noise
        self.models["ec_noise"] = ec_noise
        self.models["magic_swap_noise"] = magic_swap_noise

    def _set_physical_instructions(self):
        """Initializes the quantum processor's instructions.
        """
        # self._define_instruction_noise_models()

        phys_instructions = []

        phys_instructions.append(
            PhysicalInstruction(INSTR_INIT,
                                parallel=False,
                                topology=self.carbon_positions,
                                # q_noise_model=self.models["carbon_init_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_init_duration"]))

        phys_instructions.append(
            PhysicalInstruction(INSTR_ROT_Z,
                                parallel=False,
                                topology=self.carbon_positions,
                                # q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_z_rot_duration"]))
        electron_carbon_topologies = \
            [(self.electron_position, carbon_pos) for carbon_pos in self.carbon_positions]
        INSTR_CZXY = IGate(name = "CZXY_gate", operator = None,num_positions=    2)
        phys_instructions.append(
            PhysicalInstruction(INSTR_CZXY,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                # q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration = 0
                                ))


        phys_instructions.append(
            PhysicalInstruction(INSTR_INIT,
                                parallel=False,
                                topology=[self.electron_position],
                                # q_noise_model=self.models["electron_init_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["electron_init_duration"]))

        for instr in [INSTR_X, INSTR_Y, INSTR_Z, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_H, INSTR_ROT]:
            phys_instructions.append(
                PhysicalInstruction(instr,
                                    parallel=False,
                                    topology=[self.electron_position],
                                    # q_noise_model=self.models["electron_single_qubit_noise"],
                                    apply_q_noise_after=False,
                                    # duration=self.properties["electron_single_qubit_duration"]
                                    duration = 0))
        emit_topology = [(self.electron_position, self.emission_position)]
        #emit_topology = [(carbon_pos, self.emission_position) for carbon_pos in self.carbon_positions]

        phys_instructions.append(
            PhysicalInstruction(INSTR_EMIT,
                                paralell=False,
                                topology=emit_topology,
                                duration=self.emission_duration))

        electron_carbon_topologies = \
            [(self.electron_position, carbon_pos) for carbon_pos in self.carbon_positions]
        phys_instructions.append(
            PhysicalInstruction(INSTR_CXDIR,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                # q_noise_model=self.models["ec_noise"],
                                # apply_q_noise_after=True,
                                duration=500e3))
        phys_instructions.append(
            PhysicalInstruction(INSTR_CYDIR,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                # q_noise_model=self.models["ec_noise"],
                                # apply_q_noise_after=True,
                                duration=500e3))
                                
        phys_instructions.append(             
            PhysicalInstruction(INSTR_CX,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                # q_noise_model=self.models["ec_noise"],
                                # apply_q_noise_after=True,
                                duration=500e3))
        
    
        
        phys_instructions.append(             
            PhysicalInstruction(INSTR_CROT,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                # q_noise_model = None,
                                # q_noise_model=self.models["ec_noise"],
                                apply_q_noise_after=False,
                                duration=0)) #500e-3
        topology_electron_carbon_2 = [(1,0), (0,1)]
        phys_instructions.append(             
            PhysicalInstruction(INSTR_CROT_Y,
                                parallel=False,
                                topology=topology_electron_carbon_2,
                                # q_noise_model = None,
                                # q_noise_model=self.models["ec_noise"],
                                apply_q_noise_after=False,
                                duration=0)) #500e-3

        M0 = Operator("M0",
                      np.diag([np.sqrt(1 - self.properties["prob_error_0"]), np.sqrt(self.properties["prob_error_1"])]))
        M1 = Operator("M1",
                      np.diag([np.sqrt(self.properties["prob_error_0"]), np.sqrt(1 - self.properties["prob_error_1"])]))

        phys_instr_measure = PhysicalInstruction(INSTR_MEASURE,
                                                 parallel=False,
                                                 topology=[self.electron_position],
                                                 q_noise_model=None,
                                                 duration=self.properties["measure_duration"],
                                                 meas_operators=[M0, M1])

        phys_instructions.append(phys_instr_measure)

        if self.properties["use_magical_swap"]:
            phys_instructions.append(
                PhysicalInstruction(INSTR_SWAP,
                                    parallel=False,
                                    topology=electron_carbon_topologies,
                                    q_noise_model=self.models["magic_swap_noise"],
                                    apply_q_noise_after=True,
                                    duration=self.properties["magical_swap_gate_duration"]))

        for instruction in phys_instructions:
            self.add_physical_instruction(instruction)

    @property
    def electron_position(self):
        return self._electron_position

    @electron_position.setter
    def electron_position(self, position):
        if position < 0:
            raise ValueError("Electron position cannot be negative.")
        self._electron_position = position

