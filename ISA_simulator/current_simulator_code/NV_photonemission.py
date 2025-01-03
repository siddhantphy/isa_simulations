import numpy as np

from photon_emission import PhotonEmissionNoiseModel

from netsquid_nv.bessel_quotient import bessel_quotient

__all__ = [
    'NVPhotonEmissionNoiseModel',
]


class NVPhotonEmissionNoiseModel(PhotonEmissionNoiseModel):

    def __init__(self, delta_w=None, tau_decay=None, tau_emission=None, time_window=None,
                 std_electron_photon_phase_drift=None, p_zero_phonon=1, collection_eff=1, p_double_exc=0):
        """
        For every entanglement generation attempt the nuclear spins will experience dephasing noise such that the
        length of the Bloch vector in the X-Y gets exponentially suppressed.
        The noise model is based on 'arXiv:1802.05996' and in particular eq. (2).
        Parameters determining the dephasing noise the nuclear spins experience depend on 'alpha'
        (bright state population), 'delta_w' (coupling strength) and 'tau_decay' (decay constant).

        Effective amplitude damping noise will be applied if 'tau_emission' and 'time_window' is set and if both
        arguments are greater then 0. See notes by Filip Rozpedek.

        Additional amplitude damping noise will be due to non-unity probability ('p_zero_phonon') of emitting a photon
        in the zero phonon line and non-perfect collection efficiency ('collection_eff') in to the fibre.

        Applies effective dephasing noise to photon due to uncertainty in the phase between
        |00> and |11> of the photon-memory state.
        This depends on the standard deviation on this phase, specified by the node.

        Parameters
        ----------
        :param delta_w : float or list of floats
            Coupling strength (kHz/(2pi))
            If a single float the same value is used for all nuclear spins otherwise
            the corresponing floats in the list will be used.
        :param tau_decay : float
            Decay constant (ns)
            If a single float the same value is used for all nuclear spins otherwise
            the corresponing floats in the list will be used.
        :param tau_emission : float
            Characteristic time of the NV emission (ns)
        :param time_window: float
            The length of the time window which mid source detects photons (ns)
        :param std_electron_photon_phase_drift : float
            Standard deviation of the relative phase in the memory-photon entangled state (degrees)
        :param p_zero_phonon: float
            Probability that emitted photon is in the zero phonon line.
        :param collection_eff: float
            Collection efficiency into the fibre.
        :param p_double_exc: float
            Probability of two photons being excited.
        """
        if isinstance(delta_w, list) and isinstance(tau_decay, list):
            if not len(delta_w) == len(tau_decay):
                raise ValueError("If delta_w and tau_decay are lists, these need to be the same length.")
        if not ((p_zero_phonon >= 0) and (p_zero_phonon <= 1)):
            raise ValueError("p_zero_phonon needs to be in the interval [0,1]")
        if not ((collection_eff >= 0) and (collection_eff <= 1)):
            raise ValueError("p_zero_phonon needs to be in the interval [0,1]")
        if not ((p_double_exc >= 0) and (p_double_exc <= 1)):
            raise ValueError("p_double_exc needs to be in the interval [0,1]")
        # print("checking init of nv_photon_model")
        self.delta_w = delta_w
        self.tau_decay = tau_decay
        self.tau_emission = tau_emission
        self.time_window = time_window
        self.std_electron_photon_phase_drift = std_electron_photon_phase_drift
        self.p_zero_phonon = p_zero_phonon
        self.collection_eff = collection_eff
        self.p_double_exc = p_double_exc

        self._calculate_dephasing_photon()

    def _calculate_dephasing_nuclear(self, alpha, dw, td):
        """
        Calculates the dephasing parameter for noise after one entanglement attempt.
        """

        return alpha / 2 * (1 - np.exp(-(2 * np.pi * dw * td * 10 ** (-6)) ** 2 / 2))

    def _calculate_dephasing_photon(self):
        """
        Calculates the dephasing parameter for noise due to uncertainty in phase in the photon-memory entanglement.
        """
        if (self.std_electron_photon_phase_drift is None) or (self.std_electron_photon_phase_drift == 0):
            self._dp_photon = 0
        else:
            # Standard deviation in radians
            dr = self.std_electron_photon_phase_drift
            # Compute dephasing parameter
            self._dp_photon = (1 - bessel_quotient(0, 1 / (dr) ** 2, 0.001)) / 2

    def noise_operation(self, qubits, delta_time=0, operator=None, **kwargs):
        """
        Apply all the different noise processes to the photon, electron and nuclear spins.
        """
        try:
            alpha = kwargs['alpha']
        except KeyError:
            raise ValueError("key-argument 'alpha' was expected for 'photon_emission'.")

        # Get the photon
        photon = qubits[1]
        nuclear_spins = qubits[2:]
        print(f"the nuclear spins are {nuclear_spins} and with alpha = {alpha}")
        # Apply dephasing noise to nuclear spins
        print("checking if in noise operation")
        self._nuclear_noise_from_electron_reset(nuclear_spins, alpha)

        # Add effective dephasing noise due to uncertainty in phase
        self._phase_uncertainty_noise(photon)

        # Add effective dephaisng noise due to double excitation events
        self._double_excitation_noise(photon)

        # Apply effective amplitude damping noise due to superpostition of multiple time modes
        self._coherent_emission_noise(photon)

        # Apply effective amplitude damping noise due to non-unity collection efficiency and
        # probability of emitting in zero phonon line.
        self._collection_efficienty_noise(photon)

        return photon

    def _nuclear_noise_from_electron_reset(self, nuclear_spins, alpha):
        """
        For every entanglement generation attempt the nucler spins will experience dephasing noise such
        that the length of the Bloch vector in the X-Y gets exponentially supressed.
        The noise model is based on 'arXiv:1802.05996' and in particular eq. (2).
        """
        print("checking if in nuclear noise")
        print(f"checking values next {nuclear_spins, alpha}")
        for i in range(len(nuclear_spins)):
            q = nuclear_spins[i]
            if q is not None:
                if not ((self.delta_w is None) or (self.tau_decay is None)):
                    if isinstance(self.delta_w, list):
                        dw = self.delta_w[i]
                    else:
                        dw = self.delta_w
                    if isinstance(self.tau_decay, list):
                        td = self.tau_decay[i]
                    else:
                        td = self.tau_decay
                    dp = self._calculate_dephasing_nuclear(alpha, dw, td)
                    print(f"the dp value is {dp}")

                    self._random_dephasing_noise(q, dp)
                    print(q.qstate.qrepr)

    def _phase_uncertainty_noise(self, photon):
        """
        Applies effective dephasing noise to photon due to uncertainty in the phase between
        |00> and |11> of the photon-memory state.
        This depends on the standard deviation on this phase, specified by the node.
        """

        # Apply dephasing noise
        self._random_dephasing_noise(photon, self._dp_photon)

    def _double_excitation_noise(self, photon):
        """
        Applies effective dephasing noise to photon due to double excitation event.
        This is motivated at section III in https://arxiv.org/src/1712.07567v2/anc/SupplementaryInformation.pdf
        """
        # Apply dephasing noise
        self._random_dephasing_noise(photon, self.p_double_exc / 2)

    def _coherent_emission_noise(self, photon):
        """
        Applies effective amplitude damping noise to photon due to superposition of multiple time modes.
        Currently specific for NV.
        Amplitude damping parameter depends on 'self.tau_emission' (characteristic time of the NV emission)
        and 'self.time_window' (The length of the time window which mid source detects photons).

        If 'time_window = None', 'tau_emission = None' or 'tau_emission = 0', no amplitude damping is applied.
        """

        if self.tau_emission is not None:

            if (self.time_window is not None) and (self.tau_emission > 0):
                # Calculate amplitude damping parameter
                probAD = np.exp(-self.time_window / self.tau_emission)

                # Apply amplitude damping noise to photon
                self._random_amplitude_dampen(photon, probAD)

    def _collection_efficienty_noise(self, photon):
        tot_probAD = (1 - self.p_zero_phonon * self.collection_eff)

        # Apply amplitude damping noise to photon
        self._random_amplitude_dampen(photon, tot_probAD)
