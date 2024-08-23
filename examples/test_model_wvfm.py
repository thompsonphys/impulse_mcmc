import numpy as np
from cbc_injection_generator import CBCInjectionGenerator

# standard zero-noise matched-filter likelihood
class LnLikelihoodWvfm():
    def __init__(self, config_file):
        """class init

        Parameters
        ----------
        config_file: str
            file path to config file containing injection and recovery parameters
        """
        self.config_file = config_file
        self.injection = CBCInjectionGenerator(config_file)

        self.ifo = self.injection.ifos[0]
        self.recovery_waveform_generator = self.injection.waveform_generator

        self.frequency_mask = self.ifo.frequency_mask
        self.psd = self.ifo.power_spectral_density_array[self.frequency_mask]

        self.signal_polarizations = self.injection.injection_generator.frequency_domain_strain(self.injection.injection_parameters)
        self.signal_response = self.ifo.get_detector_response(self.signal_polarizations, self.injection.injection_parameters)[self.frequency_mask]

        self.delta_f = self.ifo.frequency_array[1] - self.ifo.frequency_array[0]

        self.ntemps = self.injection.recovery_options['ntemps']
        self.nsamples = self.injection.recovery_options['nsamples']

    def __call__(self, params):
        """Generate likelihood evaluation

        Parameters
        ----------
        params: dict
            dictionary of CBC parameters for template
        
        """
        try:
            mchirp = params[0]
            eta = params[1]
            mass_1, mass_2 = mass1_mass2_from_chirpmass_eta(mchirp, eta)
            spin1z = params[2]
            spin2z = params[3]
        except Exception:
            return -np.inf

        template_params = self.injection.injection_parameters.copy()
        template_params['mass_1'] = mass_1
        template_params['mass_2'] = mass_2
        template_params['chi_1'] = spin1z
        template_params['chi_2'] = spin2z

        template_polarizations = self.recovery_waveform_generator.frequency_domain_strain(template_params)
        template_response = self.ifo.get_detector_response(template_polarizations, template_params)[self.frequency_mask]

        result = get_overlap(template_response, self.signal_response, self.psd, self.delta_f)

        if np.isfinite(result):
            return result
        else:
            return -np.inf
        
def overlap(h1, h2, psd, delta_f):
    """Computes noise-weighted inner product

    Parameters
    ----------
    h1: complex numpy array
        complex array of frequency domain strain
    h2: complex numpy array
        complex array of frequency domain strain
    psd: numpy array
        power spectral density of detector
    delta_f: float
        frequency spacing of data
    """
    integrand = np.dot(h1, h2.conj() / psd)
    return 4.0 * delta_f * integrand.real

def get_overlap(template, signal, psd, delta_f):
    """
    Wrapper to get the loglikelihood from the noise-weighted inner product between 
    signal and model.

    Parameters
    ----------
    template: numpy array
        complex array of frequency domain template detector response
    signal: numpy array
        complex array of frequency domain signal detector response
    psd: numpy array
        power spectral density of detector
    delta_f: float
        frequency spacing
    """

    overlap_sig_temp = overlap(template, signal, psd, delta_f)
    overlap_sig_sig = overlap(signal, signal, psd, delta_f)
    overlap_temp_temp = overlap(template, template, psd, delta_f)

    return overlap_sig_temp - 0.5 * ( overlap_sig_sig + overlap_temp_temp )

# uniform prior
class LnPriorUniformWvfm():
    def __init__(self, mins, maxes):
        """
        mins: vector of minima on uniform prior
        maxes: vector of maxima on uniform prior
        """
        self.rng = np.random.default_rng()
        self.mins = mins
        self.maxes = maxes

    def __call__(self, params):
        if np.any(params < self.mins) or np.any(params > self.maxes):
            return -np.inf
        else:
            return 0

    def initial_sample(self):
        return self.rng.uniform(self.mins, self.maxes)

def mass1_mass2_from_chirpmass_eta(chirp_mass, eta):
    """Function to produce values of mass 1 and mass 2 from the
    chirp mass and symmetric mass ratio

    Args:
        chirp_mass (float): chirp mass
        eta (float): symmetric mass ratio

    Returns:
        float, float: returns mass1 and mass2, mass1 >= mass2
    """
    total_mass = chirp_mass / eta**(3.0/5.0)
    mass_1 = total_mass * (1.0 + (1.0 - 4.0 * eta)**0.5) / 2.0
    mass_2 = total_mass * (1.0 - (1.0 - 4.0 * eta)**0.5) / 2.0

    return mass_1, mass_2