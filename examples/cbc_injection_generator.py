import bilby
import numpy as np
import lalsimulation as sim
import h5py as h5
from pycbc.waveform.utils import ceilpow2
import pycbc.pnutils as utils
from pycbc.conversions import mchirp_from_mass1_mass2
import configparser

class CBCInjectionGenerator:
    def __init__(self, config_path):
        """Generate an injection class for storing useful things

        Parameters
        ----------
        config_path: str
            file path to appropriate config file
        """

        self.config_path = config_path

        # initialize injection parameters
        self.parameter_dict = _get_parameters_from_config_file(config_path)

        self.injection_options = self.parameter_dict['injection']
        self.cbc_parameters = self.parameter_dict['parameters']
        self.recovery_options = self.parameter_dict['recovery']

        # Calculate candidate sampling frequencies of hyperprior corners
        self.sampling_frequency = get_sampling_frequency(self.cbc_parameters)
        # Calculate candidate durations of hyperprior corners
        self.duration = get_duration(self.cbc_parameters, self.injection_options['f_low'])

        self.initialize_detector_network(self.injection_options['detector'])
        self.generate_injection()
        self.initialize_waveform_generator()

    def initialize_detector_network(self, interferometer):
        """Method to initialize the bilby IFO network

        Parameters
        ----------
        interferometer: str, list
            name of interferometer file to be used, or list of interferometer files
            to build detector network
        """

        if isinstance(interferometer, list):
            ifo_list = [
                bilby.gw.detector.load_interferometer((inter)) for inter in interferometer]
        else:
            ifo_list = [
                bilby.gw.detector.load_interferometer((interferometer))]
        
        self.ifo_list = ifo_list
        self.ifos = bilby.gw.detector.InterferometerList(ifo_list)

    def generate_injection(self):
        """Generate signal injection in interferometer

        """
        self.injection_arguments = dict(
            reference_frequency=self.injection_options['f_ref'],
            minimum_frequency=self.injection_options['f_low'], 
        )
        if self.injection_options['approximant'] == "NR_hdf5":
            self.injection_arguments["waveform_approximant"] = self.injection_options['approximant']
            self.injection_arguments["numerical_relativity_file"] = self.injection_options['numrel_path']
        elif self.injection_options['approximant'] == "NRHybSur3dq8_22":
            self.injection_arguments["waveform_approximant"] = "NRHybSur3dq8"
            self.injection_arguments["mode_array"] = [[2,2], [2,-2]]
        else:
            self.injection_arguments["waveform_approximant"] = self.injection_options['approximant']

        self.injection_generator = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=self.injection_arguments)
        self.injection_parameters = self.cbc_parameters.copy()
        self.injection_parameters.update({
            'luminosity_distance': 64.0
        })

        self.ifos.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=self.injection_parameters['geocent_time'] - (self.duration - 2.0)
        )
        self.ifos.inject_signal(
            waveform_generator=self.injection_generator, parameters=self.injection_parameters)
        
        # Calculate correct luminosity distance to achieve targeted network SNR
        candidate_network_snr = \
            np.sqrt(np.sum([ifo.meta_data['optimal_SNR']**2 for ifo in self.ifos]))
        self.injection_parameters['luminosity_distance'] = \
            candidate_network_snr \
            * self.injection_parameters['luminosity_distance'] \
            / self.injection_options['snr']

        # Set up network and inject waveform at target network snr
        self.ifos.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=self.injection_parameters['geocent_time'] - (self.duration - 2.0)
        )
        self.ifos.inject_signal(
            waveform_generator=self.injection_generator, parameters=self.injection_parameters)
        
    def initialize_waveform_generator(self):
        """Generate recovery waveform generator
        
        """
        self.waveform_generator = get_waveform_generator(
            self.recovery_options['approximant'], 
            self.sampling_frequency, self.duration,
            self.recovery_options['f_low'], self.recovery_options['f_ref'])


def _get_uniform_chirp_mass_q_prior(mass_1, mass_2):
    """Return uniform priors in mass ratio and chirp mass based
    on values of the component masses

    Parameters
    ----------
    mass_1: float
        primary mass
    mass_2: float
        secondary mass
    """
    q = mass_1 / mass_2
    chirp_mass = mchirp_from_mass1_mass2(mass_1, mass_2)
    q_prior = np.r_[1/(q*1.15), np.min([1/(q*0.85), 1.0])]
    chirp_mass_prior = [chirp_mass*0.85, chirp_mass*1.15]
    return q_prior, chirp_mass_prior


def _get_parameters_from_config_file(config_file):
    """Return a dictionary of various injection parameters

    Parameters
    ----------
    config_file: str
        path to config file
    return_parser: bool, optional
        return the full configparser
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    config_dict = {k:dict(config.items(k)) for k in config.sections()}

    injection_parameters = config_dict['injection'].copy()
    cbc_parameters = config_dict['parameters'].copy()
    recovery_parameters = config_dict['recovery'].copy()

    # convert floats appropriately
    injection_parameters.update({k:float(injection_parameters[k]) for k in ['f_low', 'f_ref', 'snr']})
    cbc_parameters = {a:float(b) for a,b in cbc_parameters.items()}
    recovery_parameters.update({k:float(injection_parameters[k]) for k in ['f_low', 'f_ref']})

    return {
        'injection':injection_parameters,
        'parameters':cbc_parameters,
        'recovery':recovery_parameters
    }


def _get_hyper_prior(parameter_dict):
    """Return a hyper prior based on NR parameters

    Parameters
    ----------
    parameter_dict: dict
        dictionary of CBC source parameters
    """
    # convert parameters
    # Create hyperprior to choose appropriate PE sample rate and seglen

    mass_1 = parameter_dict['mass_1']
    mass_2 = parameter_dict['mass_2']
    chi_1 = parameter_dict['chi_1']
    chi_2 = parameter_dict['chi_2']

    chi_1_prior = np.r_[-0.99, 0.99]
    chi_2_prior = np.r_[-0.99, 0.99]
    q_prior, chirp_mass_prior = _get_uniform_chirp_mass_q_prior(mass_1, mass_2)
    chirp_mass_q_prior = np.array(np.meshgrid(chirp_mass_prior, q_prior))
    chirp_mass_q_prior = chirp_mass_q_prior.T.reshape(-1, 2)
    mass_1_list, mass_2_list = utils.mchirp_q_to_mass1_mass2(
        *chirp_mass_q_prior.T)
    hyper_prior = np.array([
        [_mass_1, _mass_2, _chi_1, _chi_2]
        for (_mass_1, _mass_2) in zip(mass_1_list, mass_2_list)
        for _chi_1 in chi_1_prior
        for _chi_2 in chi_2_prior])
    hyper_prior = np.append(
        hyper_prior,
        np.array([[mass_1, mass_2, chi_1, chi_2]]),
        axis=0)
    return hyper_prior


def get_sampling_frequency(parameter_dict):
    """Return the maximum sampling_frequency based on NR parameters

    Parameters
    ----------
    parameter_dict: dict
        dictionary of CBC source parameters
    """
    hyper_prior = _get_hyper_prior(parameter_dict)
    # Calculate candidate sampling frequencies of hyperprior corners
    sampling_frequencies = utils.frequency_cutoff_from_name(
        'SEOBNRv4RD', *hyper_prior.T)
    return ceilpow2(sampling_frequencies.max())*2


def get_duration(parameter_dict, f_low):
    """Return the maximum duration based on NR parameters

    Parameters
    ----------
    parameter_dict: dict
        dictionary of CBC source parameters
    f_low: float
        starting frequency to use for duration estimation
    """
    hyper_prior = _get_hyper_prior(parameter_dict)
    durations = utils.get_imr_duration(
        *hyper_prior.T, f_low=f_low,
        approximant='IMRPhenomD')
    return ceilpow2(durations.max())


def get_waveform_generator(approximant, sampling_frequency, duration, f_low, f_ref):
    """Return the waveform generator for PE and match analysis

    Parameters
    ----------
    approximant: str
        string containing the waveform approximant
    sampling_frequency: float
        sample rate of the signal
    duration: float
        duration of the signal
    f_low: float
        frequency to start waveform generation
    f_ref: float
        frequency at which the LAL source frame is specified
    """
    # Prepare model waveform generator
    waveform_arguments = dict(
        waveform_approximant=approximant,
        reference_frequency=f_ref, minimum_frequency=f_low)
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments)
    return waveform_generator
