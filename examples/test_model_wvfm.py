import numpy as np
from cbc_injection_generator import CBCInjectionGenerator
from scipy.integrate import simpson
from pycbc.waveform.utils import ceilpow2
import pyfftw


# standard zero-noise matched-filter likelihood
class LnLikelihoodWvfm:
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

        self.signal_polarizations = (
            self.injection.injection_generator.frequency_domain_strain(
                self.injection.injection_parameters
            )
        )
        self.signal_response = self.ifo.get_detector_response(
            self.signal_polarizations, self.injection.injection_parameters
        )[self.frequency_mask]

        self.f = self.ifo.frequency_array[self.frequency_mask]

        self.signal_ovlp = overlap(
            self.signal_response, self.signal_response, self.f, self.psd
        )

        self.ntemps = self.injection.recovery_options["ntemps"]
        self.nsamples = self.injection.recovery_options["nsamples"]

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
            dist = params[4]
            dt = params[5]
            phase = params[6]
        except Exception:
            return -np.inf

        template_params = self.injection.injection_parameters.copy()
        template_params["mass_1"] = mass_1
        template_params["mass_2"] = mass_2
        template_params["chi_1"] = spin1z
        template_params["chi_2"] = spin2z
        template_params["luminosity_distance"] = dist

        template_polarizations = (
            self.recovery_waveform_generator.frequency_domain_strain(template_params)
        )
        template_response = self.ifo.get_detector_response(
            template_polarizations, template_params
        )[self.frequency_mask]

        rotation = np.exp(-1j * (2.0 * np.pi * self.f * dt + phase))

        result = get_overlap(
            template_response * rotation,
            self.signal_response,
            self.signal_ovlp,
            self.f,
            self.psd,
        )

        # result = match(template_response, self.signal_response, self.psd, 2**18, True)

        # result = 100 * result ** 4

        if np.isfinite(result):
            return result
        else:
            return -np.inf


# uniform prior
class LnPriorUniformWvfm:
    def __init__(self, mins, maxes):
        """
        mins: vector of minima on uniform prior
        maxes: vector of maxima on uniform prior
        """
        self.rng = np.random.default_rng()
        self.mins = mins
        self.maxes = maxes

    def __call__(self, params):
        if params[-1] < self.mins[-1] or params[-1] > self.maxes[-1]:
            params[-1] = np.mod(params[-1], np.pi)

        if np.any(params < self.mins) or np.any(params > self.maxes):
            return -np.inf
        else:
            return 0

    def initial_sample(self):
        return self.rng.uniform(self.mins, self.maxes)


def overlap_dot(h1_in, h2_in, psd):
    h1 = np.array(h1_in)
    h2 = np.array(h2_in)

    return 4.0 * (h1 * np.conj(h2)) / psd


def overlap(h1_in, h2_in, f, psd):

    integrand = overlap_dot(h1_in, h2_in, psd)
    return np.real(simpson(integrand, x=f))


def get_overlap(template, signal, signal_ovlp, f, psd):
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

    overlap_sig_temp = overlap(template, signal, f, psd)
    overlap_temp_temp = overlap(template, template, f, psd)

    return overlap_sig_temp - 0.5 * (signal_ovlp + overlap_temp_temp)


def mass1_mass2_from_chirpmass_eta(chirp_mass, eta):
    """Function to produce values of mass 1 and mass 2 from the
    chirp mass and symmetric mass ratio

    Args:
        chirp_mass (float): chirp mass
        eta (float): symmetric mass ratio

    Returns:
        float, float: returns mass1 and mass2, mass1 >= mass2
    """
    total_mass = chirp_mass / eta ** (3.0 / 5.0)
    mass_1 = total_mass * (1.0 + (1.0 - 4.0 * eta) ** 0.5) / 2.0
    mass_2 = total_mass * (1.0 - (1.0 - 4.0 * eta) ** 0.5) / 2.0

    return mass_1, mass_2


def pad_to_pow_2(arr, zpf, zpf_as_target_length=False):
    """Pad an array pseudo-evenly on each side

    From https://github.com/Cyberface/phenom/blob/1822f55b2631fad913301b5e7cf0c5b52e31f828/phenom/utils/utils.py

    Parameters
    ----------
    arr : np.Array
        Array to pad
    zpf : integer
        Zero padding factor. If 0 then returns original length of
        arr, rounded up to nearest power of 2.

    Returns
    -------
    np.Array
        The input array zero padded
    """
    n = len(arr)

    if zpf_as_target_length and zpf > n:
        initial_pad_left = int(np.floor((zpf - n) / 2))
        initial_pad_right = int(np.floor((zpf - n) / 2))
    else:
        initial_pad_left = zpf * n
        initial_pad_right = zpf * n
    initial_finial_length = initial_pad_left + n + initial_pad_right

    next_power_2 = ceilpow2(initial_finial_length)

    to_add = np.absolute(next_power_2 - initial_finial_length)

    # If even, add to both sides equally
    if (to_add % 2) == 0:
        to_add_left = int(to_add / 2)
        to_add_right = int(to_add / 2)
    # If odd, add extra one to left side
    else:
        to_add_left = int(to_add / 2 + 1)
        to_add_right = int(to_add / 2)

    left = np.zeros(initial_pad_left + to_add_left)
    right = np.zeros(initial_pad_right + to_add_right)

    return np.concatenate([left, arr, right])


def match(h_1, h_2, psd, zpf=3, zpf_as_target_length=False):
    """Calculate the match between two waveforms

    Parameters
    ----------
    h_1, h_2: np.Array
        Two waveforms to calculate the match for.
    psd: np.Array
        PSD for the match
    zpf : integer
        Zero padding factor. If 0 then returns original length of
        arr, rounded up to nearest power of 2. Increase above zero
        effectively improves the numerical time shift optimisation
        performed in the match

    Returns
    -------
    float
        Match between h_1 and h_2 weighted by psd
    """
    # Calculate waveform amplitudes
    A_1 = np.abs(h_1)
    A_2 = np.abs(h_2)

    # Calculate waveform norms wrt psd
    norm_1 = np.dot(A_1, A_1 / psd)
    norm_2 = np.dot(A_2, A_2 / psd)

    # Calculate match integrand
    integrand = h_1 * h_2.conj() / psd
    integrand_zero_padded = pad_to_pow_2(integrand, zpf, zpf_as_target_length)

    # Calculate complex SNR sequence to enable time shift optimisation
    complex_snr = np.asarray(pyfftw.interfaces.numpy_fft.fft(integrand_zero_padded))

    return np.max(np.abs(complex_snr)) / np.sqrt(norm_1 * norm_2)
