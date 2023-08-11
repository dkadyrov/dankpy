import math
import numpy as np
import pandas as pd
from dankpy.functions import sigmoid
from numpy import pi, polymul
from scipy.signal import medfilt, bilinear, convolve, istft, stft

## 1 atm in Pa
ps0 = 1.01325e5

def A_weighting(fs):
    """Design of an A-weighting filter.
    b, a = A_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).
    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.
    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = polymul(
        [1, 4 * pi * f4, (2 * pi * f4) ** 2], [1, 4 * pi * f1, (2 * pi * f1) ** 2]
    )
    DENs = polymul(polymul(DENs, [1, 2 * pi * f3]), [1, 2 * pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, fs)


def absorption(f, t=20, rh=60, ps=ps0):
    """In dB/m

    f: frequency in Hz
    t: temperature in °C
    rh: relative humidity in %
    ps: atmospheric pressure in Pa

    From http://en.wikibooks.org/wiki/Engineering_Acoustics/Outdoor_Sound_Propagation
    """
    T = t + 273.15
    T0 = 293.15
    T01 = 273.16

    Csat = -6.8346 * math.pow(T01 / T, 1.261) + 4.6151
    rhosat = math.pow(10, Csat)
    H = rhosat * rh * ps0 / ps

    frn = (
        (ps / ps0)
        * math.pow(T0 / T, 0.5)
        * (9 + 280 * H * math.exp(-4.17 * (math.pow(T0 / T, 1 / 3.0) - 1)))
    )

    fro = (ps / ps0) * (24.0 + 4.04e4 * H * (0.02 + H) / (0.391 + H))

    alpha = (
        f
        * f
        * (
            1.84e-11 / (math.pow(T0 / T, 0.5) * ps / ps0)
            + math.pow(T / T0, -2.5)
            * (
                0.10680 * math.exp(-3352 / T) * frn / (f * f + frn * frn)
                + 0.01278 * math.exp(-2239.1 / T) * fro / (f * f + fro * fro)
            )
        )
    )

    return 20 * alpha / math.log(10)


def a_weighted(f: float) -> float:
    """
    Converts frequency into A-weighted decibels

    Args:
        f (float): Frequency in Hz

    Returns:
        float: A-weighted decibels
    """

    return (
        1.2588966
        * (12200**2.0 * f**4)
        / (
            (f**2 + 20.6**2)
            * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
            * (f**2 + 12200**2)
        )
    )


def decibel(x: list, u: str = "power", r: float = 1) -> list:
    """
    Converts a list of values into decibels. If the signal type is voltage, the resistance must be specified.

    Args:
        x (list):
        u (str, optional): Signal Type. specifies the signal type represented by the elements of x as either 'voltage' or 'power'. Defaults to power.
        r (float, optional): The resistance for voltage measurements. Defaults to 1.

    Returns:
        list: Values in decibels
    """

    if u not in ["power", "voltage"]:
        raise "u must be power or voltage"
    elif u == "voltage":
        x = abs(x) ** 2 / r
    elif u == "power" and r is None:
        x = abs(x) ** 2

    return (10 * np.log10(x) + 300) - 300


def integrate_PSD_db(
    frequency: list or pd.Series, signal: list or pd.Series, f_low: float, f_high: float
) -> list:
    """
    Integrates a power spectral density in decibels

    Args:
        frequency (list or pd.Series): Frequency in Hz
        signal (list or pd.Series): Power spectral density in decibels
        f_low (float): Low frequency cutoff
        f_high (float): High frequency cutoff

    Returns:
        list: Integrated power spectral density in decibels
    """

    data = pd.DataFrame({"frequency": frequency, "signal": signal})

    filt = data[(data.frequency <= f_high) & (data.frequency >= f_low)]

    return 10 * np.log10(sum(10 ** (filt.signal / 10)) * data.frequency.iloc[1])


# def integrate_PSD2D_db(data, f_low, f_high):
#     filt = data[(data.frequency <= f_high) & (data.frequency >= f_low)]
#     return 10 * np.log10(sum(10 ** (filt.signal/10)) * data.
#     frequency.iloc[1])


def integrate_a_weighted(f: list, x: list) -> list:
    """
    Integrates an a-weighted power spectral density in decibels

    Args:
        f (list): Frequency in Hz
        x (list): Power spectral density in decibels

    Returns:
        list: Integrated a-weighted power spectral density in decibels
    """
    return decibel(sum(10 ** (x / 10) * (a_weighted(f) ** 2) * f[1]), "power")


def integrate_a_weighted_medfilt(f: list, x: list) -> list:
    """
    Integrates an a-weighted power spectral density in decibels with a median filter

    Args:
        f (list): Frequency in Hz
        x (list): Power spectral density in decibels

    Returns:
        list: Integrated a-weighted power spectral density in decibels with a median filter
    """

    return decibel(
        sum(10 ** (medfilt(x, 49) / 10) * (a_weighted(f) ** 2) * f[1]), "power"
    )


def atmospheric_attenuation(T: float, P: float, h: float, dist: float, f: list) -> list:
    """
    Calculates the atmospheric attenuation for a given temperature, pressure, altitude, distance, and frequency

    Args:
        T (float): Temperature in K
        P (float): Pressure in Pa
        h (float): Humidity in %
        dist (float): Distance in m
        f (list): Frequency in Hz

    Returns:
        list: Atmospheric attenuation in dB/m
    """

    T = T + 273.15  # temp input in K
    To1 = 273.15  # triple point in K
    To = 293.15  # ref temp in K

    Ps = P / 29.9212598  # static pressure in atm
    Pso = 1  # reference static pressure

    F = f / Ps  # frequency per atm

    # calculate saturation pressure
    Psat = 10 ** (
        10.79586 * (1 - (To1 / T))
        - 5.02808 * np.log10(T / To1)
        + 1.50474e-4 * (1 - 10 ** (-8.29692 * ((T / To1) - 1)))
        - 4.2873e-4 * (1 - 10 ** (-4.76955 * ((To1 / T) - 1)))
        - 2.2195983
    )

    h = h * Psat / Ps  # calculate the absolute humidity

    # Scaled relaxation frequency for Nitrogen
    FrN = (To / T) ** (1 / 2) * (
        9 + 280 * h * np.exp(-4.17 * ((To / T) ** (1 / 3) - 1))
    )

    # scaled relaxation frequency for Oxygen
    FrO = 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)

    # attenuation coefficient in nepers/m
    alpha = (
        Ps
        * F**2
        * (
            1.84e-11 * (T / To) ** (1 / 2)
            + (T / To) ** (-5 / 2)
            * (
                1.275e-2 * np.exp(-2239.1 / T) / (FrO + F**2 / FrO)
                + 1.068e-1 * np.exp(-3352 / T) / (FrN + F**2 / FrN)
            )
        )
    )

    a = 10 * np.log10(np.exp(2 * alpha)) * dist

    return a


def ground_effect(c, hr, hs, hrange, f, sigmae):
    range1 = np.hypot(hr - hs, hrange)
    range2 = np.hypot(hr + hs, hrange)

    l = c / f

    theta = np.arctan2(hrange, hr + hs)

    Z = (
        1
        + 0.0511 * (1 * f / sigmae) ** (-0.75)
        + 1 * 1j * 0.0768 * (1 * f / sigmae) ** (-0.73)
    )
    R_p = (Z * np.sin(theta) - 1) / (Z * np.sin(theta) + 1)
    F = 0.1
    Q = R_p + (1 - R_p) * F
    C = 1
    Rf = -np.sqrt(C) * abs(Q) + np.sqrt(1 + abs(Q) ** 2 + 2.0 * abs(Q) * C)
    pp = Rf * np.exp(1j * 2 * np.pi * range1 / l) + np.sqrt(C) * Q * np.exp(
        1j * 2 * np.pi * range2 / l
    )

    amp = decibel(abs(pp))
    amp = amp.replace(np.nan, 0)

    return amp


def interpolate_frequency_mask(
    sample_rate: int, nfft: int, interp_freq_mask: list
) -> np.array:
    """
    Interpolate the frequency mask and make it the same size as the stft, assuming the mask is one-sided

    Args:
        sample_rate (int): sample rate
        nfft (int): fft size
        interp_freq_mask (list): frequency mask to interpolate, a list of tuples [(freq, amplitude), ...]

    Returns:
        np.array: interpolated frequency mask
    """

    # make one-sided frequency range
    freqs = np.linspace(0, sample_rate / 2, nfft // 2 + 1)
    x = np.array([e[0] for e in interp_freq_mask])
    y = np.array([e[1] for e in interp_freq_mask])
    return np.interp(freqs, x, y)


def denoising_mask(
    absolute_stft: int,
    sample_rate: int,
    hop_length: int,
    time_constant: int,
    thresh_n_mult_nonstationary: int,
    sigmoid_slope_nonstationary: int,
) -> int:
    """
    Denoising mask for the non-stationary case

    Args:
        absolute_stft (int):  absolute value of the signal stft
        sample_rate (int): sample rate
        hop_length (int): hop length, must be smaller than nfft
        time_constant (int): time constant of smoothing in seconds
        thresh_n_mult_nonstationary (int): threshold multiplier for non-stationary signal
        sigmoid_slope_nonstationary (int): slope of sigmoid for non-stationary signal

    Returns:
        int: denoising mask
    """

    # fir convolve a hann window to smooth the signal
    time_constant_in_frames = int(time_constant * sample_rate / hop_length)
    smothing_win = np.hanning(time_constant_in_frames * 2 + 1)
    smothing_win /= np.sum(smothing_win)
    smothing_win = np.expand_dims(smothing_win, 0)

    sig_stft_smooth = convolve(absolute_stft, smothing_win, mode="same")

    # get the number of X above the mean the signal is
    sig_mult_above_thresh = (absolute_stft - sig_stft_smooth) / sig_stft_smooth

    # mask based on sigmoid
    sig_mask = sigmoid(
        sig_mult_above_thresh, -thresh_n_mult_nonstationary, sigmoid_slope_nonstationary
    )
    return sig_mask


def spectral_gating_nonstationary(
    signal: np.array,
    sample_rate: int,
    nfft: int = 1024,
    hop_length: int = 256,
    prop_decrease: int = 0.95,
    time_constant: int = 2,
    thresh_n_mult_nonstationary: int = 2,
    sigmoid_slope_nonstationary: int = 10,
    analytic_signal: bool = False,
    debug: bool = False,
    interp_freq_mask: int = None,
) -> np.array:
    """
    Non-stationary version of spectral gating using FIR for smoothing

    Args:
        signal (np.array): input signal
        sample_rate (int): sample rate
        nfft (int, optional): fft size. Defaults to 1024.
        hop_length (int, optional): _description_. Defaults to 256.
        prop_decrease (int, optional): _description_. Defaults to 0.95.
        time_constant (int, optional): _description_. Defaults to 2.
        threshold_n_multiplier_nonstationary (int, optional): threshold multiplier for non-stationary signal. Defaults to 2.
        sigmoid_slope_nonstationary (int, optional): slope of sigmoid for non-stationary signal. Defaults to 10.
        analytic_signal (bool, optional): return analytic signal instead of real signal. Defaults to False.
        debug (bool, optional): return mask and denoised stft as well. Defaults to False.
        interp_freq_mask (int, optional): frequency mask to interpolate, a list of tuples [(freq, amplitude), ...]. Defaults to None.

    Returns:
        np.array: denoised signal, if debug is true, also return mask and denoised stft
    """

    f, t, sig_stft = stft(
        signal,
        fs=sample_rate,
        nfft=nfft,
        nperseg=nfft,
        noverlap=(nfft - hop_length),
    )

    if interp_freq_mask is not None:
        additional_mask = interpolate_frequency_mask(
            sample_rate, nfft, interp_freq_mask
        )

    # get abs of signal stft
    abs_sig_stft = np.abs(sig_stft)

    # make the sigmoid mask
    sig_mask = denoising_mask(
        abs_sig_stft,
        sample_rate=sample_rate,
        hop_length=hop_length,
        time_constant=time_constant,
        thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
        sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
    )

    # apply the mask and decrease the signal by a proportion
    sig_mask = sig_mask * prop_decrease + np.ones(np.shape(sig_mask)) * (
        1.0 - prop_decrease
    )

    # multiply signal with mask
    sig_stft_denoised = sig_stft * sig_mask

    if interpolate_frequency_mask is not None:
        # apply additional mask
        sig_stft_denoised *= np.expand_dims(additional_mask, 1)

    # invert/recover the signal
    if not analytic_signal:
        # return real signal
        t, denoised_signal = istft(
            sig_stft_denoised,
            fs=sample_rate,
            nfft=nfft,
            nperseg=nfft,
            noverlap=(nfft - hop_length),
            input_onesided=True,
        )
    else:
        # return analytic signal instead

        # the analytic signal stft is the original stft with the negative frequencies zeroed out
        # so we pad the original stft with zeros and then take the ifft, using it as the two-sided stft
        analytic_signal_stft = np.zeros(
            (sig_stft_denoised.shape[0] * 2 - 2, sig_stft_denoised.shape[1]),
            dtype=sig_stft_denoised.dtype,
        )
        analytic_signal_stft[: sig_stft_denoised.shape[0], :] = sig_stft_denoised

        t, denoised_signal = istft(
            analytic_signal_stft,
            fs=sample_rate,
            nfft=nfft,
            nperseg=nfft,
            noverlap=(nfft - hop_length),
            input_onesided=False,
        )

    if len(denoised_signal) > len(signal):
        # trim the signal to the original length
        denoised_signal = denoised_signal[: len(signal)]

    if debug:
        return denoised_signal, sig_mask, sig_stft_denoised
    else:
        return denoised_signal


class AccumulateChunk:
    def __init__(self, chunk_size, padding):
        self.chunk_size = chunk_size
        self.padding = padding * 2
        self.data = np.zeros((0, 0))

    def new_data(self, new_data):
        if len(new_data.shape) == 1:
            new_data = np.expand_dims(new_data, 1)
        if self.data.shape[1] != new_data.shape[1]:
            self.data = new_data
        else:
            self.data = np.concatenate((self.data, new_data))
        pass

    def available(self):
        return len(self.data) > self.chunk_size + self.padding * 2

    def get_chunk(self):
        if not self.available():
            raise ValueError("Not enough data to get chunk")
        padded_chunk = self.data[: self.chunk_size + self.padding * 2, :]
        self.data = self.data[self.chunk_size :, :]
        
        return padded_chunk

class OnlineDenoiser:
    def __init__(self, chunk_duration=2, time_constant=2, sr=24000) -> None:
        self.time_constant = time_constant
        self.chunk_duration = chunk_duration
        self.sr = sr
        self._change_params()

    def _change_params(self, time_constant=None, sr=None, chunk_duration=None):
        if time_constant is not None:
            self.time_constant = time_constant
        if sr is not None:
            self.sr = sr
        if chunk_duration is not None:
            self.chunk_duration = chunk_duration
        self.ac = AccumulateChunk(
            self.sr * self.chunk_duration, padding=self.sr * self.time_constant
        )
        self.out_ac = AccumulateChunk(self.sr * self.chunk_duration, padding=0)

    def new_data(self, new_data):
        self.ac.new_data(new_data)
        if self.ac.available():
            chunk = self.ac.get_chunk()

            # iterate over channels
            out_chunk = np.zeros_like(chunk)
            for i in range(chunk.shape[1]):
                out_chunk[:, i] = spectral_gating_nonstationary(
                    chunk[:, i], self.sr, self.time_constant
                )

            # remove padding
            valid_out_chunk = out_chunk[
                self.sr * self.time_constant : -self.sr * self.time_constant, :
            ]
            self.out_ac.new_data(valid_out_chunk)

    def available(self):
        return self.out_ac.available()

    def get_chunk(self):
        return self.out_ac.get_chunk()

    def get_everything(self):
        out = self.out_ac.data
        self.out_ac.data = np.zeros((0, 0))

        return out