# %% A implementation of the non-stationary spectral gating algorithm
"""
 * The implementation uses FIR for smoothing the noise estimate unlike the original implementation
    thus it is slower but strictly limits the transition region between data chunks
 * Further the implementation omits the job parallelization and the pytorch, limiting the dependencies
 to numpy and scipy
 * The implementation adds filtering and analytic signal options derived from the time-frequency
    representation of the signal
  
  partially based on the code from
  https://pypi.org/project/noisereduce/ licensed under the MIT license
"""
import numpy as np
import scipy.signal

def sigmoid(x, shift, mult):
    """sigmoid function
    :param x: input
    :param shift: shift
    :param mult: multiplier
    :return: sigmoid of x
    """
    return 1 / (1 + np.exp(-(x + shift) * mult))


# %%


def make_interp_freq_mask(sr, n_fft, interp_freq_mask):
    """interpolate the frequency mask and make it the same size as the stft, assuming the mask is one-sided
    :param sr: sample rate
    :param n_fft: fft size
    :param interp_freq_mask: frequency mask to interpolate, a list of tuples [(freq, amplitude), ...]
    :return: interpolated frequency mask
    """

    # make one-sided frequency range
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    x = np.array([e[0] for e in interp_freq_mask])
    y = np.array([e[1] for e in interp_freq_mask])
    return np.interp(freqs, x, y)


def make_denoising_mask(
    abs_sig_stft,
    sr,
    hop_length,
    time_constant_s,
    thresh_n_mult_nonstationary,
    sigmoid_slope_nonstationary,
):
    """make the denoising mask for the non-stationary case
    :param abs_sig_stft: absolute value of the signal stft
    :param sr: sample rate
    :param hop_length: hop length, must be smaller than n_fft
    :param time_constant_s: time constant of smoothing in seconds
    :param thresh_n_mult_nonstationary: threshold multiplier for non-stationary signal
    :param sigmoid_slope_nonstationary: slope of sigmoid for non-stationary signal
    :return: denoising mask
    """
    # fir convolve a hann window to smooth the signal
    time_constant_in_frames = int(time_constant_s * sr / hop_length)
    smothing_win = np.hanning(time_constant_in_frames * 2 + 1)
    smothing_win /= np.sum(smothing_win)
    smothing_win = np.expand_dims(smothing_win, 0)

    sig_stft_smooth = scipy.signal.convolve(abs_sig_stft, smothing_win, mode="same")

    # get the number of X above the mean the signal is
    sig_mult_above_thresh = (abs_sig_stft - sig_stft_smooth) / sig_stft_smooth

    # mask based on sigmoid
    sig_mask = sigmoid(
        sig_mult_above_thresh, -thresh_n_mult_nonstationary, sigmoid_slope_nonstationary
    )
    return sig_mask


def spectral_gating_nonstationary(
    channel,
    sr=24000,
    n_fft=1024,
    hop_length=256,
    prop_decrease=0.95,
    time_constant_s=2,
    thresh_n_mult_nonstationary=2,
    sigmoid_slope_nonstationary=10,
    analytic_signal=False,
    debug=False,
    interp_freq_mask=None,
):
    """non-stationary version of spectral gating using FIR for smoothing

    :param channel: input signal
    :param sr: sample rate
    :param n_fft: fft size
    :param hop_length: hop length, must be smaller than n_fft
    :param prop_decrease: proportion of decrease in signal
    :param time_constant_s: time constant of smoothing in seconds
    :param thresh_n_mult_nonstationary: threshold multiplier for non-stationary signal
    :param sigmoid_slope_nonstationary: slope of sigmoid for non-stationary signal
    :param analytic_signal: return analytic signal instead of real signal
    :param debug: return mask and denoised stft as well
    :param interp_freq_mask: frequency mask to interpolate, a list of tuples [(freq, amplitude), ...]
    :return: denoised signal, if debug is true, also return mask and denoised stft

    """

    f, t, sig_stft = scipy.signal.stft(
        channel,
        fs=sr,
        nfft=n_fft,
        nperseg=n_fft,
        noverlap=(n_fft - hop_length),
    )

    if interp_freq_mask is not None:
        additional_mask = make_interp_freq_mask(sr, n_fft, interp_freq_mask)

    # get abs of signal stft
    abs_sig_stft = np.abs(sig_stft)

    # make the sigmoid mask
    sig_mask = make_denoising_mask(
        abs_sig_stft,
        sr=sr,
        hop_length=hop_length,
        time_constant_s=time_constant_s,
        thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
        sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
    )

    # apply the mask and decrease the signal by a proportion
    sig_mask = sig_mask * prop_decrease + np.ones(np.shape(sig_mask)) * (
        1.0 - prop_decrease
    )

    # multiply signal with mask
    sig_stft_denoised = sig_stft * sig_mask

    if interp_freq_mask is not None:
        # apply additional mask
        sig_stft_denoised *= np.expand_dims(additional_mask, 1)

    # invert/recover the signal
    if not analytic_signal:
        # return real signal
        t, denoised_signal = scipy.signal.istft(
            sig_stft_denoised,
            fs=sr,
            nfft=n_fft,
            nperseg=n_fft,
            noverlap=(n_fft - hop_length),
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

        t, denoised_signal = scipy.signal.istft(
            analytic_signal_stft,
            fs=sr,
            nfft=n_fft,
            nperseg=n_fft,
            noverlap=(n_fft - hop_length),
            input_onesided=False,
        )

    if len(denoised_signal) > len(channel):
        # trim the signal to the original length
        denoised_signal = denoised_signal[: len(channel)]

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


# %%

    # %%
