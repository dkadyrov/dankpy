from copy import deepcopy
from dankpy import dankly, dt, file
from datetime import datetime, timedelta

import numpy as np

np.seterr(divide="ignore")

from pydub import AudioSegment
from scipy import signal
import librosa

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import noisereduce as nr
import os
import pandas as pd
import plotly.graph_objs as go
import soundfile as sf

# from plotly_resampler import FigureResampler, FigureWidgetResampler, register_plotly_resampler

valid_audio = ["wav", "flac", "mp3", "ogg", "aiff", "au"]


class Audio:
    """
    Audio class for handling audio files
    """

    def __init__(self, filepath=None, audio=None, sample_rate=None, start=None):
        if filepath:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.audio, self.sample_rate = librosa.load(filepath)
            self.duration = librosa.get_duration(path=self.filepath)
            self.length = len(self.audio)

            self.metadata = file.metadata(self.filepath)

        if audio is not None:
            self.audio = audio
            self.sample_rate = sample_rate
            self.duration = len(self.audio) / self.sample_rate
            self.length = len(self.audio)

        self.data = pd.DataFrame()

        if isinstance(start, datetime):
            self.start = start
        else:
            if start == None:
                try:
                    self.start = dt.read_datetime(self.filename[:23])
                except:
                    self.start = dt.read_datetime("00:00:00")
                # self.start = dt.read_datetime(start)
            else:
                self.start = dt.read_datetime(start)

        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)

        self.data["datetime"] = pd.date_range(
            start=self.start, end=self.end, periods=len(self.audio)
        )

        self.data["signal"] = self.audio
        self.data["time [s]"] = self.data.index/self.sample_rate
        self.data["time [ms]"] = self.data["time [s]"] * 1000

    def add_data(self, filepath):
        """
        Adds data from another audio file to this one

        Args:
            filepath (str): filepath to audio file to add
        """

        audio = Audio(filepath)

        # TODO Check sample rate of new file and convert if necessary to match
        self.audio = np.append(self.audio, audio.audio)

        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)

        self.data = pd.DataFrame()
        self.data["datetime"] = pd.date_range(
            start=self.start, end=self.end, periods=len(self.audio)
        )
        self.data["signal"] = self.audio

        if isinstance(self.metadata, dict):
            self.metadata = [self.metadata]
            self.metadata.np.append(audio.metadata)

        self.metadata.np.append(audio.metadata)

    def trim(
        self,
        start: datetime or str,
        end: datetime or str = None,
        length: float = None,
        method="datetime",
        restart=False,
    ):
        """
        Trims audio to specified start and end times or length

        Args:
            start (datetime or str): Start time of audio
            end (datetime or str, optional): End time of audio. Defaults to None.
            length (float, optional): Length of audio sample in seconds, milliseconds, or samples. Defaults to None.

        Returns:
            audio.Audio: Trimmed audio sample
        """
        sample = deepcopy(self)

        if method == "datetime":
            if not isinstance(start, datetime):
                try:
                    start = dt.read_datetime(start)
                except:
                    start = start

            if end == None:
                if isinstance(start, datetime):
                    end = start + timedelta(seconds=length)
            else:
                if not isinstance(end, datetime):
                    end = dt.read_datetime(end)

            if length == None:
                length = (end - start).total_seconds()

            sample.data = sample.data.loc[
                (sample.data.datetime >= start) & (sample.data.datetime <= end)
            ]

        if method == "samples":
            if end == None:
                end = start + length

            sample = deepcopy(self)
            sample.data = sample.data.loc[start:end]

        if method == "seconds":
            if end == None:
                end = start + length

            sample.data = sample.data.loc[
                (sample.data["time [s]"] >= start) & (sample.data["time [s]"] <= end)
            ]

        if method == "ms":
            if end == None:
                end = start + length

            sample.data = sample.data.loc[
                (sample.data["time [ms]"] >= start) & (sample.data["time [ms]"] <= end)
            ]

        if restart:
            sample.data = sample.data.reset_index(drop=True)
            sample.data["time [s]"] = (
                sample.data.index / sample.sample_rate
            )
            sample.data["time [ms]"] = sample.data["time [s]"] * 1000

        sample.start = sample.data.datetime.iloc[0]
        sample.end = sample.data.datetime.iloc[-1]
        sample.audio = sample.data.signal.values
        sample.length = len(sample.audio)
        sample.duration = len(sample.audio) / sample.sample_rate

        return sample

    def resample(self, sample_rate: int) -> None:
        """
        Resamples audio to sample rate

        Args:
            sample_rate (int): Sample rate to resample audio to
        """

        try: 
            self.audio = librosa.resample(
                self.audio, orig_sr=self.sample_rate, target_sr=sample_rate
            )
        except: 
            self.audio = [0]*int(self.duration)*int(sample_rate)

        self.sample_rate = sample_rate
        self.data = pd.DataFrame()
        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)

        self.data["datetime"] = pd.date_range(
            start=self.start, end=self.end, periods=len(self.audio)
        )
        self.data["time [s]"] = self.data.index/self.sample_rate
        self.data["time [ms]"] = self.data["time [s]"] * 1000
        self.data["signal"] = self.audio

    def spectrogram(
        self,
        window="hann",
        window_size: int = 8192,
        nfft: int = 8192,
        noverlap: int = 4096,
        nperseg: int = 8192,
        method="datetime"
    ) -> tuple:
        """
        Generates spectrogram of audio

        Args:
            window_size (int, optional): Window size. Defaults to 8192.
            nfft (int, optional): Number for FFT. Defaults to 4096.
            noverlap (int, optional): Sample overlap. Defaults to 4096.
            nperseg (int, optional): Number of Samples. Defaults to 8192.

        Returns:
            tuple: time, frequency, Pxx
        """

        time, frequency, Pxx = spectrogram(
            self.data.signal,
            self.sample_rate,
            window=window,
            window_size=window_size,
            nfft=nfft,
            noverlap=noverlap,
            nperseg=nperseg,
            start=self.start,
            end=self.end,
            method=method
        )

        return time, frequency, Pxx

    def plot_spectrogram(
        self,
        window="hann",
        window_size: int = 8192,
        nfft: int = 8192,
        noverlap: int = 4096,
        nperseg: int = 8192,
        zmin: int = None,
        zmax: int = None,
        gain: int = 0,
        showscale: bool = False,
        cmap="jet",
        aspect="auto",
        method="datetime"
    ):
        fig, ax = plt.subplots()

        time, frequency, Pxx = self.spectrogram(
            window=window,
            window_size=window_size,
            nfft=nfft,
            noverlap=noverlap,
            nperseg=nperseg,
        )

        Pxx = 10 * np.log10(Pxx) + gain

        if zmin == None:
            zmin = Pxx.min()
        if zmax == None:
            zmax = Pxx.max()

        if method == "seconds": 
            extents = [
                self.data["time [s]"].min(),
                self.data["time [s]"].max(),
                frequency.min(),
                frequency.max(),
            ]
        elif method == "ms":
            extents = [
                self.data["time [ms]"].min(),
                self.data["time [ms]"].max(),
                frequency.min(),
                frequency.max(),
            ]
        elif method == "samples": 
            extents = [0, len(self.data), frequency.min(), frequency.max()]
        else:
            extents = [self.start, self.end, frequency.min(), frequency.max()]

        axi = ax.imshow(
            Pxx,
            cmap=cmap,
            aspect=aspect,
            extent=extents,
            origin="lower",
        )
        axi.set_clim([zmin, zmax])

        ax.set_ylabel("Frequency [Hz]")

        if method == "seconds":
            ax.set_xlabel("Time [s]")
            ax.set_xlim(self.data["time [s]"].min(), self.data["time [s]"].max())
        elif method == "ms":
            ax.set_xlabel("Time [ms]")
            ax.set_xlim(self.data["time [ms]"].min(), self.data["time [ms]"].max())
        elif method == "samples":
            ax.set_xlabel("Samples")
            ax.set_xlim(0, len(self.data))
        else: 
            ax.set_xlim([self.data.datetime.iloc[0], self.data.datetime.iloc[-1]])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

        if showscale:
            cbar = fig.colorbar(
                axi, location="right", label="Amplitude [a.u.]", ticks=[zmin, zmax]
            )

        return fig, ax

    # def spectrograph(
    #     self,
    #     window_size: int = 8192,
    #     nfft: int = 4096,
    #     noverlap: int = 4096,
    #     nperseg: int = 8192,
    #     zmin: int = None,
    #     zmax: int = None,
    #     correction: int = 0,
    #     showscale: bool = False,
    # ) -> graph.Figure:
    #     """
    #     Generates spectrograph of audio

    #     Args:
    #         window_size (int, optional): Window size in samples. Defaults to 8192.
    #         nfft (int, optional): FFT number. Defaults to 4096.
    #         noverlap (int, optional): Overlap amount in samples. Defaults to 4096.
    #         nperseg (int, optional): Number of samples per segment. Defaults to 8192.
    #         zmin (int, optional): Minimum Z value for graph. Defaults to None.
    #         zmax (int, optional): Maximum Z value for graph. Defaults to None.
    #         correction (int, optional): dB correction. Defaults to 0.

    #     Returns:
    #         graph.Figure: Spectrograph
    #     """

    #     time, frequency, Pxx = self.spectrogram(
    #         window_size=window_size, nfft=nfft, noverlap=noverlap, nperseg=nperseg
    #     )

    #     fig = spectrograph(
    #         time,
    #         frequency,
    #         Pxx,
    #         colorscale="Jet",
    #         zmin=zmin,
    #         zmax=zmax,
    #         correction=correction,
    #         showscale=showscale,
    #     )

    #     return fig

    # def waveform(self) -> graph.Figure:
    #     """
    #     Generates signal graph

    #     Returns:
    #         graph.Figure: Signal graph
    #     """
    #     # register_plotly_resampler(mode='auto')

    #     fig = graph.Figure()
    #     # fig = FigureResampler(fig)
    #     fig.add_trace(
    #         go.Scatter(
    #             x=self.data.datetime,
    #             y=self.data.signal,
    #         )
    #     )

    #     fig.update_layout(
    #         yaxis_title="Signal [a.u.]",
    #         yaxis_range=[-1.5, 1.5],
    #     )

    #     return fig

    def plot_waveform(self, method: str = "datetime"):
        fig, ax = plt.subplots()

        if method == "datetime":
            ax.plot(self.data.datetime, self.data.signal)
            ax.set_xlim(self.start, self.end)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

        if method == "seconds":
            ax.plot(self.data["time [s]"], self.data.signal)
            ax.set_xlabel("Time [s]")
            ax.set_xlim(self.data["time [s]"].min(), self.data["time [s]"].max())

        if method == "ms":
            ax.plot(self.data["time [ms]"], self.data.signal)
            ax.set_xlabel("Time [ms]")
            ax.set_xlim(self.data["time [ms]"].min(), self.data["time [ms]"].max())

        if method == "samples":
            ax.plot(self.data.index, self.data.signal)
            ax.set_xlabel("Samples")
            ax.set_xlim(0, len(self.data))
            
        ax.set_ylabel("Amplitude [a.u.]")
        
        return fig, ax
    
    def plot_envelope(self, method: str = "datetime"):
        fig, ax = plt.subplots()

    

        if method == "datetime":
            ax.plot(self.data.datetime, self.envelope())
            ax.set_xlim(self.start, self.end)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))

        if method == "seconds":
            ax.plot(self.data["time [s]"], self.envelope())
            ax.set_xlabel("Time [s]")
            ax.set_xlim(self.data["time [s]"].min(), self.data["time [s]"].max())

        if method == "ms":
            ax.plot(self.data["time [ms]"], self.envelope())
            ax.set_xlabel("Time [ms]")
            ax.set_xlim(self.data["time [ms]"].min(), self.data["time [ms]"].max())

        if method == "samples":
            ax.plot(self.data.index, self.envelope())
            ax.set_xlabel("Samples")
            ax.set_xlim(0, len(self.data))
            
        ax.set_ylabel("Amplitude [a.u.]")
        
        return fig, ax
    
    def psd(self, window_size: int = 4096) -> tuple:
        """
        Generates the power spectral density of the audio

        Args:
            window_size (int, optional): Sample window size. Defaults to 4096.

        Returns:
            tuple: frequency, power
        """
        frequency, power = psd(
            self.data.signal, self.sample_rate, window_size=window_size
        )

        return frequency, power

    def plot_psd(self, window_size: int = 4096) -> tuple:
        """
        Plots the power spectral density of the audio

        Args:
            window_size (int, optional): Sample window size. Defaults to 4096.

        Returns:
            tuple: figure, axis
        """
        frequency, power = self.psd(window_size=window_size)

        fig, ax = plt.subplots()
        ax.plot(frequency, power)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [dB]")

        return fig, ax

    def lowpass_filter(self, cutoff, order=4, overwrite=False):
        """
        Lowpass filter using Butterworth filter

        Args:
            cutoff (int): Cutoff frequency
            order (int): Order of filter

        Returns:
            Audio: Filtered audio
        """

        audio = butter_lowpass_filter(self.data.signal, cutoff, self.sample_rate, order)

        if overwrite == True:
            self.data.signal = audio
            self.audio = audio
        else:
            return list(audio)

    def highpass_filter(self, cutoff, order=4, overwrite=False):
        """
        Highpass filter using Butterworth filter

        Args:
            cutoff (int): Cutoff frequency
            order (int): Order of filter

        Returns:
            Audio: Filtered audio
        """

        audio = butter_highpass_filter(self.data.signal, cutoff, self.sample_rate, order)

        if overwrite == True:
            self.data.signal = audio
            self.audio = audio
        else:
            return list(audio)

    def bandpass_filter(self, lowcut, highcut, order=4, overwrite=False):
        """
        Bandpass filter using Butterworth filter

        Args:
            lowcut (int): Low cutoff frequency
            highcut (int): High cutoff frequency
            order (int): Order of filter

        Returns:
            Audio: Filtered audio
        """

        audio = butter_bandpass_filter(
            self.data.signal, lowcut, highcut, self.sample_rate, order
        )

        if overwrite == True:
            self.data.signal = audio
            self.audio = audio
        else:
            return list(audio)  

    def reduce_noise(
        self,
        nfft=2048,
        hop_length=512,
        time_mask_smooth_ms=200,
        time_constant_s=3,
        freq_mask_smooth_hz=50,
        replace=False,
    ):
        """
        Reduces noise in audio
        """

        data = nr.reduce_noise(
            y=self.data.signal,  # audio data
            sr=self.sample_rate,  # sample rate
            prop_decrease=0.98,  # decrease noise by 98% (not an entirely binary mask)
            n_fft=nfft,  # number of FFT bins
            hop_length=hop_length,  # number of samples between FFT windows
            time_mask_smooth_ms=time_mask_smooth_ms,  # mask smoothing parameter
            time_constant_s=time_constant_s,  # time smoothing parameter
            freq_mask_smooth_hz=freq_mask_smooth_hz,  # mask smoothing parameter
        )

        if replace:
            self.data.signal = data
            self.audio = data

        return data

    def envelope(self):
        return np.abs(signal.hilbert(self.data.signal))

    def write_audio(self, filepath: str) -> None:
        """
        Writes audiofile of data with set samplerate. Omit extension, will output only wav.

        Args:
            data (list or pd.Series): data to output
            filepath (str): filepath of output
            sample_rate (int): desired file sample rate
        """
        if ".wav" not in filepath: 
            filepath = filepath + ".wav"

        sf.write(filepath, self.data.signal, self.sample_rate)





def combine_audio(list_of_files):
    """
    Combines audio files into one audio file

    Args:
        list_of_files (list): List of audio files to combine

    Returns:
        Audio: Combined audio file
    """

    combined = None

    for file in list_of_files:
        if combined == None:
            combined = Audio(file)
        else:
            combined.data.np.append(Audio(file).data)

    return combined

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = signal.butter(order, cutoff, btype="lowpass", analog=False)
    
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    
    return y


def spectrogram(
    data: list or pd.Series,
    sample_rate: int,
    window_size: int = 8192,
    window="hann",
    nfft: int = 4096,
    noverlap: int = 4096,
    nperseg: int = 8192,
    method: str = "datetime",
    start: datetime = None,
    end: datetime = None,
) -> tuple:
    """
    Generates spectrogram of audio

    Args:
        data (list or pd.Series): Data to generate spectrogram of
        sample_rate (int): Sample rate of data
        window_size (int, optional): Window size in samples. Defaults to 8192.
        nfft (int, optional): FFT number. Defaults to 4096.
        noverlap (int, optional): Overlap amount in samples. Defaults to 4096.
        nperseg (int, optional): Number of samples per segment. Defaults to 8192.
        start (datetime, optional): Start time. Defaults to None.
        end (datetime, optional): End time. Defaults to None.

    Returns:
        tuple: time, frequency, Pxx
    """
    if window == "hann":
        window = signal.windows.hann(window_size)
    elif window == "hamming":
        window = signal.windows.hamming(window_size)
    elif window == "blackman":
        window = signal.windows.blackman(window_size)
    elif window == "bartlett":
        window = signal.windows.bartlett(window_size)

    frequency, time, Pxx = signal.spectrogram(
        data,
        sample_rate,
        window=window,
        nfft=nfft,
        noverlap=noverlap,
        nperseg=nperseg,
        mode="psd",
    )

    if method == "datetime": 
        if start:
            if end == None:
                end = start + timedelta(seconds=len(data) / sample_rate)
            datetime = pd.date_range(start, end, periods=len(time))

            time = datetime
    elif method == "samples":
        time = time * sample_rate
    elif method == "ms":
        time = time * 1000
    elif method == "seconds":
        time = time

    return time, frequency, Pxx


# def spectrograph(
#     time: list or pd.Series,
#     frequency: list or pd.Series,
#     Pxx: list or pd.Series,
#     colorscale: str = "Jet",
#     zmin: int = None,
#     zmax: int = None,
#     correction: int = 0,
#     showscale: bool = False,
# ) -> graph.Figure:
#     """
#     Generates spectrograph of audio

#     Args:
#         time (list or pd.Series): Time of spectrogram
#         frequency (list or pd.Series): Frequency of spectrogram
#         Pxx (list or pd.Series): Power of spectrogram
#         colorscale (str, optional): Colorscale of graph. Defaults to "Jet".
#         zmin (int, optional): Minimum Z value for graph. Defaults to None.
#         zmax (int, optional): Maximum Z value for graph. Defaults to None.
#         correction (int, optional): dB correction. Defaults to 0.
#     Returns:
#         graph.Figure: _description_
#     """
#     fig = graph.Figure()
#     fig.add_trace(
#         go.Heatmap(
#             x=time,
#             y=frequency,
#             z=10 * np.log10(Pxx) + correction,
#             colorscale=colorscale,
#             zmin=zmin,
#             zmax=zmax,
#             zsmooth="best",
#             showscale=showscale,
#             colorbar=dict(title="Power [dBFS]", titleside="right"),
#         )
#     )
#     if showscale == True:
#         fig.update_layout(coloraxis_colorbar=dict(title="Power [dBFS]"))

#     fig.update_layout(yaxis=dict(title="Frequency [Hz]"))

#     return fig


def write_audio(data: list or pd.Series, filepath: str, sample_rate: int) -> None:
    """
    Writes audiofile of data with set samplerate. Omit extension, will output only wav.

    Args:
        data (list or pd.Series): data to output
        filepath (str): filepath of output
        sample_rate (int): desired file sample rate
    """

    sf.write(filepath + ".wav", data, sample_rate)


def mp3_to_wav(input: str, output: str, output_format: str = "wav") -> None:
    """
    Converts mp3 file to wav file.

    Args:
        file (file): filepath of input
        output (str): filepath of output
        output_format (str, optional): Output format. Defaults to "wav".
    """
    sound = AudioSegment.from_mp3(input)
    sound.export(output, format=output_format)


# def psd(x: list or pd.Series, sample_rate: int, window_size: int = 4096) -> tuple:
#     """
#     Compute the power spectral density of a signal.

#     Args:
#         x (array): signal
#         sample_rate (int): sample rate of the signal
#         sample_window (int, optional): length of the window to use for the FFT. Defaults to 4096.

#     Returns:
#         tuple: power spectral density
#     """

#     f = np.fft.rfft(x)
#     f1 = f[0 : int(window_size / 2)]
#     pf1 = 2 * np.abs(f1 * np.conj(f1)) / (sample_rate * window_size)
#     lpf1 = 10 * np.log10(pf1)
#     w = np.arange(1, window_size / 2 + 1)
#     lp = lpf1[1 : int(window_size / 2)]
#     w1 = sample_rate * w / window_size

#     return w1, lp

def psd(x: list or pd.Series, sample_rate: int, window_size: int = 4096, window: str ="blackmanharris", scaling: str ="spectrum") -> tuple:
    if window == "blackmanharris":
        window = signal.windows.blackmanharris(window_size)
    elif window == "hann":
        window = signal.windows.hann(window_size)
    elif window == "hamming":
        window = signal.windows.hamming(window_size)
    elif window == "bartlett":
        window = signal.windows.bartlett(window_size)
    elif window == "blackman":
        window = signal.windows.blackman(window_size)
    elif window == "boxcar":
        window = signal.windows.boxcar(window_size)
    
    freq, amp = signal.periodogram(x, fs=sample_rate, window=window, scaling=scaling)
    amp = 10*np.log10(amp)

    return freq, amp

# %%
def peak_hold(data, window=8*1024, sample_rate=24000):
    df = pd.DataFrame()
    samples = 0 
    while samples < sample_rate * len(data): 
        d = data[samples:samples+window]
        if len(d) < window:
            break
        # freq, amp = audio.psd(d, sample_rate=sample_rate, window_size=sample_size)
        freq, amp = signal.periodogram(d, fs=sample_rate, window=signal.windows.blackmanharris(window), scaling="spectrum")
        amp = 10*np.log10(amp)
        if "frequency" not in df.columns:
            df["frequency"] = freq
        if "amplitude" not in df.columns:
            df["amplitude"] = amp
        else: 
            df["amplitude"] = [amp[i] if amp[i] > df.amplitude[i] else df.amplitude[i] for i in range(len(amp))]
        samples += window
    return df