from asyncio import start_server
from dankpy import graph, dt, file
import plotly.graph_objs as go
import pandas as pd
from scipy import signal
from numpy import log10, fft, conj, abs, arange, append
from datetime import datetime, timedelta
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import soundfile as sf
import librosa
import librosa.display
from pydub import AudioSegment
from copy import deepcopy
import os

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
            self.length = self.duration

            self.metadata = file.metadata(self.filepath)

        if audio is not None:
            self.audio = audio
            self.sample_rate = sample_rate
            self.duration = len(self.audio) / self.sample_rate
            self.length = self.duration

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


    def add_data(self, filepath):
        """
        Adds data from another audio file to this one

        Args:
            filepath (str): filepath to audio file to add
        """

        audio = Audio(filepath)
        
        # TODO Check sample rate of new file and convert if necessary to match
        self.audio = append(self.audio, audio.audio)
        
        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)
        
        self.data = pd.DataFrame()
        self.data["datetime"] = pd.date_range(
            start=self.start, end=self.end, periods=len(self.audio)
        )
        self.data["signal"] = self.audio
    
        if isinstance(self.metadata, dict):
            self.metadata = [self.metadata]
            self.metadata.append(audio.metadata)

        self.metadata.append(audio.metadata)

    def trim(
        self, start: datetime or str, end: datetime or str = None, length: float = None
    ):
        """
        Trims audio to specified start and end times or length

        Args:
            start (datetime or str): Start time of audio
            end (datetime or str, optional): End time of audio. Defaults to None.
            length (float, optional): Length of audio sample in seconds. Defaults to None.

        Returns:
            audio.Audio: Trimmed audio sample
        """

        if not isinstance(start, datetime):
            start = dt.read_datetime(start)

        if end == None:
            end = start + timedelta(seconds=length)
        else:
            if not isinstance(end, datetime):
                end = dt.read_datetime(end)

        if length == None:
            length = (end - start).total_seconds()

        # number = round((start - start).total_seconds()/self.sample_rate)
        # number_end = round((end - start).total_seconds()/self.sample_rate)

        sample = deepcopy(self)
        sample.data = sample.data.loc[
            (sample.data.datetime >= start) & (sample.data.datetime <= end)
        ]
        sample.start = start
        sample.end = end
        sample.audio = sample.data.signal.values
        sample.length = len(sample.audio) / sample.sample_rate
        sample.duration = sample.length

        return sample

    def resample(self, sample_rate: int) -> None:
        """
        Resamples audio to sample rate

        Args:
            sample_rate (int): Sample rate to resample audio to
        """

        self.audio = librosa.resample(
            self.audio, orig_sr=self.sample_rate, target_sr=sample_rate
        )
        self.sample_rate = sample_rate
        self.data = pd.DataFrame()
        self.end = self.start + timedelta(seconds=len(self.audio) / self.sample_rate)

        self.data["datetime"] = pd.date_range(
            start=self.start, end=self.end, periods=len(self.audio)
        )
        self.data["signal"] = self.audio

    def spectrogram(
        self,
        window_size: int = 8192,
        nfft: int = 4096,
        noverlap: int = 4096,
        nperseg: int = 8192,
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
            window_size=window_size,
            nfft=nfft,
            noverlap=noverlap,
            nperseg=nperseg,
            start=self.start,
            end=self.end,
        )

        return time, frequency, Pxx

    def spectrograph(
        self,
        window_size: int = 8192,
        nfft: int = 4096,
        noverlap: int = 4096,
        nperseg: int = 8192,
        zmin: int = None,
        zmax: int = None,
        correction: int = 0,
    ) -> graph.Figure:
        """
        Generates spectrograph of audio

        Args:
            window_size (int, optional): Window size in samples. Defaults to 8192.
            nfft (int, optional): FFT number. Defaults to 4096.
            noverlap (int, optional): Overlap amount in samples. Defaults to 4096.
            nperseg (int, optional): Number of samples per segment. Defaults to 8192.
            zmin (int, optional): Minimum Z value for graph. Defaults to None.
            zmax (int, optional): Maximum Z value for graph. Defaults to None.
            correction (int, optional): dB correction. Defaults to 0.

        Returns:
            graph.Figure: Spectrograph
        """

        time, frequency, Pxx = self.spectrogram(
            window_size=window_size, nfft=nfft, noverlap=noverlap, nperseg=nperseg
        )

        fig = spectrograph(
            time,
            frequency,
            Pxx,
            colorscale="Jet",
            zmin=zmin,
            zmax=zmax,
            correction=correction,
        )

        return fig

    def signal(self) -> graph.Figure:
        """
        Generates signal graph

        Returns:
            graph.Figure: Signal graph
        """

        fig = graph.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.data.datetime,
                y=self.data.signal,
            )
        )

        fig.update_layout(
            yaxis_title="Signal [a.u.]",
            yaxis_range=[-1.5, 1.5],
        )

        return fig

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
            combined.data.append(Audio(file).data)

    return combined

def spectrogram(
    data: list or pd.Series,
    sample_rate: int,
    window_size: int = 8192,
    nfft: int = 4096,
    noverlap: int = 4096,
    nperseg: int = 8192,
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

    window = signal.windows.hann(window_size)

    frequency, time, Pxx = signal.spectrogram(
        data, sample_rate, window=window, mode="psd", noverlap=noverlap
    )

    if start:
        if end == None:
            end = start + timedelta(seconds=len(data) / sample_rate)
        datetime = pd.date_range(start, end, periods=len(time))

        time = datetime

    return time, frequency, Pxx


def spectrograph(
    time: list or pd.Series,
    frequency: list or pd.Series,
    Pxx: list or pd.Series,
    colorscale: str = "Jet",
    zmin: int = None,
    zmax: int = None,
    correction: int = 0,
) -> graph.Figure:
    """
    Generates spectrograph of audio

    Args:
        time (list or pd.Series): Time of spectrogram
        frequency (list or pd.Series): Frequency of spectrogram
        Pxx (list or pd.Series): Power of spectrogram
        colorscale (str, optional): Colorscale of graph. Defaults to "Jet".
        zmin (int, optional): Minimum Z value for graph. Defaults to None.
        zmax (int, optional): Maximum Z value for graph. Defaults to None.
        correction (int, optional): dB correction. Defaults to 0.
    Returns:
        graph.Figure: _description_
    """
    fig = graph.Figure()
    fig.add_trace(
        go.Heatmap(
            x=time,
            y=frequency,
            z=10 * log10(Pxx) + correction,
            colorscale=colorscale,
            showscale=False,
            zmin=zmin,
            zmax=zmax,
            zsmooth="best"
        )
    )
    fig.update_layout(yaxis=dict(title="Frequency [Hz]"))

    return fig


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


def psd(x: list or pd.Series, sample_rate: int, window_size: int = 4096) -> tuple:
    """
    Compute the power spectral density of a signal.

    Args:
        x (array): signal
        sample_rate (int): sample rate of the signal
        sample_window (int, optional): length of the window to use for the FFT. Defaults to 4096.

    Returns:
        tuple: power spectral density
    """

    f = fft.rfft(x)
    f1 = f[0 : int(window_size / 2)]
    pf1 = 2 * abs(f1 * conj(f1)) / (sample_rate * window_size)
    lpf1 = 10 * log10(pf1)
    w = arange(1, window_size / 2 + 1)
    lp = lpf1[1 : int(window_size / 2)]
    w1 = sample_rate * w / window_size

    return w1, lp
