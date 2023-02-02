from asyncio import start_server
from dankpy import graph, dt, file
import plotly.graph_objs as go
import pandas as pd
from scipy import signal
from numpy import log10, fft, conj, abs, arange
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


class audiofile(object):
    """
    _summary_

    Args:
        object (_type_): _description_
    """


    def __init__(self, filepath, start=None):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.audio, self.sample_rate = librosa.load(filepath)
        self.duration = librosa.get_duration(filename=self.filepath)

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

        self.metadata = file.metadata(self.filepath)
        self.data["signal"] = self.audio

    def trim(self, start, end=None, length=None):

        if not isinstance(start, datetime):
            start = dt.read_datetime(start)

        if end == None:
            end = start + timedelta(seconds=length)
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

        return sample

    def spectrogram(self, window_size=8192, nfft=4096, noverlap=4096, nperseg=8192):
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
        window_size=8192,
        nfft=4096,
        noverlap=4096,
        nperseg=8192,
        zmin=None,
        zmax=None,
        correction=0
    ):

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
            correction=correction
        )

        return fig

    def signal(self):

        fig = graph.graph()
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

    def psd(self, window_size=4096):
        frequency, power = psd(
            self.data.signal, self.sample_rate, window_size=window_size
        )

        return frequency, power


# def audio_file(filepath, start, end):


def spectrogram(
    data,
    sample_rate,
    window_size=8192,
    nfft=4096,
    noverlap=4096,
    nperseg=8192,
    start=None,
    end=None,
):
    """Generates a spectrogram

    :param data: audio data
    :type data: list or array
    :param sample_rate: sampling rate of data
    :type sample_rate: int or float
    :param window_size: window size for hanning window, defaults to 2*8192
    :type window_size: int or float, optional
    :param noverlap: overlap, defaults to 1024/16
    :type noverlap: int or float, optional
    :param start: start time of audio data, defaults to None
    :type start: Datetime, optional
    :param end: end time of audio data, defaults to None
    :type end: Datetime, optional
    :return: output of spectrogram [time, freqs, Pxx]
    :rtype: array
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
    time,
    frequency,
    Pxx,
    colorscale="Jet",
    zmin=None,
    zmax=None,
    correction=0,
    save=None,
):
    """Generates spectrograph

    :param time: time
    :type time: array
    :param frequency: frequency array
    :type frequency: array
    :param Pxx: power
    :type Pxx: array of arrays
    :param colorscale: Heatmap colorscale. See plotly documentation, defaults to "Jet"
    :type colorscale: str, optional
    :param zmin: minimum cmap color scale, defaults to None
    :type zmin: int, optional
    :param zmax: maximum cmap color scale, defaults to None
    :type zmax: int, optional
    :param correction: magnitude, defaults to None
    :type correction: int or float, optional
    :param save: location to save file, defaults to None
    :type save: str, optional
    :return: spectrograph
    :rtype: plotly.go figure
    """
    fig = graph.graph()
    fig.add_trace(
        go.Heatmap(
            x=time,
            y=frequency,
            z=10 * log10(Pxx) + correction,
            colorscale=colorscale,
            showscale=False,
            zmin=zmin,
            zmax=zmax
            # colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(yaxis=dict(title="Frequency [Hz]"))

    if save:
        fig.save_image(save)

    return fig


def write_audio(data, filepath, sample_rate):
    """Writes audiofile of data with set samplerate. Omit extension, will output only wav.

    :param data: data to output
    :type data: numpy.array
    :param filepath: filepath of output
    :type filepath: str
    :param sample_rate: desired file sample rate
    :type sample_rate: int
    """
    sf.write(filepath + ".wav", data, sample_rate)


# def write_spectrogram(filepath, data):
#     window_size = 1024
#     window = np.hanning(window_size)
#     stft = librosa.core.spectrum.stft(
#         np.array(data), n_fft=window_size, hop_length=512, window=window
#     )
#     out = 2 * abs(stft) / sum(window)
#     fig = plt.Figure()
#     canvas = FigureCanvas(fig)
#     ax = fig.add_subplot(111)
#     p = librosa.display.specshow(
#         librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis="log", x_axis="time"
#     )
#     fig.savefig(filepath + ".png")


def mp3towav(file, output, output_format="wav"):
    """Converts mp3 to wav

    :param filepath: filepath of mp3
    :type filepath: str
    """
    sound = AudioSegment.from_mp3(file)
    sound.export(output, format=output_format)


def psd(x, sample_rate, window_size=4096):
    """
    Compute the power spectral density of a signal.

    Args:
        x (array): signal
        sample_rate (_type_): sample rate of the signal
        sample_window (int, optional): length of the window to use for the FFT. Defaults to 4096.

    Returns:
        _type_: power spectral density
    """

    f = fft.rfft(x)
    f1 = f[0 : int(window_size / 2)]
    pf1 = 2 * abs(f1 * conj(f1)) / (sample_rate * window_size)
    lpf1 = 10 * log10(pf1)
    w = arange(1, window_size / 2 + 1)
    lp = lpf1[1 : int(window_size / 2)]
    w1 = sample_rate * w / window_size

    return w1, lp
