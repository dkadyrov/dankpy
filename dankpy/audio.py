from starpy.utilities import graph
import plotly.graph_objs as go
import pandas as pd
from scipy import signal
from numpy import log10
from datetime import timedelta
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import soundfile as sf
import librosa
import librosa.display
from pydub import AudioSegment

valid_audio = ["wav", "flac", "mp3", "ogg", "aiff", "au"]


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

    freqs, time, Pxx = signal.spectrogram(
        data, sample_rate, window=window, mode="psd", noverlap=noverlap
    )

    if start:
        if end == None:
            end = start + timedelta(seconds=len(data) / sample_rate)
        datetime = pd.date_range(start, end, periods=len(time))

        time = datetime

    return time, freqs, Pxx


def spectrograph(
    time,
    frequency,
    Pxx,
    colorscale="Jet",
    zmin=None,
    zmax=None,
    correction=None,
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
    if correction == None:
        correction = 0

    fig = graph()
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
