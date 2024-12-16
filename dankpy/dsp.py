import numpy as np
import pandas as pd
from scipy.signal import correlate, savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal.windows import tukey


def count_crossings(arr, target):
    count = 0
    previous = arr[0]
    
    for current in arr[1:]:
        if (previous < target and current >= target) or (previous > target and current <= target):
            count += 1
        previous = current
    
    return count


def lowess_smooth(x, y, size):
    """Generates lowess smoothing

    :param x: x data
    :type x: numpy.array
    :param y: y data
    :type y: numpy.array
    :param size: smoothing size (fraction)
    :type size: float
    :return: smoothed y data values
    :rtype: numpy.array
    """

    data = pd.DataFrame()
    data["x"] = x
    data["y"] = y

    lws = lowess(y, x, frac=size)

    smooth = pd.DataFrame()
    smooth["x"] = lws[:, 0]
    smooth["y"] = lws[:, 1]

    data = pd.merge_ordered(data, smooth, on="x", suffixes=["orig", "smooth"])

    return data["ysmooth"].values
    # smooth["x"] =


def smooth(data, window="tukey", num=8, ratio=0.8):
    """performs rolling window smoothing

    :param data: data to smooth
    :type data: numpy.array
    :param window: window. Input as numpy array or use "tukey" as default, defaults to "tukey"
    :type window: numpy.array or str, optional
    :param num: window size, defaults to 8
    :type num: int, optional
    :param ratio: smoothing ratio, defaults to 0.8
    :type ratio: float, optional
    :return: smoothed array
    :rtype: numpy.array
    """

    if window == "tukey":
        window = tukey(num, ratio)
        window = window / sum(window)

    # npad = len(window)-1

    # u_pad = np.pad(data, (npad//2, npad - npad//2), mode = "constant")

    # valid = np.convolve(u_pad, window, "valid")

    # full = np.convolve(data, window, "full")

    # first = npad - npad//2

    # conv = full[first:first+len(data)]

    # conv = np.convolve(data, window, mode='same')
    # valid_trim = int(np.floor(num/2))
    # conv[0:1+valid_trim] = 0
    # conv[-1-valid_trim:-1] = 0

    # conv = np.convolve(data, window, mode='valid')
    # conv = np.append(conv, np.zeros((1, abs(len(data)-len(conv)))))

    conv = np.convolve(data, window, mode="same")
    valid_trim = int(np.floor(num / 2))
    conv[0 : 1 + valid_trim] = 0
    conv[-1 - valid_trim : -1] = 0

    # padded = np.pad(data, (npad//2, npad-npad//2), mode="edge")

    # conv = np.convolve(data, window, "same")
    # npad = len(window) - 1

    # padded = np.pad(data, (npad//2, npad-npad//2), mode="constant")

    return conv
    # full = np.convolve(data, window, "full")
    # first = npad - npad//2

    # return full[first: first+len(data)]
    # return np.convolve(data, window, "same")


def smooth_data_np_cumsum_my_average(arr, span):
    cumsum_vec = np.cumsum(arr)
    moving_average = (cumsum_vec[2 * span :] - cumsum_vec[: -2 * span]) / (2 * span)

    # The "my_average" part again. Slightly different to before, because the
    # moving average from cumsum is shorter than the input and needs to be padded
    front, back = [np.average(arr[:span])], []
    for i in range(1, span):
        front.append(np.average(arr[: i + span]))
        back.insert(0, np.average(arr[-i - span :]))
    back.insert(0, np.average(arr[-2 * span :]))
    return np.concatenate((front, moving_average, back))


def smooth_data_savgol_0(arr, span):
    return savgol_filter(arr, span * 2 + 1, 0)


def smooth_data_savgol_1(arr, span):
    return savgol_filter(arr, span * 2 + 1, 1)


def new_smooth(data, num):
    window = tukey(num)

    return np.convolve(data, window, "valid")


def window(array, window_size, freq):
    """Splits an array into subarrays of specified window_size and with a overlap of set frequency

    :param array: array to split
    :type array: numpy.array
    :param window_size: subarray length
    :type window_size: int
    :param freq: _description_
    :type freq: _type_
    :return: _description_
    :rtype: _type_
    """

    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], freq)]


def tukeywin(window_length, alpha=0.5):
    """The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.

    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output

    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html

    """
    # Special cases
    if alpha <= 0:
        return np.ones(window_length)  # rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x < alpha / 2
    w[first_condition] = 0.5 * (
        1 + np.cos(2 * np.pi / alpha * (x[first_condition] - alpha / 2))
    )

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha / 2)
    w[third_condition] = 0.5 * (
        1 + np.cos(2 * np.pi / alpha * (x[third_condition] - 1 + alpha / 2))
    )

    return w


def histogram(x, bins, edges=True):
    if edges is True:
        bin_edges = np.r_[-np.Inf, 0.5 * (bins[:-1] + bins[1:]), np.Inf]
        counts, edges = np.histogram(x, bin_edges)

    else:
        counts, edges = np.histogram(x, bins)

    return counts, edges
