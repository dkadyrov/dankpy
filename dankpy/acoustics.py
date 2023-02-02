import math
import numpy as np
import pandas as pd
from numpy import pi, polymul
from scipy.signal import medfilt
from scipy.signal import bilinear


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


## 1 atm in Pa
ps0 = 1.01325e5


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


def a_weighted(f):
    """
    Converts frequency into A-weighted decibels

    :param f: _description_
    :return: _description_
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


def decibel(x, u=None, r=None):
    """
    Converts a value to decibels

    :param x: _description_
    :param u: _description_, defaults to None
    :param r: _description_, defaults to None
    :return: _description_
    """

    if u is None and r is None:
        u = "power"
        x = abs(x) ** 2
    else:
        if u is None and isinstance(r, str):
            r = u
            u = "voltage"
        else:
            r = 1

    if u not in ["power", "voltage"]:
        raise "u must be power or voltage"
    if u == "voltage":
        x = abs(x) ** 2 / r

    return (10 * np.log10(x) + 300) - 300


def integrate_PSD_db(frequency, signal, f_low, f_high):
    data = pd.DataFrame({"frequency": frequency, "signal": signal})

    filt = data[(data.frequency <= f_high) & (data.frequency >= f_low)]

    return 10 * np.log10(sum(10 ** (filt.signal / 10)) * data.frequency.iloc[1])


# def integrate_PSD2D_db(data, f_low, f_high):
#     filt = data[(data.frequency <= f_high) & (data.frequency >= f_low)]
#     return 10 * np.log10(sum(10 ** (filt.signal/10)) * data.
#     frequency.iloc[1])


def integrate_a_weighted(f, x):
    """
    _summary_

    :param f: _description_
    :type f: _type_
    :param x: _description_
    :type x: _type_
    :return: _description_
    :rtype: _type_
    """
    return decibel(sum(10 ** (x / 10) * (a_weighted(f) ** 2) * f[1]), "power")

def integrate_a_weighted_medfilt(f, x):
    return decibel(
        sum(10 ** (medfilt(x, 49) / 10) * (a_weighted(f) ** 2) * f[1]), "power"
    )


def atmospheric_attenuation(Tin, Psin, hrin, dist, f):
    T = Tin + 273.15  # temp input in K
    To1 = 273.15  # triple point in K
    To = 293.15  # ref temp in K

    Ps = Psin / 29.9212598  # static pressure in atm
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

    h = hrin * Psat / Ps  # calculate the absolute humidity

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
    # amp(isnan(amp))=0

    return amp
