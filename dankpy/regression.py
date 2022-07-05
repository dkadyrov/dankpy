from scipy.optimize import curve_fit, leastsq
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
# import matlab.engine


def linear(x, a, b):
    return a * x + b


def exponential(x, a, b, c):
    return a * np.exp(b * (x - c))


def quadratic(x, a, b, c):
    return a * (x ** 2.0) + b * x + c


def logistic(x, x0, k, L=None):
    if L == None:
        return 1 / (1 + np.exp(-k * (x0 - x)))
    else:
        # def logistic_mag(x, L, k, x0):
        return L / (1 + np.exp(-k * (x0 - x)))
        # return L / (1+np.exp(-(k*x+x0)))


def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def exponential_model(x, y, x2):

    params, corr = curve_fit(exponential, x, y, p0=[1, 0, 1], maxfev=5000)

    y2 = exponential(x2, *params)

    perr = np.sqrt(np.diag(corr))

    results = {"residuals": corr, "sigma": perr}

    return y2, results


def logistic_model(x, y, x2, weights=None):
    ## scale = y[-1]

    df = pd.DataFrame()
    df["x"] = x
    df["y"] = y

    df.iloc[-1, df.columns.get_loc('y')] = 0

    # df = df[df["y"] >= 0]

    p0 = [
        df.iloc[(df["y"] - df.y.mean()).abs().argsort()[:1]].iloc[0].x,
        1 / 20,
        max(y),
    ]

    # p0 = [0, 0, 0]

    # p0= [250, 1/40, 1]

    # weights = np.diag([1/x if x != 0 else 1 for x in weights])

    ## average_per_row = [sum(row) / len(row) if type(row[0]) is not str else None for row in data]

    # weights = [(max(1/_**2 , 0.00001)) if _ else 1 for _ in weights]
    # weights = weights**2

    weights = 1/weights
    weights[-1] = sum(weights)

    params, corr = curve_fit(
        logistic, df.x, df.y, p0, sigma=weights, method="lm", maxfev=10000000
    )
    ## params, corr = curve_fit(logistic, df.x, df.y, p0, maxfev=10000000)

    y2 = logistic(x2, *params)

    perr = np.sqrt(np.diag(corr))

    results = {"residuals": params, "sigma": perr}

    return y2, results


# def logistic_fit(x, y, x2, weights=None):

#     eng = matlab.engine.start_matlab()

#     ft = eng.fittype("1./(1 + exp(-k*(x0-x)))", "independent", "x", "dependent", "y")

#     weights = tuple(weights)

#     opts = eng.fitoptions("Method", "NonlinearLeastSquares", "Weight", weights)
#     opts.Display = "Off"
#     opts.StartPoint = [1 / 100, 500]

#     # % Fit model to data.
#     fitresult, other = eng.fit(x.values, y.values, ft, opts)
#     c0 = eng.coeffvalues(fitresult)

#     return fitresult, c0


# def logistic_lmsr(x, y, x2, weights=None):
#     x, flag = leastsq(logistic, x, )


def quadratic_model(x, y, x2):
    def quadratic(x, a, b, c):
        return a * (x ** 2.0) + b * x + c

    params, corr = curve_fit(quadratic, x, y, p0=[1, 1, 1])

    perr = np.sqrt(np.diag(corr))

    y2 = quadratic(x2, *params)

    results = {"residuals": corr, "sigma": perr}

    return y2, results


def linear_model(x, y, x2):
    def linear(x, a, b):
        return a * x + b

    params, corr = curve_fit(linear, x, y, p0=[1, 1])

    y2 = linear(x2, *params)

    perr = np.sqrt(np.diag(corr))

    results = {"residuals": corr, "sigma": perr}

    return y2, results


def gaussian_model(x, y, x2):
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    params, corr = curve_fit(gaussian, x, y)

    y2 = gaussian(x2, *params)

    perr = np.sqrt(np.diag(corr))

    results = {"residuals": corr, "sigma": perr}

    return y2, results
