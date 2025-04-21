import pandas as pd


def remove_outliers(df: pd.DataFrame, col, iqr_coefficient: float = 2) -> pd.DataFrame:
    """
    Removes outliers from a dataframe column using the IQR method

    Args:
        df (pd.DataFrame): DataFrame to remove outliers from
        col (_type_): Column name to remove outliers from
        iqr_coefficient (float, optional): IQR coefficient. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    distance = iqr_coefficient * (Q3 - Q1)
    lower_fence, upper_fence = Q1 - distance, Q3 + distance

    # Keep rows where values are within the bounds
    df_out = df.loc[~((df[col] < lower_fence) | (df[col] > upper_fence))]

    return df_out

def trimean(data: pd.Series) -> float:
    """
    Calculate the trimean of a series of numbers.

    The trimean is a measure of central tendency that combines the mean and median.
    It is calculated as (Q1 + 2 * Q2 + Q3) / 4, where Q1 is the first quartile,
    Q2 is the median, and Q3 is the third quartile.

    Args:
        data (pd.Series): A pandas Series containing the data.

    Returns:
        float: The trimean of the data.
    """
    q1 = data.quantile(0.25)
    q2 = data.median()
    q3 = data.quantile(0.75)

    return (q1 + 2 * q2 + q3) / 4

def harmonic_mean(data: pd.Series) -> float:
    """
    Calculate the harmonic mean of a series of numbers.

    The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals.
    It is defined as n / (1/x1 + 1/x2 + ... + 1/xn), where n is the number of values.

    Args:
        data (pd.Series): A pandas Series containing the data.

    Returns:
        float: The harmonic mean of the data.
    """
    return len(data) / sum(1 / x for x in data if x != 0)

def geometric_mean(data: pd.Series) -> float:
    """
    Calculate the geometric mean of a series of numbers.

    The geometric mean is the nth root of the product of n values.
    It is defined as (x1 * x2 * ... * xn)^(1/n), where n is the number of values.

    Args:
        data (pd.Series): A pandas Series containing the data.

    Returns:
        float: The geometric mean of the data.
    """
    return data.prod() ** (1 / len(data))

def RMS(data: pd.Series) -> float:
    """
    Calculate the root mean square (RMS) of a series of numbers.

    The RMS is the square root of the average of the squares of the values.
    It is defined as sqrt((x1^2 + x2^2 + ... + xn^2) / n), where n is the number of values.

    Args:
        data (pd.Series): A pandas Series containing the data.

    Returns:
        float: The root mean square of the data.
    """
    return (data ** 2).mean() ** 0.5

def MAD(data: pd.Series) -> float:
    """
    Calculate the mean absolute deviation (MAD) of a series of numbers.

    The MAD is the average of the absolute deviations from the mean.
    It is defined as (|x1 - mean| + |x2 - mean| + ... + |xn - mean|) / n,
    where n is the number of values.

    Args:
        data (pd.Series): A pandas Series containing the data.

    Returns:
        float: The mean absolute deviation of the data.
    """
    return (data - data.mean()).abs().mean()

def midrange(data: pd.Series) -> float:
    """
    Calculate the mid-range of a series of numbers.

    The mid-range is the average of the maximum and minimum values.
    It is defined as (max + min) / 2.

    Args:
        data (pd.Series): A pandas Series containing the data.

    Returns:
        float: The mid-range of the data.
    """
    return (data.max() + data.min()) / 2