import numpy as np 
import pandas as pd


def round_to_nearest(x: float, base: int = 50) -> float:
    """
    Rounds up or down to nearest base value

    Args:
        x (float): value to round
        base (int, optional): value to round to. Defaults to 50.

    Returns:
        float: rounded value
    """
    try:
        return base * round(x / base)
    except Exception as e:
        print(e)


def lower_keys(tree: dict) -> dict:
    """
    Normalizes a dictionary to have lowercase and snakecase keys

    Args:
        tree (dict): dictionary to normalize

    Returns:
        dict: normalized dictionary
    """
    data = {}
    for k in tree.keys():
        if isinstance(tree[k], dict):
            data[k.lower().replace(" ", "_")] = lower_keys(tree[k])
        else:
            data[k.lower().replace(" ", "_")] = tree[k]

    return data


def list_to_pd(list_of_dicts: list) -> pd.DataFrame:
    """
    Converts a list of dictionaries to pandas Dataframe object with keys as columns

    Args:
        list_of_dicts (list): list of dictionaries to convert

    Returns:
        _type_: _description_
    """
    return pd.DataFrame(x.__dict__ for x in list_of_dicts)


def coiled_cable_length(d, num):
    return np.pi * d * num

def rms(x):
    return np.sqrt(np.mean(x**2))