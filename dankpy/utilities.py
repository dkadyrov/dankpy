import numpy as np
from datetime import datetime, timedelta
import math
import pandas as pd

def pd_listfilter(df: pd.DataFrame,  filter: list, column=None) -> pd.DataFrame:
    pattern = '|'.join(filter)

    if column is None: 
        df = df[df.apply(lambda r: r.str.contains(pattern, na=False).any(), axis=1)]
    if isinstance(column, list): 
        df = df[df[column].apply(lambda r: r.str.contains(pattern, na=False).any(), axis=1)]
    elif isinstance(column, str): 
        df = df[df[column].str.contains(pattern, na=False)]

    return df

def latex_table(filepath: str, table: pd.DataFrame, caption: str) -> None:
    """
    Generates latex table from pandas Dataframe

    Args:
        filepath (str): path to file
        table (pd.DataFrame): table to convert
        caption (str): caption for table
    """
    with open(filepath, "w") as f:
        f.write(
            table.to_latex(
                index=False,
                position="h!",
                caption=caption,
                bold_rows=True,
                escape=False,
            )
        )


def round_to_nearest(x: float, base:int=50) -> float:
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
    # if base - x % base <= 10:
    #     return base * math.ceil(x / base) + base
    # try:
    #     return base * math.ceil(x / base)
    except:
        print(x)


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


def pd_to_html(data: pd.DataFrame) -> str:
    """
    Outputs pandas Dataframe to HTML table

    Args:
        data (pd.DataFrame): dataframe to convert

    Returns:
        str: HTML table
    """
    return (
        data.to_html(
            index=False, classes=["table-bordered", "table-striped", "table-hover"]
        )
        .replace("\n", "")
        .replace("dataframe", "table")
    )
