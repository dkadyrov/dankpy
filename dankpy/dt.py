import xlrd
import dateutil.parser as parser
from datetime import datetime, timedelta

import pandas as pd

class DT(datetime):
    """
    Datetime class with additional methods
    """
    def __init__(self, *args, **kw):
        super(DT, self).__init__(*args, **kw)


def read_datetime(string: str) -> datetime:
    """
    Reads and converts datetime

    Args:
        string (String): datetime string

    Returns:
        datetime.Datetime: converted datetime
    """

    try:
        dt = datetime.strptime(string, "%Y_%m_%d")
    except:
        try:
            dt = datetime.strptime(string, "%Y_%m_%d_%H_%M_%S.%f")
        except:
            try:
                dt = datetime.strptime(string, "%Y_%m_%d_%H_%M_%S")
            except:
                dt = parser.parse(string, fuzzy=True)

    return dt


def read_matlab_date(date: str) -> datetime:
    """
    Converts matlab date to datetime

    Args:
        date (str): date to convert

    Returns:
        datetime.datetime: converted date
    """

    return (
        datetime.fromordinal(int(date)) + timedelta(days=date % 1) - timedelta(days=366)
    )


def read_excel_date(date: float) -> datetime:
    """
    Converts excel date to datetime

    Args:
        date (str): date to convert

    Returns:
        datetime.datetime: converted date
    """

    return xlrd.xldate.xldate_as_datetime(date, 0)


def round_seconds(obj: datetime) -> datetime:
    """
    Rounds milliseconds to nearest second

    Args:
        obj (datetime): datetime to round

    Returns:
        datetime: rounded datetime
    """
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)

    return obj.replace(microsecond=0)


def write_time(time, milliseconds: bool = False) -> str:
    """
    Writes time in HH:MM:SS format

    Args:
        time (datetime): Time to write
        milliseconds (bool, optional): Option to add milliseconds. Defaults to False.

    Returns:
        str: Time in HH:MM:SS format
    """

    if milliseconds:
        return time.strftime("%H:%M:%S.%f")
    else:
        return time.strftime("%H:%M:%S")


def write_date(date, seperator: str = "-") -> str:
    """
    Writes date in YYYY-MM-DD format

    Args:
        date (datetime): Date to write
        seperator (str, optional): Specifies date seperator. Defaults to "-".

    Returns:
        str: Date in YYYY-MM-DD format
    """

    return date.strftime(f"%Y{seperator}%m{seperator}%d")


def write_datetime(dt, seperator: str = "-", ms: bool = False, directory=False) -> str:
    """
    Writes datetime in YYYY-MM-DD HH:MM:SS format

    Args:
        dt (datetime): Datetime to write
        seperator (str, optional): Specifies date seperator. Defaults to "-".
        milliseconds (bool, optional): Specifies if milliseconds. Defaults to False.

    Returns:
        str: Datetime in YYYY-MM-DD HH:MM:SS format
    """

    if directory: 
        if ms: 
            return dt.strftime(f"%Y{seperator}%m{seperator}%d{seperator}%H{seperator}%M{seperator}%S.%f")
        else: 
            return dt.strftime(f"%Y{seperator}%m{seperator}%d{seperator}%H{seperator}%M{seperator}%S")

    if ms:
        return dt.strftime(f"%Y{seperator}%m{seperator}%d %H:%M:%S.%f")
    else:
        return dt.strftime(f"%Y{seperator}%m{seperator}%d %H:%M:%S")