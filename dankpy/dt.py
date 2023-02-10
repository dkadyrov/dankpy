import xlrd
import dateutil.parser as parser
from datetime import datetime, timedelta

class DT(datetime):
    def __init__(self, *args, **kw):
        super(DT, self).__init__(*args, **kw)

    def write_time(self, milliseconds:bool=False) -> str:
        if milliseconds:
            return self.strftime("%H:%M:%S.%f")
        else: 
            return self.strftime("%H:%M:%S")

    def write_date(self, seperator:str="-") -> str:
        return self.strftime(f"%Y{seperator}%m{seperator}%d")
    
    def write_datetime(self, seperator:str="-", milliseconds:bool=False) -> str:
        return self.strftime(f"%Y{seperator}%m{seperator}%d %H:%M:%S.%f")

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
                dt = parser.parse(string)
                
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

def write_time(dt: datetime) -> str:
    """
    Writes time in HH:MM:SS format
    """

    return dt.strftime("%H:%M:%S")