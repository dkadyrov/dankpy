def meters2feet(m):
    """
    Converts Meters to Feet

    Args:
        m: Float in meters

    Returns:
        Float in feet

    """
    return 3.28084 * m


def feet2meters(ft):
    """
    Converts Feet to Meters

    Args:
        ft: Float in feet

    Returns:
        Float in meters

    """
    return ft / 3.28084


def f2c(f):
    """
    Converts Fahrenheight to Celcius

    Args:
        f: Float in Fahrenheight

    Returns:
        Float in Celcius

    """
    return (f - 32) * 5 / 9


def c2f(c):
    """
    Converts Celcius to Fahrenheight

    Args:
        f: Float in Celcius

    Returns:
        Float in Fahrenheight

    """
    return (c * 9 / 5) + 32

def c2k(c):
    """Converts Celcius to Kelvin

    :param c: Celcius 
    :type c: float
    :return: Kelvin
    :rtype: float
    """

    return c + 273.15

def k2c(c):
    """Converts Kelvin to Celcius

    :param k: Kelvin 
    :type k: float
    :return: Celcius
    :rtype: float
    """

    return c + 273.15