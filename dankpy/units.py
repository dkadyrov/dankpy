def meters_to_feet(m: float) -> float:
    """
    Converts Meters to Feet

    Args:
        m: Float in meters

    Returns:
        Float in feet

    """
    return 3.28084 * m

def inches_to_meters(inches: float) -> float:
    """
    Converts Inches to Meters

    Args:
        inches: Float in inches

    Returns:
        Float in meters

    """
    return inches * 0.0254


def feet_to_meters(ft: float) -> float:
    """
    Converts Feet to Meters

    Args:
        ft: Float in feet

    Returns:
        Float in meters

    """
    return ft / 3.28084


def fahrenheight_to_celcius(f: float) -> float:
    """
    Converts Fahrenheight to Celcius

    Args:
        f: Float in Fahrenheight

    Returns:
        Float in Celcius

    """
    return (f - 32) * 5 / 9


def celcius_to_fahrenheight(c: float) -> float:
    """
    Converts Celcius to Fahrenheight

    Args:
        f: Float in Celcius

    Returns:
        Float in Fahrenheight

    """
    return (c * 9 / 5) + 32


def celcius_to_kelvin(c: float) -> float:
    """
    Converts Celcius to Kelvin

    Args:
        c (float): Celcius

    Returns:
        float: Kelvin
    """

    return c - 273.15


def kelvin_to_celcius(k: float) -> float:
    """
    Converts Kelvin to Celcius

    Args:
        k (float): Kelvin

    Returns:
        float: Celcius
    """

    return k + 273.15
