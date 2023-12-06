import numpy as np


def distance(x1:float, y1:float, x2:float, y2:float) -> float:
    """
    Calculates distance between two points

    Args:
        x1 (float): x coordinate of point 1
        y1 (float): y coordinate of point 1
        x2 (float): x coordinate of point 2
        y2 (float): y coordinate of point 2

    Returns:
        float: distance between two points
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def circle_intersection(x0:float, y0:float, r0:float, x1:float, y1:float, r1:float, x2=None, y2=None, r2=None) -> list:
    """
    Calculates intersection of two or three circles

    Args:
        x0 (float): x coordinate of circle 1
        y0 (float): y coordinate of circle 1
        r0 (float): radius of circle 1
        x1 (float): x coordinate of circle 2
        y1 (float): y coordinate of circle 2
        r1 (float): radius of circle 2
        x2 (_type_, optional): x coordinate of circle 3. Defaults to None.
        y2 (_type_, optional): y coordinate of circle 3. Defaults to None.
        r2 (_type_, optional): radius of circle 3. Defaults to None.

    Returns:
        list: list of intersection points
    """

    epsilon = 0.0000000000001

    dx = x1 - x0
    dy = y1 - y0

    d = np.sqrt(dy**2 + dx**2)

    if d > (r0 + r1):
        return []
    if d < abs(r0 - r1):
        return []

    a = (r0**2 - r1**2 + d**2) / (2 * d)

    p_x = x0 + (dx * a / d)
    p_y = y0 + (dy * a / d)

    h = np.sqrt(r0**2 - a**2)

    rx = -dy * (h / d)
    ry = dx * (h / d)

    point1_x = p_x + rx
    point2_x = p_x - rx
    point1_y = p_y + ry
    point2_y = p_y - ry

    intersections = []

    if x2 is None:
        intersections.append([point1_x, point1_y])
        intersections.append([point2_x, point2_y])

    else:
        dx = point1_x - x2
        dy = point1_y - y2
        d1 = np.sqrt(dy**2 + dx**2)

        dx = point2_x - x2
        dy = point2_y - y2
        d2 = np.sqrt(dy**2 + dx**2)

        if abs(d1 - r2) < epsilon:
            intersections.append([point1_x, point1_y])

        if abs(d2 - r2) < epsilon:
            intersections.append([point2_x, point2_y])

    return intersections
