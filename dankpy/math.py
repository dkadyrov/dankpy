import numpy as np


def distance(x1, y1, x2, y2):
    """
    Calculates distance between two points

    Args:
        x1 (float): x coordinate of point 1
        y1 (float): y coordinate of point 1
        x2 (float): x coordinate of point 2
        y2 (float): y coordinate of point 2

    Returns:
        _type_: _description_
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def circle_intersection(x0, y0, r0, x1, y1, r1, x2=None, y2=None, r2=None):
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
        array: array of intersection points
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

    if x2 == None:
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
