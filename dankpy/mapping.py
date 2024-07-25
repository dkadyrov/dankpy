import math
from numpy import cos, absolute, pi
import numpy as np
from dankpy.maputils import mymaptiles


def lla_to_flatdumb(
    lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float
) -> tuple:
    """
    Converts longitude, latitude, altitude to flat earth coordinates using a reference point

    Args:
        lat (float): Latitude
        lon (float): Longitude
        alt (float): Altitude
        lat0 (float): Reference latitude for the origin of the estimation and the origin of the flat Earth coordinate system
        lon0 (float): Reference longitude for the origin of the estimation and the origin of the flat Earth coordinate system
        alt0 (float): Reference altitude for the origin of the estimation and the origin of the flat Earth coordinate system

    Returns:
        tuple: Flat earth coordinates (x, y, z)
    """

    re = 6378137
    re_c = re * cos((pi / 180) * absolute(lat0))
    x = (lon - lon0) * (re_c * pi) / 180
    y = (lat - lat0) * (re * pi) / 180
    z = alt - alt0
    return (x, y, z)


def flat_to_lla(
    x: float, y: float, z: float, lat0: float, lon0: float, alt0: float
) -> tuple:
    """
    Converts flat earth coordinates to longitude, latitude, altitude using a reference point

    Args:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
        lat0 (float): Reference latitude for the origin of the estimation and the origin of the flat Earth coordinate system
        lon0 (float): Reference longitude for the origin of the estimation and the origin of the flat Earth coordinate system
        alt0 (float): Reference altitude for the origin of the estimation and the origin of the flat Earth coordinate system

    Returns:
        tuple: Longitude, latitude, altitude tuple (lat, lon, alt)
    """

    re = 6378137
    re_c = re * cos((pi / 180) * absolute(lat0))
    lon = x * 180 / (pi * re_c) + lon0
    lat = y * 180 / (pi * re) + lat0
    alt = z + alt0

    return (lat, lon, alt)


def lla2flat(lla: list, llo: list, psio: float, href: float) -> list:
    """
    Converts latitude, longitude, altitude to Flat Earth Coordinates

    Args:
        lla (list): Geodetic coordinates (latitude, longitude, and altitude), in [degrees, degrees, meters].
        latitude and longitude values can be any value.  However, latitude values of +90 and -90 may return unexpected values because of singularity at the poles
        llo (list): Reference location, in degrees, of latitude and longitude, for the origin of the estimation and the origin of the flat Earth coordinate system.
        psio (float): Angular direction of flat Earth x-axis (degrees clockwise from north), which is the angle in degrees used for converting flat Earth x and y coordinates to the North and East coordinates
        href (float): Reference height from the surface of the Earth to the flat Earth frame with regard to the flat Earth frame, in meters.

    Returns:
        list: Flat Earth coordinates (x, y, z)
    """

    R = 6378137.0  # Equator radius in meters
    f = 0.00335281066474748071  # 1/298.257223563, inverse flattening

    Lat_p = lla[0] * math.pi / 180.0  # from degrees to radians
    Lon_p = lla[1] * math.pi / 180.0  # from degrees to radians
    Alt_p = lla[2]  # meters

    # Reference location (lat, lon), from degrees to radians
    Lat_o = llo[0] * math.pi / 180.0
    Lon_o = llo[1] * math.pi / 180.0

    psio = psio * math.pi / 180.0  # from degrees to radians

    dLat = Lat_p - Lat_o
    dLon = Lon_p - Lon_o

    ff = (2.0 * f) - (f**2)  # Can be precomputed

    sinLat = math.sin(Lat_o)

    # Radius of curvature in the prime vertical
    Rn = R / math.sqrt(1 - (ff * (sinLat**2)))

    # Radius of curvature in the meridian
    Rm = Rn * ((1 - ff) / (1 - (ff * (sinLat**2))))

    dNorth = (dLat) / math.atan2(1, Rm)
    dEast = (dLon) / math.atan2(1, (Rn * math.cos(Lat_o)))

    # Rotate matrice clockwise
    Xp = (dNorth * math.cos(psio)) + (dEast * math.sin(psio))
    Yp = (-dNorth * math.sin(psio)) + (dEast * math.cos(psio))
    Zp = -Alt_p - href

    return Xp, Yp, Zp


def zoom_center(
    lons: tuple = None,
    lats: tuple = None,
    lonlats: tuple = None,
    format: str = "lonlat",
    projection: str = "mercator",
    width_to_height: float = 2.0,
) -> tuple:
    """
    _summary_

    Raises:
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError("Must pass lons & lats or lonlats")

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        "lon": round((maxlon + minlon) / 2, 6),
        "lat": round((maxlat + minlat) / 2, 6),
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array(
        [
            0.0007,
            0.0014,
            0.003,
            0.006,
            0.012,
            0.024,
            0.048,
            0.096,
            0.192,
            0.3712,
            0.768,
            1.536,
            3.072,
            6.144,
            11.8784,
            23.7568,
            47.5136,
            98.304,
            190.0544,
            360.0,
        ]
    )

    if projection == "mercator":
        margin = 1.2
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(f"{projection} projection is not implemented")

    return zoom, center

def find_extents(latitudes, longitudes): 
    """
    Find the extents of a set of latitudes and longitudes

    Args:
        latitudes (list): List of latitude values
        longitudes (list): List of longitude values

    Returns:
        tuple: Extents of the latitudes and longitudes (minlon, maxlon, minlat, maxlat)
    """
    minlon = min(longitudes)
    maxlon = max(longitudes)
    minlat = min(latitudes)
    maxlat = max(latitudes)
    
    return minlon, minlat, maxlon, maxlat


def plot_basemap(
    ax, extents, map_url="http://tile.openstreetmap.org/{z}/{x}/{y}.png", z=16
):
    # plot the map
    (ax0, axi) = mymaptiles.draw_map(extents, tile=map_url, ax=ax, z=z)
    axi.set_interpolation("lanczos")

    return ax0, axi

def get_meters_per_lat_lon(lat):
    """Returns the meters per degree latitude and longitude at a given latitude.

    :param lat: latitude in decimal degrees
    :return: meters per degree latitude, meters per degree longitude
    """
    r = 6378101.0  # Radius of earth in meters.
    meters_per_lat = np.pi * r / 180.0
    meters_per_lon = np.pi * r / 180.0 * np.cos(np.radians(lat))
    return meters_per_lat, meters_per_lon

def axes_aspect_expander(extents, sz, pad_meters=100):
    """Returns the extents of a map that will fit the given extents
    with the given aspect ratio.
    """
    sz_extent = [extents[2] - extents[0], extents[3] - extents[1]]
    center = [(extents[0] + extents[2]) / 2, (extents[1] + extents[3]) / 2]
    meters_per_lat, meters_per_lon = get_meters_per_lat_lon(center[1])
    sz_extent[0] += pad_meters * 2 / meters_per_lon
    sz_extent[1] += pad_meters * 2 / meters_per_lat
    # make sure the aspect ratio of underlying map is correct
    if sz_extent[0] / sz_extent[1] > sz[0] / sz[1]:
        sz_extent[1] = sz_extent[0] * sz[1] / sz[0]
    else:
        sz_extent[0] = sz_extent[1] * sz[0] / sz[1]

    extents = [
        center[0] - sz_extent[0] / 2,
        center[1] - sz_extent[1] / 2,
        center[0] + sz_extent[0] / 2,
        center[1] + sz_extent[1] / 2,
    ]
    return extents

def map_auto_zoom(lon1, lon2) -> int:
    # Heuristic determination of zoom level
    #   we find z such that: 360/2^z ~ width/2
    #   i.e. roughly two tiles to cover the width
    lon1 = lon1 % 360
    lon2 = lon2 % 360
    width = lon2 - lon1
    if width < 0:
        # case when the range overlaps the longitude zero point.
        # e.g. 350 to 20 --> -330 + 360 = 30
        width += 360

    return round(np.log2(360 * 2.0) - np.log2(width / 1.5))

sources = {
    "World_Light_Gray_Base": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    "OSM": "http://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "World_Street_Map": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    "World_Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "USGSImageryTopo": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryTopo/MapServer/tile/{z}/{y}/{x}",
    "USGSImageryOnly": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
    "USGSTopo": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
    "NatGeo_World_Map": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
    "ESRI WorldImagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}