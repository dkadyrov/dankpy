from zipfile import ZipFile, ZIP_DEFLATED
from lxml import html
import pandas as pd
import datetime
import simplekml

colors = {
    "aliceblue": "fffff8f0",
    "antiquewhite": "ffd7ebfa",
    "aqua": "ffffff00",
    "aquamarine": "ffd4ff7f",
    "azure": "fffffff0",
    "beige": "ffdcf5f5",
    "bisque": "ffc4e4ff",
    "black": "ff000000",
    "blanchedalmond": "ffcdebff",
    "blue": "ffff0000",
    "blueviolet": "ffe22b8a",
    "brown": "ff2a2aa5",
    "burlywood": "ff87b8de",
    "cadetblue": "ffa09e5f",
    "chartreuse": "ff00ff7f",
    "chocolate": "ff1e69d2",
    "coral": "ff507fff",
    "cornflowerblue": "ffed9564",
    "cornsilk": "ffdcf8ff",
    "crimson": "ff3c14dc",
    "cyan": "ffffff00",
    "darkblue": "ff8b0000",
    "darkcyan": "ff8b8b00",
    "darkgoldenrod": "ff0b86b8",
    "darkgray": "ffa9a9a9",
    "darkgrey": "ffa9a9a9",
    "darkgreen": "ff006400",
    "darkkhaki": "ff6bb7bd",
    "darkmagenta": "ff8b008b",
    "darkolivegreen": "ff2f6b55",
    "darkorange": "ff008cff",
    "darkorchid": "ffcc3299",
    "darkred": "ff00008b",
    "darksalmon": "ff7a96e9",
    "darkseagreen": "ff8fbc8f",
    "darkslateblue": "ff8b3d48",
    "darkslategray": "ff4f4f2f",
    "darkslategrey": "ff4f4f2f",
    "darkturquoise": "ffd1ce00",
    "darkviolet": "ffd30094",
    "deeppink": "ff9314ff",
    "deepskyblue": "ffffbf00",
    "dimgray": "ff696969",
    "dimgrey": "ff696969",
    "dodgerblue": "ffff901e",
    "firebrick": "ff2222b2",
    "floralwhite": "fff0faff",
    "forestgreen": "ff228b22",
    "fuchsia": "ffff00ff",
    "gainsboro": "ffdcdcdc",
    "ghostwhite": "fffff8f8",
    "gold": "ff00d7ff",
    "goldenrod": "ff20a5da",
    "gray": "ff808080",
    "grey": "ff808080",
    "green": "ff008000",
    "greenyellow": "ff2fffad",
    "honeydew": "fff0fff0",
    "hotpink": "ffb469ff",
    "indianred": "ff5c5ccd",
    "indigo": "ff82004b",
    "ivory": "fff0ffff",
    "khaki": "ff8ce6f0",
    "lavender": "fffae6e6",
    "lavenderblush": "fff5f0ff",
    "lawngreen": "ff00fc7c",
    "lemonchiffon": "ffcdfaff",
    "lightblue": "ffe6d8ad",
    "lightcoral": "ff8080f0",
    "lightcyan": "ffffffe0",
    "lightgoldenrodyellow": "ffd2fafa",
    "lightgray": "ffd3d3d3",
    "lightgrey": "ffd3d3d3",
    "lightgreen": "ff90ee90",
    "lightpink": "ffc1b6ff",
    "lightsalmon": "ff7aa0ff",
    "lightseagreen": "ffaab220",
    "lightskyblue": "ffface87",
    "lightslategray": "ff998877",
    "lightslategrey": "ff998877",
    "lightsteelblue": "ffdec4b0",
    "lightyellow": "ffe0ffff",
    "lime": "ff00ff00",
    "limegreen": "ff32cd32",
    "linen": "ffe6f0fa",
    "magenta": "ffff00ff",
    "maroon": "ff000080",
    "mediumaquamarine": "ffaacd66",
    "mediumblue": "ffcd0000",
    "mediumorchid": "ffd355ba",
    "mediumpurple": "ffd87093",
    "mediumseagreen": "ff71b33c",
    "mediumslateblue": "ffee687b",
    "mediumspringgreen": "ff9afa00",
    "mediumturquoise": "ffccd148",
    "mediumvioletred": "ff8515c7",
    "midnightblue": "ff701919",
    "mintcream": "fffafff5",
    "mistyrose": "ffe1e4ff",
    "moccasin": "ffb5e4ff",
    "navajowhite": "ffaddeff",
    "navy": "ff800000",
    "oldlace": "ffe6f5fd",
    "olive": "ff008080",
    "olivedrab": "ff238e6b",
    "orange": "ff00a5ff",
    "orangered": "ff0045ff",
    "orchid": "ffd670da",
    "palegoldenrod": "ffaae8ee",
    "palegreen": "ff98fb98",
    "paleturquoise": "ffeeeeaf",
    "palevioletred": "ff9370d8",
    "papayawhip": "ffd5efff",
    "peachpuff": "ffb9daff",
    "peru": "ff3f85cd",
    "pink": "ffcbc0ff",
    "plum": "ffdda0dd",
    "powderblue": "ffe6e0b0",
    "purple": "ff800080",
    "red": "ff0000ff",
    "rosybrown": "ff8f8fbc",
    "royalblue": "ffe16941",
    "saddlebrown": "ff13458b",
    "salmon": "ff7280fa",
    "sandybrown": "ff60a4f4",
    "seagreen": "ff578b2e",
    "seashell": "ffeef5ff",
    "sienna": "ff2d52a0",
    "silver": "ffc0c0c0",
    "skyblue": "ffebce87",
    "slateblue": "ffcd5a6a",
    "slategray": "ff908070",
    "slategrey": "ff908070",
    "snow": "fffafaff",
    "springgreen": "ff7fff00",
    "steelblue": "ffb48246",
    "tan": "ff8cb4d2",
    "teal": "ff808000",
    "thistle": "ffd8bfd8",
    "tomato": "ff4763ff",
    "turquoise": "ffd0e040",
    "violet": "ffee82ee",
    "wheat": "ffb3def5",
    "white": "ffffffff",
    "whitesmoke": "fff5f5f5",
    "yellow": "ff00ffff",
    "yellowgreen": "ff32cd9a",
}


def coordinates(lon, lat, alt):
    coordinates = []

    for i in range(len(lon)):
        coordinates.append((lon[i], lat[i], alt[i]))

    return coordinates


def read_kmz(filename, target):

    target = str(target)

    with ZipFile(filename, "r") as kmz:
        kml = kmz.open("doc.kml", "r").read()
    doc = html.fromstring(kml)

    dt = []
    lon = []
    lat = []
    alt = []
    img = []

    for folder in doc.cssselect("Document Folder"):
        for pm in folder.cssselect("Placemark"):
            if target in pm.cssselect("name")[0].text_content():
                tmp = pm.cssselect("track")
                if len(tmp):
                    tmp = tmp[0]
                    for desc in tmp.iterdescendants():
                        content = desc.text_content()
                        if desc.tag == "when":
                            dt.append(content)
                        elif desc.tag == "coord":
                            c = (
                                pm.cssselect("Point coordinates")[0]
                                .text_content()
                                .split(",")
                            )
                            lat.append(c[0])
                            lon.append(c[1])
                            alt.append(c[2])
                        else:
                            print("Skipping empty tag %s" % desc.tag)
                else:
                    c = pm.cssselect("Point coordinates")[0].text_content().split(",")
                    lon.append(c[0])
                    lat.append(c[1])
                    alt.append(c[2])

                    t = pm.cssselect("TimeStamp when")[0].text_content()
                    img.append(
                        pm.cssselect("description")[0]
                        .text_content()
                        .split('src="')[-1]
                        .split('" />')[0]
                    )

                    dt.append(datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ"))

    img_files = []
    with ZipFile(filename, "r") as kmz:
        for item in kmz.namelist():
            if item in img:
                img_files.append(kmz.read(item))

    data = pd.DataFrame(
        {
            "target": target,
            "datetime": dt,
            "longitude": lon,
            "latitude": lat,
            "altitude": alt,
            "img name": img,
            "img data": img_files,
        }
    )

    data["img name"] = data["img name"].replace("images/", "", regex=True)

    return data
