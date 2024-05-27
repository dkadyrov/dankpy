from collections import OrderedDict
import matplotlib.pyplot as plt
from dankpy.color import Color, hex_to_rgb
import numpy as np

linestyle = [
    ("solid", "solid"),  # Same as (0, ()) or '-'
    ("dotted", "dotted"),  # Same as (0, (1, 1)) or ':'
    ("dashed", "dashed"),  # Same as '--'
    ("dashdot", "dashdot"),
    ("loosely dashed", (0, (5, 10))),
    ("densely dashed", (0, (5, 1))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("loosely dotted", (0, (1, 10))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


class Okabeito(object):
    def __init__(self):
        self.colors = OrderedDict()
        self.colors["lightblue"] = Color(hex="#56B4E9")
        self.colors["yellow"] = Color(hex="#E69F00")
        self.colors["orange"] = Color(hex="#F0E442")
        self.colors["green"] = Color(hex="#009E73")
        self.colors["purple"] = Color(hex="#CC79A7")
        self.colors["red"] = Color(hex="#D55E00")
        self.colors["blue"] = Color(hex="#0072B2")
        self.colors["black"] = Color(hex="#000000")

    def __getitem__(self, key):
        return self.colors[key]

    def __iter__(self):
        return iter(self.colors.values())

    def hex_list(self):
        return [c.hex for c in self.colors.values()]


Okabeito = Okabeito()


def legend_options(ax, loc="outside center left"):
    if loc == "outside center left":
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    elif loc == "upper upper left":
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))


def subplots(
    nrows=1, ncols=1, margin=5, header=5, subheight=8, subwidth=8, units="inches"
):  ## now it is in cm
    if units == "cm":
        m = margin / 2.54
        h = header / 2.54

        a = subheight / 2.54
        b = subwidth / 2.54

    else:
        m = margin
        h = header

        a = subheight
        b = subwidth

    ## Here I calculate the figure size that you need for these parameters, as OP asked for.

    width = ncols * (m + b + m)
    height = nrows * (h + a + h)

    axarr = np.empty((nrows, ncols), dtype=object)

    # print(height, width)

    fig = plt.figure(figsize=(width, height))

    for i in range(nrows):
        for j in range(ncols):
            axarr[i, j] = fig.add_axes(
                [
                    (m + j * (2 * m + b)) / width,
                    (height - (i + 1) * (2 * h + a) + h) / height,
                    b / width,
                    a / height,
                ]
            )

    if len(axarr) == 1:
        axarr = axarr[0][0]

    return fig, axarr


def set_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)
