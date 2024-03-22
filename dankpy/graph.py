from collections import OrderedDict
import matplotlib.pyplot as plt
from dankpy.color import Color, hex_to_rgb

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