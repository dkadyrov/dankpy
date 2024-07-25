#%%
from dankpy import maputils, mapping
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np 

plt.style.use("dankpy.styles.simple")

latitudes = [40.71427000]
longitudes = [-74.00597000]

extents = mapping.find_extents(latitudes=latitudes, longitudes=longitudes)
fig, ax = plt.subplots()

extents = mapping.axes_aspect_expander(extents, sz=ax.figure.get_size_inches(), pad_meters=2000)

mapurl = next(iter(mapping.sources.values()))
(ax0, axi) = mapping.plot_map(ax=ax, extents=extents, map_url=mapurl, z=16)
axi.set_interpolation("lanczos")
ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax.set_ylim(extents[1], extents[3])
ax.set_xlim(extents[0], extents[2])
ax.set_xticks(np.linspace(extents[0], extents[2], 4))
ax.set_yticks(np.linspace(extents[1], extents[3], 4))
#%%