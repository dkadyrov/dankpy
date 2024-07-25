#%%
import maptiles
from dankpy import maputils, mymaptiles
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.style.use("dankpy.styles.mdpi")

latitudes = [40.573555]
longitudes = [-73.861427]


# latitudes = [40.572, 40.573]
# longitudes = [-73.861, -73.865]

extents = maputils.find_extents(latitudes=latitudes, longitudes=longitudes)
fig, ax = plt.subplots()

extents = maputils.axes_aspect_expander(extents, sz=ax.figure.get_size_inches(), pad_meters=200)

mapurl = next(iter(maputils.sources.values()))
(ax0, axi) = mymaptiles.draw_map(ax=ax, bounds=extents, tile=mapurl, z=16)
axi.set_interpolation("lanczos")
ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax.set_ylim(extents[1], extents[3])
ax.set_xlim(extents[0], extents[2])
ax.set_xticks([extents[0], extents[2]])
ax.set_yticks([extents[1], extents[3]])
fig
#%%