#%%

import matplotlib.pyplot as plt
import dankpy
plt.style.use("dankpy.styles.latex")
# %%
import numpy as np 

x = np.linspace(0, 100, 1000)
y = np.sin(x)

plt.plot(x, y)
# %%
plt.savefig("test.png")
# %%
