#%%
import matplotlib.pyplot as plt
import numpy as np 
from scipy import signal 
#%% 
# Create a signal
x = np.linspace(0, 1000, 10000)

low = 250
high = 750
n = 10

center = np.sqrt(low * high)
delta = (high - low)

#%%
lp = 1 / np.sqrt(1+(x/center)**(2*n))
hp = 1 - 1 / np.sqrt(1+(x/center)**(2*n))
bp = 1 / np.sqrt(1+((center**2-x**2)/(x*delta))**(2*n))
# bp2 = 1 / np.sqrt(1+((x-center)/delta)**(2*n))

fig, ax = plt.subplots()
ax.plot(x, lp, label="Low-pass")
ax.plot(x, hp, label="High-pass")
ax.plot(x, bp, label="Band-pass")
# ax.plot(x, bp2, label="Band-pass 2")

ax.vlines([low, high], 0, 1, linestyles='dashed')
ax.legend()
ax.set_xlim(0, 1000)
#%%