#%%
from dankpy import audio
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dankpy.styles.mdpi")
#%%
fs = 5000.0

# Filter a noisy signal.
T = 0.05
nsamples = T * fs
t = np.arange(0, nsamples) / fs
a = 0.02
f0 = 600.0
x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
x += a * np.cos(2 * np.pi * f0 * t + .11)
x += 0.03 * np.cos(2 * np.pi * 2000 * t)

#%%
a = audio.Audio(audio=x, fs=fs)
#%%