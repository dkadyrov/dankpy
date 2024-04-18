#%%
from dankpy import audio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.style.use("dankpy.styles.default")
#%%
a = audio.Audio("examples/dankpy.wav", sample_rate=16000)

ticks = np.arange(0, a.data.index.max()+1, a.data.index.max()/5)

#%%
fig, ax = a.plot_envelope(method="samples")
ax.set_xlim(0, a.data.index.max())
ax.set_xticks(ticks)
ax.set_title("Original")
#%%
# Envelope -> Bandpass Filter
a.envelope(overwrite=True)
a.bandpass_filter(1500, 2500, order=10, overwrite=True)

fig, ax = a.plot_spectrogram(
    window_size=128,
    nfft=128,
    noverlap=64,
    nperseg=128,
    zmin=-100,
    zmax=-50,
    method="samples",
)
ax.set_xlim(0, a.data.index.max())
ax.set_xticks(ticks)
ax.set_ylim(0, 8000)
ax.set_yticks([0, 8000])
ax.set_title(r"Envelope $\rightarrow$ Bandpass Filter")

fig, ax = a.plot_waveform(method="samples")
ax.set_xlim(0, a.data.index.max())
ax.set_xticks(ticks)
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.5, 1])
ax.set_title(r"Envelope $\rightarrow$ Bandpass Filter")
#%%
# Bandpass Filter -> Envelope
b = audio.Audio("examples/dankpy.wav", sample_rate=16000)

b.bandpass_filter(1500, 2500, order=10, overwrite=True)
b.envelope(overwrite=True)

fig, ax = b.plot_spectrogram(
    window_size=128,
    nfft=128,
    noverlap=64,
    nperseg=128,
    zmin=-100,
    zmax=-50,
    method="samples",
)
ax.set_xlim(0, a.data.index.max())
ax.set_xticks(ticks)
ax.set_ylim(0, 8000)
ax.set_yticks([0, 8000])
ax.set_title(r"Bandpass Filter $\rightarrow$ Envelope")

fig, ax = b.plot_waveform(method="samples")
ax.set_title(r"Bandpass Filter $\rightarrow$ Envelope")
ax.set_xlim(0, a.data.index.max())
ax.set_xticks(ticks)
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.5, 1])
#%%