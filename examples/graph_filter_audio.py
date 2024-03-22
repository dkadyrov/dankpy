#%%
from dankpy import audio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.style.use("dankpy.styles.default")
#%%
a = audio.Audio("examples/dankpy.wav", sample_rate=16000)

ticks = np.arange(0, a.data.index.max()+1, a.data.index.max()/5)

fig, ax = a.plot_spectrogram(window_size=128, nfft=128, noverlap=64, nperseg=128, zmin=-100, zmax=-50, method="samples")
ax.set_ylim(0, 5000)
ax.set_xlim(0, a.data.index.max())
ax.set_xticks(ticks)
ax.set_yticks([0, 2500, 5000])
ax.set_title("Original Signal")

#%%
b = audio.Audio("examples/dankpy.wav", sample_rate=16000)

b.bandpass_filter(1500, 2500, order=10, overwrite=True)

fig, ax = b.plot_spectrogram(window_size=128, nfft=128, noverlap=64, nperseg=128, zmin=-100, zmax=-50, method="samples")
ax.set_title("Bandpass Filter")
ax.set_ylim(0, 5000)
ax.set_xlim(0, b.data.index.max())
ax.set_xticks(ticks)

#%%
l = audio.Audio("examples/dankpy.wav", sample_rate=16000)

l.lowpass_filter(2000, order=10, overwrite=True)

fig, ax = l.plot_spectrogram(window_size=128, nfft=128, noverlap=64, nperseg=128, zmin=-100, zmax=-50, method="samples")
ax.set_ylim(0, 5000)
ax.set_xlim(0, l.data.index.max())
ax.set_xticks(np.arange(0, l.data.index.max()+1, b.data.index.max()/5))

#%%
h = audio.Audio("examples/dankpy.wav", sample_rate=16000)

h.highpass_filter(2000, order=10, overwrite=True)

fig, ax = h.plot_spectrogram(window_size=128, nfft=128, noverlap=64, nperseg=128, zmin=-100, zmax=-50, method="samples")
ax.set_ylim(0, 5000)
ax.set_xlim(0, h.data.index.max())
ax.set_xticks(np.arange(0, h.data.index.max()+1, h.data.index.max()/5))

# %%
fig, ax = plt.subplots()
f, p = signal.welch(a.data.signal, fs=a.sample_rate, window="blackmanharris")
ax.plot(f, 10*np.log10(p), label="Original")

low_pass = a.lowpass_filter(2000, order=10)
f, p = signal.welch(low_pass, fs=a.sample_rate, window="blackmanharris")
ax.plot(f, 10*np.log10(p), label="Low Pass")

hi_pass = a.highpass_filter(2000, order=10)
f, p = signal.welch(hi_pass, fs=a.sample_rate, window="blackmanharris")
ax.plot(f, 10*np.log10(p), label="High Pass")

band_pass = a.bandpass_filter(3000, 5000, order=10)
f, p = signal.welch(band_pass, fs=a.sample_rate, window="blackmanharris")
ax.plot(f, 10*np.log10(p), label="Band Pass")
ax.set_xlim(0, 8000)
ax.set_ylim(-200, 0)
fig.legend(loc="upper right", ncols=4)
#%%
