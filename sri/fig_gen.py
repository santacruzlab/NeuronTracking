#%%

"""
Simulated local field potential (LFP) over 2 seconds.

Real LFPs are dominated by low-frequency, 1/f-shaped power (more power
at low frequencies, falling off at higher ones) with occasional
oscillatory rhythms (e.g. theta ~6-10 Hz) riding on top. This builds
a signal that has that same qualitative character:
  1. Pink-ish (1/f) background noise
  2. A theta-band oscillation modulating on top
  3. A bit of high-frequency sensor noise
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

SRI_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SRI_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
OUTPUT_DATA_FOLDER = os.path.join(DATA_FOLDER, 'output_data')
FIG_FOLDER = os.path.join(PROJECT_FOLDER, 'figs')
SFC_FOLDER = os.path.join(OUTPUT_DATA_FOLDER, 'channel_sfc')

#%%
def pink_noise(n_samples, fs, seed=None):
    """Generate 1/f ('pink') noise by shaping white noise in the frequency domain."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    spectrum = np.fft.rfft(white)

    # scale amplitude by 1/sqrt(f) (avoid divide-by-zero at f=0)
    scale = np.ones_like(freqs)
    scale[1:] = 1 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=n_samples)
    return pink / np.std(pink)


def simulate_lfp(duration_s=2.0, fs=1000, theta_freq=8, theta_amp=0.6,
                  pink_amp=1.0, noise_amp=0.15, seed=None):
    """
    duration_s : length of the signal in seconds
    fs         : sampling rate in Hz
    theta_freq : frequency (Hz) of the oscillatory rhythm riding on top
    theta_amp  : amplitude of that rhythm
    pink_amp   : amplitude of the 1/f background
    noise_amp  : amplitude of additive white sensor noise
    """
    n_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    rng = np.random.default_rng(seed)

    background = pink_amp * pink_noise(n_samples, fs, seed=seed)

    # slowly wandering amplitude envelope so the rhythm isn't perfectly periodic
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi)))
    rhythm = theta_amp * envelope * np.sin(2 * np.pi * theta_freq * t)

    sensor_noise = noise_amp * rng.normal(0, 1, n_samples)

    lfp = background + rhythm + sensor_noise
    return t, lfp


t, lfp = simulate_lfp(seed=0)

plt.figure(figsize=(9, 3))
plt.plot(t, lfp, color='black', linewidth=0.9)
plt.xlabel('Time (s)')
plt.ylabel('LFP (a.u.)')
plt.title('Simulated LFP')
plt.xlim(0, 2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, 'ex_lfp', 'ex_lfp.svg'))
plt.show()
#%% Example coherence figure
def plot_panel(ax, kappa, phi0=0.0, n_spikes=8, n_cycles=4, seed=None,
               label=None, show_phi=False):
    rng = np.random.default_rng(seed)
 
    # --- Field trace (sine wave) ---
    t = np.linspace(0, n_cycles * 2 * np.pi, 1000)
    field = np.sin(t)
    ax.plot(t, field * 0.4 - 0.8, color='black', linewidth=1.8)
 
    # --- Spike phases ---
    # von Mises gives an angle in [-pi, pi]; kappa controls concentration
    # (kappa=0 -> uniform/random, kappa large -> tightly locked to phi0)
    phases = rng.vonmises(mu=phi0, kappa=kappa, size=n_spikes)
    # map phases into the time axis, spread across the cycles, keep sorted
    cycle_choices = np.sort(rng.choice(n_cycles, size=n_spikes, replace=True))
    spike_times = cycle_choices * 2 * np.pi + (phases % (2 * np.pi))
    spike_times = np.clip(spike_times, 0, n_cycles * 2 * np.pi)
 
    # --- Spike ticks ---
    for st in spike_times:
        ax.plot([st, st], [0.3, 1.1], color='black', linewidth=2.5)

    pad = 0.15

    x0, x1 = t[0] - pad, t[-1] + pad
    y0, y1 = 0.3 - pad, 1.1 + pad
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor='black', linewidth=1.2))
    
    x0, x1 = t[0] - pad, t[-1] + pad
    y0, y1 = -0.8 - 0.4 - pad, -0.8 + 0.4 + pad
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor='black', linewidth=1.2))
 
    # --- Labels ---
    if label:
        ax.text(np.mean(t), 1.6, label, fontsize=20, ha='center')
    if show_phi:
        ax.text(t[-1] * 0.85, 1.7, r'$\phi = 0\degree$', fontsize=16, va='center')
 
    ax.set_xlim(-0.3, n_cycles * 2 * np.pi + 0.3)
    ax.set_ylim(-1.8, 2.0)
    ax.axis('off')
 
 
def make_figure(save_path='sfc_schematic.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    # Left = low coherence -> small kappa (near-uniform phases)
    plot_panel(axes[0], kappa=0.05, n_spikes=8, seed=3, label='Low')
 
    # Right = high coherence -> large kappa (phases tightly locked to 0)
    plot_panel(axes[1], kappa=25, phi0=90.0, n_spikes=8, seed=2,
               label='High', show_phi=True)
 
    fig.text(0.12, 0.85, 'Coherence', fontsize=20, ha='center')
    fig.text(0.12, 0.62, 'Spikes', fontsize=20, ha='center')
    fig.text(0.12, 0.25, 'Field', fontsize=20, ha='center')
 
    plt.tight_layout(rect=[0.20, 0, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
 
make_figure(os.path.join(FIG_FOLDER, 'ex_coherence', 'ex_sfc.svg'))

#%% Example coherence figure with firing-rate row
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
 
 
def compute_firing_rate(spike_times, t, window_size=40, sigma=8):
    """
    Convert spike times into a smooth firing-rate trace.
 
    1. Bin spikes onto the same time grid as `t` (a binary spike train).
    2. Slide-average with a boxcar window of `window_size` samples
       (this is the "sliding average" step -- a simple moving-window
       spike count, like a coarse PSTH).
    3. Smooth that further with a Gaussian kernel (`sigma` in samples)
       to get a continuous-looking rate curve instead of a blocky one.
    """
    # binary spike train on the same grid as t
    spike_train = np.zeros_like(t)
    bin_idx = np.searchsorted(t, spike_times)
    bin_idx = np.clip(bin_idx, 0, len(t) - 1)
    spike_train[bin_idx] = 1
 
    # step 1: sliding (boxcar) average
    kernel = np.ones(window_size) / window_size
    slide_avg = np.convolve(spike_train, kernel, mode='same')
 
    # step 2: gaussian smoothing on top
    smoothed = gaussian_filter1d(slide_avg, sigma=sigma)
 
    return smoothed
 
 
def plot_panel(ax, kappa, phi0=0.0, n_spikes=8, n_cycles=4, seed=None,
               label=None, show_phi=False):
    rng = np.random.default_rng(seed)
 
    # --- Field trace (sine wave) ---
    t = np.linspace(0, n_cycles * 2 * np.pi, 1000)
    field = np.sin(t)
    ax.plot(t, field * 0.3 - 1.8, color='black', linewidth=1.8)
 
    # --- Spike phases ---
    # von Mises gives an angle in [-pi, pi]; kappa controls concentration
    # (kappa=0 -> uniform/random, kappa large -> tightly locked to phi0)
    phases = rng.vonmises(mu=phi0, kappa=kappa, size=n_spikes)
    # map phases into the time axis, spread across the cycles, keep sorted
    cycle_choices = np.sort(rng.choice(n_cycles, size=n_spikes, replace=True))
    spike_times = cycle_choices * 2 * np.pi + (phases % (2 * np.pi))
    spike_times = np.clip(spike_times, 0, n_cycles * 2 * np.pi)
 
    # --- Spike ticks ---
    for st in spike_times:
        ax.plot([st, st], [1.3, 2.1], color='black', linewidth=2.5)
 
    # --- Firing rate row (between spikes and field) ---
    rate = compute_firing_rate(spike_times, t)
    rate = rate / rate.max() if rate.max() > 0 else rate  # normalize to [0, 1]
    rate_y0, rate_height = -0.9, 0.9
    ax.plot(t, rate * rate_height + rate_y0, color='black', linewidth=1.8)
 
    pad = 0.15
 
    # box around spikes
    x0, x1 = t[0] - pad, t[-1] + pad
    y0, y1 = 1.3 - pad, 2.1 + pad
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor='black', linewidth=1.2))
 
    # box around firing rate
    y0, y1 = rate_y0 - pad, rate_y0 + rate_height + pad
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor='black', linewidth=1.2))
 
    # box around field
    y0, y1 = -1.8 - 0.3 - pad, -1.8 + 0.3 + pad
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor='black', linewidth=1.2))
 
    # --- Labels ---
    if label:
        ax.text(np.mean(t), 2.6, label, fontsize=20, ha='center')
    if show_phi:
        ax.text(t[-1] * 0.85, 2.7, r'$\phi = 0\degree$', fontsize=16, va='center')
 
    ax.set_xlim(-0.3, n_cycles * 2 * np.pi + 0.3)
    ax.set_ylim(-2.5, 3.0)
    ax.axis('off')
 
 
def make_figure(save_path='sfc_schematic.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
 
    # Left = low coherence -> small kappa (near-uniform phases)
    plot_panel(axes[0], kappa=0.05, n_spikes=8, seed=3, label='Low')
 
    # Right = high coherence -> large kappa (phases tightly locked to 0)
    plot_panel(axes[1], kappa=25, phi0=90.0, n_spikes=8, seed=2,
               label='High', show_phi=True)
 
    fig.text(0.12, 0.85, 'Coherence', fontsize=20, ha='center')
    fig.text(0.12, 0.65, 'Spikes', fontsize=20, ha='center')
    fig.text(0.12, 0.42, 'Firing rate', fontsize=20, ha='center')
    fig.text(0.12, 0.20, 'Field', fontsize=20, ha='center')
 
    plt.tight_layout(rect=[0.20, 0, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
 
make_figure(os.path.join(FIG_FOLDER, 'ex_coherence', 'ex_sfc.svg'))
#%% Example waveforms figure
def gaussian(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)
 
 
def spike_waveform(duration_ms=0.6, n_points=200, amplitude=1.0,
                    trough_time=0.2, trough_width=0.03,
                    peak_time=0.3, peak_width=0.07, peak_ratio=0.4):
    """Generate one smooth, noise-free spike waveform."""
    t = np.linspace(0, duration_ms, n_points)
    trough = -(amplitude + 0.35) * gaussian(t, trough_time, trough_width)
    rebound = amplitude * peak_ratio * gaussian(t, peak_time, peak_width)
    waveform = trough + rebound
    return t, waveform

variants = [
    dict(peak_ratio=0.4, color='#f000ff'),
    dict(peak_ratio=0.45, color="#aa00ff"),
    dict(peak_ratio=0.5, color="#4c00ff"),
    dict(peak_ratio=0.55, color="#0040ff"),
    dict(peak_ratio=0.6, color="#00b3ff"),
    dict(peak_ratio=0.65, color='#00f0ff'),
]
days = [0, 8, 14, 22, 37, 45]

fig, axes = plt.subplots(1, len(variants), figsize=(10, 2.5))
for ax, v, d in zip(axes, variants, days):
    t, wf = spike_waveform(peak_ratio=v['peak_ratio'])
    ax.plot(t, wf, v['color'])
    ax.text(0.3, .90, f'Day {d}', ha='center', fontsize=20)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-1.2, 0.7)
    ax.axis('off')

plt.suptitle('Tracked Neuron Waveforms', fontsize=22)
plt.tight_layout(w_pad=2)
plt.savefig(os.path.join(FIG_FOLDER, 'ex_waveforms', 'waveform_days.svg'), dpi=200, bbox_inches='tight')
plt.show()