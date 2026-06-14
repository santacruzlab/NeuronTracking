import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_file(filepath):
    df = np.load(filepath) # 3 dimentional matrix, dimensions are (n_sessions, n_time_bins, n_frequency_bins)
    sessions = range(df.shape[0])
    n_sessions = df.shape[0]
    n_time_bins = df.shape[1]
    n_freq_bins = df.shape[2]

    fs = 1000 # Sampling frequency, adjust as needed
    N_pts = int(0.6 * fs)
    frequencies = np.fft.rfftfreq(N_pts, 1/fs)
    # Generate unique colors for each frequency bin for a single session
    f_start = np.argmin(np.abs(frequencies)) # Start at the lowest frequency bin
    f_end = np.argmin(np.abs(frequencies - 10)) # End at 10
    freq_bins_to_plot = range(f_start, f_end + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, f_end - f_start + 1))
    plt.figure(figsize=(20, 8))

    for f in freq_bins_to_plot:
        plt.plot(range(n_time_bins), df[1, :, f], color=colors[f - f_start], alpha=0.5)
    plt.title(f"SFC Coherogram for Session {sessions[1]} (Frequency bins {f_start} to {f_end})\n{os.path.basename(filepath)}")
    plt.xlabel("Time bins")
    plt.ylabel("Coherence")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filepath = r"" #CSV Containing folder
    plot_file(filepath)