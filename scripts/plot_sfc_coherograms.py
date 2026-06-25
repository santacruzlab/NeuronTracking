#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
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
    f_start = np.argmin(np.abs(frequencies - 2)) # Start at the lowest frequency bin
    print(f_start)
    f_end = np.argmin(np.abs(frequencies - 30)) # End at 30
    colors = plt.cm.plasma(np.linspace(0, 1, 5))
    plt.figure(figsize=(20, 8))
    for session in sessions:
        for i in range(f_start, f_end + 1, 5): # Plot every 5 frequency bins together to avoid overcrowding the plot
            freq_bins_to_plot = range(i, i + 5) # Get the next 5 frequency bins
            for f in freq_bins_to_plot: # Plot every 5 frequency bins together to avoid overcrowding the plot
                plt.plot(range(n_time_bins), df[session, :, f], color=colors[f - i], alpha=0.5)
            plt.ylim(0, 0.2) # Set y-axis limits to 0 and 1 for better visualization of coherence values
            plt.title(f"SFC Coherogram for Session {session} (Frequency bins {frequencies[i]:.2f} to {frequencies[i + 4]:.2f})\n{os.path.basename(filepath)}")
            plt.xlabel("Time bins")
            plt.ylabel("Coherence")
            plt.tight_layout()
            # add a key
            plt.legend([f"Freq {frequencies[f]:.2f}" for f in freq_bins_to_plot], bbox_to_anchor=(1.05, 1), loc='upper left')
            os.chdir(r"C:\Users\sco595\Downloads\coherogram_graphs")
            plt.savefig(f"{os.path.basename(filepath).split('.')[0]}_s{session}_{frequencies[i]:.2f}_{frequencies[i + 4]:.2f}.png", bbox_inches='tight')
            plt.show()

if __name__ == "__main__":
    filepath = r"F:\ug_proj\coherences\coherograms\airp_c0.npy" 
    plot_file(filepath)
# %%
