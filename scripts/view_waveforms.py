import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SCRIPT_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
AIRP_FOLDER = os.path.join(DATA_FOLDER, 'airp')
BRAZ_FOLDER = os.path.join(DATA_FOLDER, 'braz')

os.chdir(AIRP_FOLDER)

# Load data
waveforms = np.load("waveforms.npy")
metadata = pd.read_csv("waveforms_metadata.csv")

print("Waveforms shape:", waveforms.shape)
# print("Metadata preview:")
# print(metadata.head())

# Plot logic
plt.figure(figsize=(10, 4))

if waveforms.ndim == 1:
    # Single waveform
    plt.plot(waveforms)
    plt.title("Waveform")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

elif waveforms.ndim == 2:
    # Multiple waveforms: plot the first one
    plt.plot(waveforms[0])
    plt.title("Waveform (First Entry)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
# Print all the waveforms
    for wf in waveforms:
        plt.plot(wf, alpha=0.3)

else:
    raise ValueError("Unsupported waveform array shape")

plt.tight_layout()
plt.show()