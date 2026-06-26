#%% Imports and constants for UG analysis
# Run imports and constants for UG analysis. 
import random
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from scipy import cluster, stats
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram

SRI_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SRI_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
OUTPUT_DATA_FOLDER = os.path.join(DATA_FOLDER, 'output_data')
FIG_FOLDER = os.path.join(PROJECT_FOLDER, 'figs')

print(os.getcwd())

from utils import sessions as ses
from utils import sri_master as sri

SUBJECT = ['airp', 'braz']
SESSIONS = dict(zip(SUBJECT, [ses.AIRPORT_SESSIONS, ses.BRAZOS_SESSIONS]))
ROTATION = dict(zip(ses.AIRPORT_SESSIONS + ses.BRAZOS_SESSIONS, 
                    ses.AIRPORT_ROTATION + ses.BRAZOS_ROTATION))
STORAGE_PATHS = [r"X:\storage\rawdata", r"Y:\storage\rawdata", r"W:\storage\rawdata"] # santacruz2, santacruz1, santacruz3
SAVE_FOLDER = {}
for i, sessions in enumerate([ ses.AIRPORT_SESSIONS_1, ses.AIRPORT_SESSIONS_2, ses.BRAZOS_SESSIONS ]):
    for session in sessions:
        SAVE_FOLDER[session] = STORAGE_PATHS[i]

# plotting constants
SUBJECT_COLOR = dict(zip(SUBJECT, ['g', 'b'])) # airp is green, braz is blue.
ROTATION_CLR = {50: 'blue', 90: 'red', 270: 'green', 310: 'orange'}


#%% Initialize tracking objects
# Initialize tracking objects for Airport and Brazos.

# airp = sri.Tracking('airp', 
#                     sessions=SESSIONS['airp'][0:5],
#                     rotation=ROTATION,
#                     save_folder=SAVE_FOLDER)

braz = sri.Tracking('braz', 
                    sessions=SESSIONS['braz'][0:11], 
                    rotation=ROTATION, 
                    save_folder=SAVE_FOLDER)

#%% Read LFP data for tracking objects
# Read LFP data for each session in the tracking objects.
# sri.read_lfp_later(airp)
sri.read_lfp_later(braz)
#%% SFC function
def sfc_of_tracked_neuron(subj: sri.Tracking, cluster: pd.Series, rand: bool = False):
    """
    Plot trial-averaged SFC for each session in a tracked neuron.
    
    cluster: usage like airp.useful_clusters.iloc[0]
    title: for the title and the saved filename
    rand: if this is to randomized aligned points.   
    
    """
    
    start_sec, end_sec = -1, 1
    
    # plt.figure(figsize=(4,4)) # Each line is data from a neuron
    
    sessions = []

    cluster_coherences = [] # To store the coherences for each cluster across sessions
    
    for n in range(cluster.n_unit): # For each neuron in a cluster
        session, unit_code, channel = cluster.neuron.df[['session','unit_code','channel']].iloc[n]
        print(session)
        sessions.append(session)

        bmi = subj.raw_data[session]
        ns2 = bmi.ns2file
        ind = bmi.index
        
        try:
            spike_times = bmi.pklfile['spks'].get(unit_code) # Spike times
            fs = 1000 # Sampling frequency
    
            lfp = ns2.getdata()['data'][channel] # Read LFP data
            
            # [Estimate firing rate] - upsampled to the same rate as LFP
            fr = sri.__calc_firing_rate(spike_times)
            fr_time  = np.arange(0, len(fr) / 20, 1 / 20)
            lfp_time = np.arange(0, len(lfp) / fs, 1 / fs)
            interp_func = interp1d(fr_time, fr, kind='linear', fill_value='extrapolate')
            fr = interp_func(lfp_time)
            
            # All non-error-clamped trials in the first block
            trial = ind[(ind['block_type']==1)&(ind['error_clamp']==0)] 
            align_pts = np.array(bmi.rpp_target[trial['trial_number']] / 30, dtype=int)
            
            # This is for randomized align_pts
            if rand: 
                align_pts = (np.random.random(len(align_pts)) * align_pts[-1]).astype(int)
            
            # Variable structures
            N_trials = len(align_pts)

            N_pts = int(0.6*fs) # Use N_pts//2 data points to compute spectrum
            N_freqs = N_pts//2 + 1 # So we have N_freqs of frequency for the spectrums
            f = np.fft.rfftfreq(N_pts, 1/fs)
            
            # Divide the aligned period (from start_sec to end_sec around align_pts) into N_timesteps sections
            N_timesteps = 100
            t = np.linspace(start_sec, end_sec, N_timesteps)
            
            # Create a matrix to store all the detailed align points
            fine_align_pts = np.zeros((N_trials, N_timesteps), dtype=int)
            for t in range(N_trials):
                fine_align_pts[t] = np.linspace(align_pts[t] + start_sec * fs, 
                                                align_pts[t] + end_sec * fs, 
                                                N_timesteps, 
                                                dtype=int)
            
            coherogram = np.zeros((1, N_freqs))

            f_bands = [(1, 4), (5, 8), (9, 12), (13, 30), (31, 50)] # divide frequencies into bins for delta, theta, alpha, beta, gamma
            
            session_coherences = np.zeros((N_trials, len(f_bands)))
            
            for t in range(N_trials):
                Sxx = np.zeros(int(N_pts/2+1)) # Field spectrum.
                Syy = np.zeros(int(N_pts/2+1)) # Spike spectrum.
                Sxy = np.zeros(int(N_pts/2+1), dtype=complex) # Cross spectrum.

                for ts in range(N_timesteps):
                    pt = fine_align_pts[t, ts]

                    field_raw = lfp[pt-N_pts//2: pt+N_pts//2]
                    spike_raw = fr[pt-N_pts//2: pt+N_pts//2]
                    sxx, syy, sxy = sri.calc_spectrum(spike_raw, field_raw, fs=1000)

                    # Directly adding the averaged values
                    Sxx += (sxx / N_trials)
                    Syy += (syy / N_trials)
                    Sxy += (sxy / N_trials)

                cohr = abs(Sxy) / np.sqrt(Syy) / np.sqrt(Sxx)
                coherogram = cohr

                coherences = np.zeros(len(f_bands))
                coherences_std = np.zeros(len(f_bands))

                for i in range(len(f_bands)):
                    f_low, f_high = f_bands[i]
                    idx_low = np.argmin(np.abs(f - f_low))
                    idx_high = np.argmin(np.abs(f - f_high))
                    
                    coherence = coherogram[idx_low:idx_high].mean() # Average coherence in the frequency band
                    coherence_std = coherogram[idx_low:idx_high].std() / np.sqrt(coherogram.shape)

                    
                
                    coherences[i] = coherence
                    # coherences_std[i] = coherence_std

                session_coherences[t] = coherences

            cluster_coherences.append(session_coherences)

        # print the error and skip if any error occurs (e.g., no spike times, or not enough data points for the aligned period)
        except Exception as e:
            print(f'Error occurred while processing cluster {cluster.cluster_ID} in session {session}')
            pass
        
    return np.array(cluster_coherences), np.array(sessions)

# out, ses = sfc_of_tracked_neuron(braz, braz.useful_clusters.iloc[1])

#%% Channel-specific sfc for each cluster in the tracking objects.
for subj in [braz]: # For each subject
    # all_channels = list(subj.useful_channel)
    all_channels = list(subj.useful_clusters['channel'].unique().astype(int)) # Get unique channels from useful_clusters
    channels = random.sample(all_channels, min(5, len(all_channels))) # Randomly select 5 channels or all if less than 5
    print(f'{subj.subject} - Selected channels: {channels}')
    # look through channel to find all clusters for that channel
    # then run sfc_of_tracked_neuron for each cluster and save the results

    # look through self.useful_clusters, maybe self.useful_df for info
    clusters = subj.useful_clusters[subj.useful_clusters['channel'].isin(channels)]

    print(f'Processing {len(clusters)} clusters for subject {subj.subject}.')

    with h5py.File(os.path.join(OUTPUT_DATA_FOLDER, f'{subj.subject}_sfc.h5'), 'a') as h5file:
        for cluster in clusters.itertuples():
            # print progress through clusters
            print(f'Cluster {cluster.cluster_ID} on channel {cluster.channel}: {clusters.index.get_loc(cluster.Index)+1}/{len(clusters)}')
            out, ses = sfc_of_tracked_neuron(subj, cluster)
            
            path = f'{cluster.channel}/{cluster.cluster_ID}' 
            grp = h5file.require_group(path)
            if "coherograms" in grp:
                del grp["coherograms"]  # Delete existing dataset if it exists
            grp.create_dataset("coherograms", data=out)

            if "sessions" in grp:
                del grp["sessions"]  # Delete existing dataset if it exists
            grp.create_dataset("sessions", data=ses.astype('S'))  # Store sessions as bytes

# %%
os.chdir(FIG_FOLDER)

with h5py.File(os.path.join(OUTPUT_DATA_FOLDER, f'braz_sfc.h5'), 'r') as h5file:
    # h5file.visit(print)  # Print all paths in the HDF5 file
    for channel in h5file.keys():
        print(f'Channel: {channel}')
        for cluster in h5file[channel].keys():
            print(f'  Cluster: {cluster}')
            coherograms = h5file[f'{channel}/{cluster}/coherograms'][:]
            sessions = h5file[f'{channel}/{cluster}/sessions'][:]
            print(f'    Coherograms shape: {coherograms.shape}')
            print(f'    Sessions: {[s.decode("utf-8") for s in sessions]}')  # Decode bytes to string

            num_plots = coherograms.shape[0]

            f_bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
            c_f_bands = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
            
            for i in range(num_plots):
                x = np.arange(coherograms.shape[1]) + 1

                for j in range(coherograms.shape[2]):
                    y = coherograms[i, :, j]
                    axes[i].plot(x, y, color=c_f_bands[j], label=f'{f_bands[j]}')
                    y_mean = np.mean(y[~np.isnan(y)]);
                    axes[i].axhline(y=y_mean, color=c_f_bands[j], linestyle='--')
                    if j == 0:
                        axes[i].text(axes[i].get_xlim()[1], y_mean, s=f'{y_mean:.2f}', color = c_f_bands[j], va='bottom', ha='right')

                axes[i].set_ylim(0, 0.7)
                axes[i].set_title(f'Cluster {cluster} - Session {sessions[i].decode("utf-8")}')
                axes[i].set_xlabel('Trial Number')
                axes[i].legend()

            plt.tight_layout()
            plt.savefig(f'Channel_{channel}_{cluster}')
            
            
            

#%% Generate trial-averaged SFC coherograms for useful clusters in the tracking objects.
# Generate SFC coherograms for each useful cluster in the tracking objects.

COHEROGRAM_FOLDER = r"C:\Users\seano\Downloads\coherograms"
os.chdir(COHEROGRAM_FOLDER)

for subj in [braz]: # For each subject

    subj.useful_clusters['stable_sfc'] = None

    # for k in range(len(subj.useful_clusters)): # Go through each useful cluster
    for k in range(1): # do one cluster for testing    
        print(f'Cluster [{k}]')
        example = subj.useful_clusters.iloc[k]
        title = f'[{subj.subject}]_SFC_[{k}]'
        out, ses = sri.sfc_of_tracked_neuron(subj, example, title)
            
        if out.ndim > 1:
            np.savetxt(f'{subj.subject}_c{k}_ses.csv', ses.T, delimiter=',', fmt='%s')
            np.save(f'{subj.subject}_c{k}.npy', out)
            print(f'{subj.subject}_c{k}: SFC saved.')
            
        else:
            print(f'{subj.subject}_c{k}: Not enough data for SFC calculation.')


