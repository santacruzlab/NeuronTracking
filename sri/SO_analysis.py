#%% Imports and constants for UG analysis
# Run imports and constants for UG analysis. 
import random
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

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
SFC_FOLDER = os.path.join(OUTPUT_DATA_FOLDER, 'channel_sfc')

print(os.getcwd())

from utils import sessions as ses
from utils import sri_master as sri

DUMMY_NUMBER = 1e7

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

braz = None
airp = None

#%% Initialize tracking objects
# Initialize tracking objects for Airport and Brazos.

# airp = sri.Tracking('airp', 
#                     sessions=SESSIONS['airp'][0:5],
#                     rotation=ROTATION,
#                     save_folder=SAVE_FOLDER)

braz = sri.Tracking('braz', 
                    sessions=SESSIONS['braz'], 
                    rotation=ROTATION, 
                    save_folder=SAVE_FOLDER)

#%% Read LFP data for tracking objects
# Read LFP data for each session in the tracking objects.
# sri.read_lfp_later(airp)
sri.read_lfp_later(braz)

#%% SFC Function per unit
# SFC function per unit
def sfc_of_single_unit(subj: sri.Tracking, session: str, unit_code: str, channel: int, getNeighbors: bool = False, rand: bool = False):
    """
    Plot trial-averaged SFC for each session in a tracked neuron.
    
    cluster: usage like airp.useful_clusters.iloc[0]
    title: for the title and the saved filename
    rand: if this is to randomized aligned points.   
    
    """
    
    start_sec, end_sec = -1, 1

    bmi = subj.raw_data[session]
    ns2 = bmi.ns2file
    ind = bmi.index
    
    spike_times = bmi.pklfile['spks'].get(unit_code) # Spike times
    fs = 1000 # Sampling frequency

    lfp = ns2.getdata()['data'][channel] # Read LFP data

    # if lfp is None or len(lfp) == 0:
    #     raise ValueError(f"LFP data for session {session} and channel {channel} is empty or None.")
    
    # [Estimate firing rate] - upsampled to the same rate as LFP
    fr = sri.__calc_firing_rate(spike_times)
    fr_time  = np.arange(0, len(fr) / 20, 1 / 20)
    lfp_time = np.arange(0, len(lfp) / fs, 1 / fs)
    interp_func = interp1d(fr_time, fr, kind='linear', fill_value='extrapolate')
    fr = interp_func(lfp_time)
    
    # All non-error-clamped trials in the first block
    trial = ind[(ind['block_type']==1)&(ind['error_clamp']==0)] 
    directions = np.array(trial['direction'])
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

    # print the error and skip if any error occurs (e.g., no spike times, or not enough data points for the aligned period)
        
    return np.array(session_coherences), directions
#%% Save SFC dfs function
# Save SFC dfs function
def save_sfc_df(subj: sri.Tracking, channels = None):
    all_channels = np.array(subj.useful_channel)
    processed_channels = []

    with os.scandir(SFC_FOLDER) as entries: # find all channels already extracted and saved
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.pkl'):
                processed_channels.append(int(entry.name[-7:-4]))

    all_channels = np.setdiff1d(all_channels, processed_channels)
    if channels is None:
        channels = np.random.choice(all_channels, min(10, len(all_channels)), replace=False) # Randomly select 10 channels or all if less than 10
        print(f"{subj.subject} - Remaining channels to process: {len(all_channels)}")
        print(f"{subj.subject} - Selected channels: {[int(x) for x in channels]}")
    else:
        print(f"{subj.subject} - Channel selection overriden: {[int(x) for x in channels]}")

    for ch in channels:
        ch_df = subj.useful_df[subj.useful_df['channel'] == ch]
        ch_df = ch_df.dropna(subset=['cluster_ID']) # Drop rows where cluster_ID is NaN
        sfc_df = pd.DataFrame(columns=['session', 'date', 'channel', 'unit_code', 'coherogram', 'wf_cluster_ID'])

        for unit in range(len(ch_df)):
            unit_code = ch_df.iloc[unit]['unit_code']
            session = ch_df.iloc[unit]['session']
            channel = ch_df.iloc[unit]['channel']
            date = ch_df.iloc[unit]['date']
            cluster_ID = int(ch_df.iloc[unit]['cluster_ID'])

            print(f'Processing unit {unit_code} in session {session} on channel {channel}')
            try: 
                out = sfc_of_single_unit(subj, session, unit_code, channel)
            except Exception as e:
                print(f'Error occurred while processing unit {unit_code} in session {session}')
                continue
            
            sfc_df = pd.concat([sfc_df, pd.DataFrame({
                'session': [session],
                'date': [date],
                'channel': [channel],
                'unit_code': [unit_code],
                'coherogram': [out],
                'wf_cluster_ID': [cluster_ID]
            })], ignore_index=True)

        # Save the results for this channel
        ch_string = str(ch).zfill(3)  # Pad channel number with zeros to make it 3 digits
        save_path = os.path.join(SFC_FOLDER, f'{subj.subject}_sfc_ch{ch_string}.pkl')
        sfc_df.to_pickle(save_path)
        print(f'Saved SFC for channel {ch} to {save_path}')

for subj in [braz]:
    save_sfc_df(braz)
    

#%% Add direction to SFC dataframes
# Add direction to SFC dataframes
def add_direction_to_sfc_df(subj: sri.Tracking, channels = None):
    processed_channels = []

    with os.scandir(SFC_FOLDER) as entries: # find all channels already extracted and saved
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.pkl'):
                file_path = os.path.join(SFC_FOLDER, entry.name)
                ch_sfc_df = pd.read_pickle(file_path)
                if not 'direction' in ch_sfc_df.columns:
                    ch_sfc_df['direction'] = None
                    # add the direction column into the df
                    for i in range(len(ch_sfc_df)):
                        ses = ch_sfc_df.iloc[i]['session']
                        bmi = subj.raw_data[ses]
                        ind = bmi.index

                        trials = ind[(ind['block_type']==1)&(ind['error_clamp']==0)]
                        directions = np.array(trials['direction'])
                        ch_sfc_df.at[i, 'direction'] = directions

                    # overwrite the currently saved .pkl file with the new one that has 'direction' column
                    ch_sfc_df.to_pickle(file_path)
                
                    print(f"Replaced channel {int(entry.name[-7:-4])}")
                    processed_channels.append(int(entry.name[-7:-4]))

    if len(processed_channels) < 1:
        print("No modifications made.")

add_direction_to_sfc_df(subj = braz)

#%% Load SFC from file
# Load SFC from file
braz_sfc_df = pd.DataFrame(columns=['session', 'date', 'channel', 'unit_code', 'coherogram', 'wf_cluster_ID'])
with os.scandir(SFC_FOLDER) as entries: # find all channels already extracted and saved
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.pkl'):
                braz_sfc_df = pd.concat([braz_sfc_df, pd.read_pickle(os.path.join(SFC_FOLDER, entry.name))], ignore_index=True)

braz_useful_df = pd.read_pickle(os.path.join(OUTPUT_DATA_FOLDER, 'braz_useful_df.pkl'))
if braz is not None:
    braz.useful_df = braz.useful_df
    braz.sfc_df = braz_sfc_df
else:
    braz_useful_df = pd.read_pickle(os.path.join(OUTPUT_DATA_FOLDER, 'braz_useful_df.pkl'))

#%% calc cluster slopes function
# calc cluster slopes function
def calc_cluster_slope(sfc_df: pd.DataFrame, cIDs: list = None):

    sfc_df_filtered = sfc_df[sfc_df.groupby('wf_cluster_ID')['wf_cluster_ID'].transform('size') >= 3]
    
    if cIDs is None:
        cIDs = sfc_df_filtered['wf_cluster_ID'].unique()
        

    slope_dict = dict(cluster=[], channel=[], slope=[], p=[])

    for c in cIDs:
        c_df = sfc_df_filtered[sfc_df_filtered['wf_cluster_ID']==c].sort_values(by='date').reset_index()
        c_df['date_rel'] = None
        for i in range(len(c_df)):
            c_df.at[i, 'date'] = datetime.strptime(c_df.iloc[i]['date'], '%Y%m%d')
        
        c_df['date_rel'] = [(c_df['date'].iloc[i] - c_df['date'].min()).days for i in range(len(c_df))]

        result = stats.linregress(
            list(c_df['date_rel']), 
            list(np.mean(c_df['coherogram'].iloc[i][:,0][~np.isnan(c_df['coherogram'].iloc[i][:,0])]) for i in range(len(c_df)))
            )

        slope_dict['cluster'].append(c)
        slope_dict['channel'].append(c_df.iloc[0]['channel'])
        slope_dict['slope'].append(result.slope)
        slope_dict['p'].append(result.pvalue)
    
    return pd.DataFrame(slope_dict)

braz_slopes = calc_cluster_slope(braz_sfc_df)

#%% Violin plot function
# violin plot function
def fig_cluster_violin_plot(cID: int, sfc_df: pd.DataFrame):
    temp_sfc_df = sfc_df[braz_sfc_df['wf_cluster_ID']==cID].sort_values(by='date')

    plotting_df = pd.DataFrame(columns=['date_abs', 'coherence'])

    for i in range(len(temp_sfc_df)):
        plotting_df = pd.concat([plotting_df, pd.DataFrame({
            'date_abs': datetime.strptime(temp_sfc_df.iloc[i]['date'], '%Y%m%d'), 
            'coherence': temp_sfc_df.iloc[i]['coherogram'][:, 0]})], 
            ignore_index=True)

    plotting_df['date_rel'] = [(plotting_df['date_abs'].iloc[i] - plotting_df['date_abs'].min()).days for i in range(len(plotting_df))]

    means = plotting_df.groupby('date_rel')['coherence'].mean().reset_index()

    result = stats.linregress(list(means['date_rel']), list(means['coherence']))

    print(f"Channel: {temp_sfc_df.iloc[0]['channel']}")
    print(f"Cluster: {cID}")
    print(f"Slope: {result.slope:.4f}")
    print(f"P-value: {result.pvalue}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='date_rel', y='coherence', data=plotting_df, native_scale=True, inner=None, ax=ax)
    # plt.plot(means['date_rel'], means['coherence'], color='red', marker='o', linestyle='-', label='Mean Coherence')
    # plt.axline((0, result.intercept), slope = result.slope, color = 'black', label = f'Slope: {result.slope:.4f}')
    sns.regplot(x='date_rel', 
                y='coherence', 
                data=means, 
                ci=None, 
                line_kws={"linestyle": "--", "color": "black"}, 
                scatter_kws={"color": "black", "s": 50}, 
                label=f'Slope: {result.slope:.3f}')
    plt.title(f'Delta Band Neuron Coherence Over Days', fontsize=20)
    plt.xlabel('Days since 1st session', fontsize=20)
    plt.ylabel('Coherence', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.text(
    #     x = 0.95,
    #     y = 0.95,
    #     s = f"Slope: {result.slope:.4f}"
    # )
    plt.legend()
    plt.savefig(os.path.join(FIG_FOLDER, 'violin', f'[braz]_{cID}_violin.svg'))
    plt.show()

fig_cluster_violin_plot(cID=1080, sfc_df=braz_sfc_df)

#%% ANOVA function
# ANOVA function
def run_cluster_anova(sfc_df: pd.DataFrame, cID): 
    c_sfc_df = sfc_df[sfc_df['wf_cluster_ID'] == cID]
    cluster_coherences = [c_sfc_df.iloc[i]['coherogram'][:,0] for i in range(len(c_sfc_df))]
    f_statistic, p_value = stats.f_oneway(*cluster_coherences)

    print(f"Channel {c_sfc_df.iloc[0]['channel']}, Cluster {cID}")
    print(f"P-value: {p_value:.4e}")




useful_sfc_df = braz_sfc_df[braz_sfc_df.groupby('wf_cluster_ID')['wf_cluster_ID'].transform('size') >= 3]
clusters_to_analyze = useful_sfc_df['wf_cluster_ID'].unique()
for c in clusters_to_analyze:
    run_cluster_anova(sfc_df = useful_sfc_df, cID = c)
#%% Calc cluster sfc similarity function
# calc cluster sfc similarity function
def calc_cluster_sfc_similarity(subj: sri.Tracking = None, sfc_df: pd.DataFrame = None, cID_1: int = None, cID_2: int = None):
        """
        Calculate the similarity score for each pair of units within the same channels across all sessions.
        Here we use pearson correlation and Euclidean distance to estimate total similarity.
        The similarity data are stored in self.sim_df: pd.DataFrame.
        
        Each row will be the similarity results from one useful channel.
        There will be 4 columns: 'correlation', 'euclidean', 'unit', and 'total'.
        
        The 'unit' column for each row is a list of units in that channel.
        The other columns store matrices of similarity whose shapes are (m, m),
        where m is the number of units across all sessions in that channel. 
        """
        
        # [[sfc_sim_df]] - stores the total similarity matrix for each channel

        if subj is not None: 
            if sfc_df is None:
                sfc_df = subj.sfc_df
        else: 
            if sfc_df is None:
                raise ValueError("If 'subj' is not provided, 'sfc_df' must be provided.")
            
        if cID_1 is None or cID_2 is None:
            cID_1, cID_2 = np.random.choice(subj.sfc_df['cluster_ID'].unique(), 2, replace=False)


        # sfc_sim_df = pd.DataFrame(columns=['correlation', 'euclidean', 'unit']) - move this to other code running it

        c1_df = sfc_df[sfc_df['wf_cluster_ID']==cID_1].sort_values(by='date')
        c2_df = sfc_df[sfc_df['wf_cluster_ID']==cID_2].sort_values(by='date')

        sim_dict = dict() # stores dataframes for each metric
        sim_dict['clusters'] = (cID_1, cID_2)

        for metric in ['correlation', 'euclidean']:
            sim_temp = np.zeros((len(c1_df), len(c2_df))) # stores similarity

            for i in range(len(c1_df)):
                for j in range(len(c2_df)):
                    wf1 = c1_df['coherogram'].iloc[i][:, 0] # Use the first frequency band (Delta) for similarity calculation
                    wf2 = c2_df['coherogram'].iloc[j][:, 0] # Use the first frequency band (Delta) for similarity calculation

                    exclude_indices = np.where(np.isnan(wf1) | np.isnan(wf2))[0]

                    if len(exclude_indices) > 0:
                        wf1 = np.delete(wf1, exclude_indices)
                        wf2 = np.delete(wf2, exclude_indices)

                    sim_temp[i,j] = sri.Tracking.similarity(
                        wf1,
                        wf2,
                        metric
                    )

                    # print(f'[{metric}] Similarity between cluster {cID_1} (session {i}) and cluster {cID_2} (session {j}): {sim_temp[i,j]:.4f}')

            sim_temp = sri.Tracking.rescale(sim_temp, metric=metric)
            sim_dict[metric] = sim_temp

        sim_dict['total'] = (sim_dict['correlation'] + sim_dict['euclidean']) / 2
        return sim_dict

#%% Make stability figures
# Make stability figures
cID_1, cID_2 = 2107, 2553
# cID_1, cID_2 = 1096, 1098

sim_dict = calc_cluster_sfc_similarity(sfc_df = braz_sfc_df, cID_1 = cID_1, cID_2 = cID_1)
# sim_df = pd.DataFrame(sim_dict)

plt.figure(figsize=(10, 6))
sns.heatmap(sim_dict['total'], xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5], annot=True, cmap='Blues', vmin=0, vmax=1)
plt.title("Delta Band SFC Similarities", fontsize=20)
plt.xlabel(f"Tracked Session", fontsize=20)
plt.ylabel(f"Tracked Session", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(os.path.join(FIG_FOLDER, 'sim_matrix', f'[braz]_{cID_1}_{cID_1}_similarity.svg'))
plt.show()

# sim_dict = calc_cluster_sfc_similarity(sfc_df = braz_sfc_df, cID_1 = cID_2, cID_2 = cID_2)
# # sim_df = pd.DataFrame(sim_dict)

# plt.figure(figsize=(6, 5))
# sns.heatmap(sim_dict['total'], annot=True, cmap='Blues', vmin=0, vmax=1)
# plt.title("Cluster similarities")
# plt.xlabel(f"Cluster {cID_2} Sessions")
# plt.ylabel(f"Cluster {cID_2} Sessions")
# plt.savefig(os.path.join(FIG_FOLDER, 'sim_matrix', f'[braz]_{cID_2}_{cID_2}_similarity.svg'))
# plt.show()

# sim_dict = calc_cluster_sfc_similarity(sfc_df = braz_sfc_df, cID_1 = cID_1, cID_2 = cID_2)
# # sim_df = pd.DataFrame(sim_dict)

# plt.figure(figsize=(6, 5))
# sns.heatmap(sim_dict['total'], annot=True, cmap='Blues', vmin=0, vmax=1)
# plt.title("Cluster similarities")
# plt.xlabel(f"Cluster {cID_1} Sessions")
# plt.ylabel(f"Cluster {cID_2} Sessions")
# plt.savefig(os.path.join(FIG_FOLDER, 'sim_matrix', f'[braz]_{cID_1}_{cID_2}_similarity.svg'))
# plt.show()

# temp_sfc_df = braz_sfc_df[braz_sfc_df['wf_cluster_ID']==cID_2].sort_values(by='date')

# plotting_df = pd.DataFrame(columns=['date_abs', 'coherence'])

# for i in range(len(temp_sfc_df)):
#     plotting_df = pd.concat([plotting_df, pd.DataFrame({
#         'date_abs': datetime.strptime(temp_sfc_df.iloc[i]['date'], '%Y%m%d'), 
#         'coherence': temp_sfc_df.iloc[i]['coherogram'][:, 0]})], 
        
#         ignore_index=True)

# plotting_df['date_rel'] = [(plotting_df['date_abs'].iloc[i] - plotting_df['date_abs'].min()).days for i in range(len(plotting_df))]

# means = plotting_df.groupby('date_rel')['coherence'].mean().reset_index()

# result = stats.linregress(list(means['date_rel']), list(means['coherence']))

# print(f"Slope: {result.slope:.4f}")
# print(f"P-value: {result.pvalue}")

# sns.violinplot(x='date_rel', y='coherence', data=plotting_df, native_scale=True)
# # plt.plot(means['date_rel'], means['coherence'], color='red', marker='o', linestyle='-', label='Mean Coherence')
# sns.regplot(x='date_rel', y='coherence', data=means, scatter=True, ci=None, color='red', label='Linear Fit')
# plt.title(f'Cluster {cID_2} coherence over sessions')
# plt.xlabel('Days since first session')
# plt.ylabel('Coherence')
# plt.legend()
# plt.savefig(os.path.join(FIG_FOLDER, 'violin', f'[braz]_{cID_2}_violin.svg'))
# plt.show()

f_bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
c_f_bands = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

plt.figure(figsize=(12,4))

for i in range(len(f_bands)):
    x = np.arange(braz_sfc_df[braz_sfc_df['wf_cluster_ID'] == cID_1].iloc[0]['coherogram'].shape[0]) + 1
    y = braz_sfc_df[braz_sfc_df['wf_cluster_ID'] == cID_1].iloc[0]['coherogram'][:,i]
    plt.plot(x, y, color=c_f_bands[i], label=f'{f_bands[i]}')
    y_mean = np.mean(y[~np.isnan(y)])
    plt.axhline(y=y_mean, color=c_f_bands[i], linestyle='--')
    if i == 0:
        plt.text(plt.xlim()[1], y_mean, s=f'{y_mean:.2f}', color = c_f_bands[i], va='bottom', ha='right')
    
plt.ylim(0, 0.6)
plt.title('Log Coherence Over Trials')
plt.xlabel('Trial Number')
plt.ylabel('Log Coherence')
plt.legend(ncols = 5)
plt.savefig(os.path.join(FIG_FOLDER, 'line', f'[braz]_{cID_1}_line_1'))
plt.show()

plt.figure(figsize=(12,8))

sfc_means = np.zeros(len(f_bands))
sfc_std = np.zeros(len(f_bands))

for i in range(len(f_bands)):
    x = np.arange(braz_sfc_df[braz_sfc_df['wf_cluster_ID'] == cID_1].iloc[0]['coherogram'].shape[0]) + 1
    y = braz_sfc_df[braz_sfc_df['wf_cluster_ID'] == cID_1].iloc[0]['coherogram'][:,i]
    sfc_means[i] = np.mean(y[~np.isnan(y)])
    sfc_std[i] = np.std(y[~np.isnan(y)])
    
plt.bar(
    f_bands, 
    sfc_means, 
    yerr=sfc_std,
    error_kw=dict(ecolor="black", elinewidth=2, capsize=5, capthick=2, alpha=0.75)
    )
# plt.ylim(0, 0.6)
plt.title('Coherence by Frequency Band', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Frequency Band', fontsize=20)
plt.ylabel('Mean Coherence', fontsize=20)
plt.savefig(os.path.join(FIG_FOLDER, 'bar', f'[braz]_{cID_1}_freq_bands_1'))
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
 
make_figure(os.path.join(FIG_FOLDER, 'ex_coherence', 'sfc_schematic.svg'))
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

#%% UNFINISHED - thresholding for SFC functions
SAVEFIG = True
                
def get_sfc_threshold(subj: sri.Tracking, pct: float, PLOT: bool):

    # Obtain the amount of units for each channel in each session.
    # useful_channel is used to ensure there are at least 5 wavefroms
    subj_ch_session = np.zeros((len(subj.useful_channel), len(subj.sessions)))
    for i, ch in enumerate(subj.useful_channel):
        for j, session in enumerate(subj.sessions):
            subj_ch_session[i,j] = len(subj.useful_df[(subj.useful_df.session==session) & 
                                                      (subj.useful_df.channel==ch)])
    
    loc = np.zeros((len(subj.sessions), 2))
    loc[:,1] = np.arange(len(subj.sessions))
    
    threshold = []
    
    for it in range(200):
        
        print(f'{it} iteration')
        run = 0
        
        dist = []
        n_unit = []
        
        while len(dist) < 5000:
            
            run += 1        
            loc[:,0] = random.sample(range(len(subj.useful_channel)), len(subj.sessions))
            
            used_pair = []
            for l in loc:
                if subj_ch_session[int(l[0]), int(l[1])] != 0:
                    used_pair.append((int(subj.useful_channel[int(l[0])]), 
                                      subj.sessions[int(l[1])]))
            
            sim_temp = np.zeros((2, len(used_pair), len(used_pair)))
            for m, metric in enumerate(['correlation', 'euclidean']):
                for i, (ch1, ses1) in enumerate(used_pair):
                    for j, (ch2, ses2) in enumerate(used_pair):
                        
                        # transform sfc values per trial to a 1D array for similarity calculation
                        # do this for every frequency band

                        cID_1 = int(subj.useful_df[(subj.useful_df['channel']==ch1)&(subj.useful_df['session']==ses1)]['cluster_ID'])
                        cID_2 = int(subj.useful_df[(subj.useful_df['channel']==ch2)&(subj.useful_df['session']==ses2)]['cluster_ID'])

                        # Randomly pick one if > 1 unit in that channel.
                        cID_1 = cID_1.iloc[random.randint(0, len(cID_1)-1) if len(cID_1) > 1 else 0]
                        cID_2 = cID_2.iloc[random.randint(0, len(cID_2)-1) if len(cID_2) > 1 else 0]

                        idx_1 = subj.sfc_df[(subj.sfc_df['cluster_ID']==cID_1)]['sessions'].index(ses1) # find which unit in the cluster corresponds to the selected unit based on session and channel
                        idx_2 = subj.sfc_df[(subj.sfc_df['cluster_ID']==cID_2)]['sessions'].index(ses2) # find which unit in the cluster corresponds to the selected unit based on session and channel

                        wf1 = subj.sfc_df[(subj.sfc_df['cluster_ID']==cID_1)]['coherograms'][idx_1, :, 0] # Use the first frequency band (Delta) for similarity calculation
                        wf2 = subj.sfc_df[(subj.sfc_df['cluster_ID']==cID_2)]['coherograms'][idx_2, :, 0] # Use the first frequency band (Delta) for similarity calculation

                        sim_temp[m,i,j] = subj.similarity(wf1, wf2, metric)
                sim_temp[m] = subj.rescale(sim_temp[m], metric=metric)
            total_temp = sim_temp.mean(axis=0)
        
            dist += [float(total_temp[i, j]) 
                     for i in range(len(total_temp)) 
                     for j in range(i + 1, len(total_temp))]
            n_unit.append(len(total_temp))
            
            # if PLOT:
            #     plt.figure(figsize=(5,5)) # Fig size (5,5) for airport, (7,10) for brazos
            #     plt.pcolormesh(subj_ch_session,cmap='Greys', vmin=0, vmax=5)
            #     plt.yticks(np.arange(len(subj.useful_channel)), subj.useful_channel, fontsize=5)
            #     plt.xticks(np.arange(len(subj.sessions)), subj.sessions, rotation=90, fontsize=5)
            #     plt.xlabel('Session', fontsize=5)
            #     plt.ylabel('Channel', fontsize=5)
            #     plt.grid(True, color='k', lw=0.1)
                
            #     for l in loc:
            #         c = 'r' if subj_ch_session[int(l[0]), int(l[1])] == 0 else 'k'
            #         plt.scatter(l[1]+0.5, l[0]+0.5, marker='s', c=c, s=5)
            #     plt.show()
                
            #     plt.figure(figsize=(4,4))
            #     plt.pcolormesh(total_temp, cmap='Greys', vmax=0.7, vmin=0)
            #     plt.xticks([])
            #     plt.yticks([])
            #     if SAVEFIG:
            #         plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_null_total_similarity.svg'))
            #     plt.show()
                
        
        thres = np.percentile(dist, pct)
                
        if PLOT:
            plt.figure(figsize=(4,4))
            plt.hist(dist, bins=80, density=True)
            # plt.axvline(np.mean(dist), c='k', ls='--', label=f'Mean: {mean:.3f}')
            plt.axvline(thres, c='r', ls='--', label=f'Threshold = {thres:.3f}')
            plt.legend(frameon=False)
            plt.xlabel('Total similarity')
            plt.ylabel('Density')
            if SAVEFIG:
                plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_null_distribution_[{it}].svg'))
            plt.show()
            
        threshold.append(thres)
    
    plt.figure(figsize=(4,4))
    plt.hist(threshold, bins=30)
    plt.xlabel('Similarity threshold')
    plt.ylabel('Count')
    plt.axvline(np.mean(threshold), c='k', ls='--', lw=2)
    plt.title(f'Mean: {np.mean(threshold):.3f}, std: {np.std(threshold):.3f}')
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_thresholds.svg'))
    plt.show()

#%% SFC function
# SFC function
def sfc_of_tracked_neuron(subj: sri.Tracking, cluster: pd.Series, getNeighbors: bool = False, rand: bool = False):
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

#%% Run SFC function: select channels and only useful clusters
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
        h5file.attrs['sessions'] = np.array(subj.all_sessions) 
        h5file.attrs['channels'] = np.array(channels)
        for cluster in clusters.itertuples():
            # print progress through clusters
            print(f'Cluster {cluster.cluster_ID} on channel {cluster.channel}: {clusters.index.get_loc(cluster.Index)+1}/{len(clusters)}')
            out, ses = sfc_of_tracked_neuron(subj, cluster)
            
            path = f'{cluster.channel}/{cluster.cluster_ID}' 
            grp = h5file.require_group(path)
            if "coherograms" in grp:
                del grp["coherograms"]  # Delete existing dataset if it exists
            grp.create_dataset("coherograms", data=out)
            grp.attrs['sessions'] = np.array(ses)

# %% Visualize coherence over trials for each cluster in the tracking objects.
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


