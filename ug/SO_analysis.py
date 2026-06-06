#%% Imports and constants for UG analysis
# Run imports and constants for UG analysis. 
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from scipy import stats
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram

UG_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(UG_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

from utils import sessions as ses
from utils import ug_master as ug

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

airp_track = ug.Tracking('airp', 
                    sessions=SESSIONS['airp'][0:1],
                    rotation=ROTATION,
                    save_folder=SAVE_FOLDER)

# braz = ug.Tracking('braz', 
#                     sessions=SESSIONS['braz'][0], 
#                     rotation=ROTATION, 
#                     save_folder=SAVE_FOLDER)

#%% Read LFP data for tracking objects
# Read LFP data for each session in the tracking objects.
# ug.read_lfp_later(airp)
ug.read_lfp_later(braz)

#%% Generate trial-averaged SFC coherograms for useful clusters in the tracking objects.
# Generate SFC coherograms for each useful cluster in the tracking objects.

COHEROGRAM_FOLDER = r"F:\ug_proj\coherences\coherograms"
os.chdir(COHEROGRAM_FOLDER)

for subj in [braz]: # For each subject

    subj.useful_clusters['stable_sfc'] = None

    for k in range(len(subj.useful_clusters)): # Go through each useful cluster
        
        print(f'Cluster [{k}]')
        example = subj.useful_clusters.iloc[k]
        title = f'[{subj.subject}]_SFC_[{k}]'
        out, ses = ug.sfc_of_tracked_neuron(subj, example, title)
            
        if out.ndim > 1:
            np.savetxt(f'{subj.subject}_c{k}_ses.csv', ses.T, delimiter=',', fmt='%s')
            np.save(f'{subj.subject}_c{k}.npy', out)
            print(f'{subj.subject}_c{k}: SFC saved.')
            
        else:
            print(f'{subj.subject}_c{k}: Not enough data for SFC calculation.')

#%% Initialize Spacial objects for each subject.
# Initialize Spacing objects for each subject.
airp = ug.Spacial('airp',
                    sessions=SESSIONS['airp'],
                    rotation=ROTATION,
                    save_folder=SAVE_FOLDER)

ug.read_lfp_later(airp)
#%% Generate unit-specific SFC coherograms per trial. 
# Generate unit-specific SFC coherograms per trial for each useful cluster in the Spacial objects.
units = airp.
airp.unit_sfc(units=, rand=True)