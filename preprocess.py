#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:51:20 2025

@author: hungyunlu
"""


import os
import re
import tables
import scipy
import pickle
import pandas as pd
import numpy as np

root = '/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects'
BMI_FOLDER = os.path.join(root, 'bmi_python')
PROJECT_FOLDER = os.path.join(root, 'neuron_tracking')
SCRIPT_FOLDER = os.path.join(PROJECT_FOLDER, 'NeuronTracking')
NSX_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'blackrock')
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
FIG_FOLDER = os.path.join(PROJECT_FOLDER, 'plots')
os.chdir(NSX_FOLDER)
from brpylib import NsxFile
os.chdir(NS_FOLDER)
from nsfile import NSFile
os.chdir(PROJECT_FOLDER)
from sessions import AIRPORT_SESSIONS, BRAZOS_SESSIONS, AIRPORT_ROTATION, BRAZOS_ROTATION

# Constants
LETTER_CODE = {2.: 'a', 4.: 'b', 8.: 'c', 16.: 'd'}
ROTATION_CLR = {50: 'blue', 90: 'red', 270: 'green', 310: 'orange'}
SUBJECT = ['airp', 'braz']
SESSIONS = dict(zip(SUBJECT, [AIRPORT_SESSIONS, BRAZOS_SESSIONS]))
ROTATION = dict(zip(AIRPORT_SESSIONS + BRAZOS_SESSIONS, 
                    AIRPORT_ROTATION + BRAZOS_ROTATION))
SUBJECT_COLOR = dict(zip(SUBJECT, ['g', 'b']))
DUMMY_NUMBER = 1e7 
N_BOOTSTRAP = 1000
USEFUL_N_UNIT = 3
SAVEFIG = True



def _generate_nev_output(
        all_sessions: list[str] = AIRPORT_SESSIONS + BRAZOS_SESSIONS,
        count: bool = True):
    """
    Generate the nev_output.pkl file for each session.
    """
    
    def count_processed_files(sessions: list[str]):
        
        counts = 0
        for file in sessions:
            f = os.path.join(PROJECT_FOLDER, 'data', f'{file}_nev_output.pkl')
            if os.path.exists(f):
                counts += 1
    
        return counts
            

    for session in all_sessions:
        
        nev_output = os.path.join(PROJECT_FOLDER, 'data', f'{session}_nev_output.pkl')
        nev_input = os.path.join(PROJECT_FOLDER, 'data', f'{session}.nev')
    
        print(session)
        if not os.path.exists(nev_output):
            if os.path.exists(nev_input):
                print('Reading...')
                task = BMI(session)
                task.load_data()
                task.extract_waveform()
                nev_result = task.waveform
            
                with open(nev_output, 'wb') as f:
                    pickle.dump(nev_result.to_dict(), f)
                    print('Saved.')
                    
    if count:
        count_processed_files(all_sessions)
        
    return None


class BMI:
    
    def __init__(self, session: str):
        
        """
        General guidelines for navigating the files.
        
        [Data]
        There are multiple files for a single recording block with the same prefix.
        1. Files collected during the experiments: ns5, ns2, nev, pkl, and hdf.
        2. Files generated after the experiments: mat. 

        [Filename prefix]: SUBJYYYYMMDD_NN_teXXXX
        - SUBJ: first 4 characters of the subject names: airp for Airport, braz for Brazos
        - YYYYMMDD: date for the recording.
        - NN: the number of recording on that day that starts with 01 for a new session.
        - XXXX: incremental unique ID for each recording.

        [What do these files mean]
        1. ns5: Analog signal collected at 30 kHz from Ripple. Used to sync hdf and Ripple signal.
        2. ns2: LFP signal collected at 1 kHz.
        3. nev: Spiking information such as waveforms and spike times.
        4. hdf: Behavioral data generated from BMI3D.
        5. mat: Synced time stamps between Ripple and behaviors.
        6. pkl: Decoder files used in the actual BMI.

        [What do we need to start processing data]
        - We need the time stamps for spikes and behaviors, as well as waveforms.
        - The sole purpose of the ns5 files (which are usually large) is to generate the mat files.
          If mat files already exists, discard the ns5 files; if mat files do not exist, run Sync(session.ns5)
        - After extracting spike times and waveforms, the nev can be removed.
        - Therefore, we need nev, mat, and hdf files for spike processing; include ns2 for LFP processing.
        
        """
        
        self.session = session
        self.file_prefix = os.path.join(PROJECT_FOLDER, 'data', self.session)
        
        # [Initiate different data files]
        self.ns2file = None
        self.hdffile = None
        self.matfile = None
        self.pklfile = None
        self.ns5file = None
        self.nevfile = None
        
        ## Data
        print(f'[{self.session}] Read data')
        # [raw data]
        self.has_hdf = False # For behavior
        self.has_ns5 = False # For syncing 
        self.has_ns2 = False # For LFP
        self.has_nev = False # For waveform 
        self.has_decoder = False # For decoder
        # [generated data]
        self.has_mat = False
        self.has_nev_output = False
        self.check_files()
        
        ## Behavioral metrics can be parsed by parse_behavior()
        # [from hdf file]
        self.task_msg = None
        self.task_time = None
        self.decoder_state = None
        self.target_position = None
        self.spike_counts = None
        # [hdf index]
        self.hdf_reward = None
        self.hdf_wait = None
        self.hdf_holdcenter = None
        self.hdf_target = None
        self.hdf_holdtarget = None
        # [state time]
        self.time_reward = None
        self.time_wait = None
        self.time_holdcenter = None
        self.time_target = None
        self.time_holdtarget = None
        # [ripple numbers]
        self.rpp_holdcenter = None
        self.rpp_target = None
        self.rpp_holdtarget = None
        # [general metrics]
        self.n_total_trials = None
        self.avg_moving_time = None
        self.velocity_direction = None
        # [rotation angle and direct units]
        self.rotation_angle = None
        self.direct_units = None
        
            
    @staticmethod
    def hdf_to_sample(hdf_states, hdf_times):
        sample_number = np.zeros(hdf_states.size)
        hdf_rows = hdf_times['row_number'][0]
        ripple = hdf_times['ripple_samplenumber'][0]
        
        for i in range(len(hdf_states)):
            hdf_index = np.argmin(np.abs(hdf_rows - hdf_states[i]))
            if np.abs(hdf_rows[hdf_index] - hdf_states[i])==0:
                sample_number[i] = ripple[hdf_index]
            elif hdf_rows[hdf_index] > hdf_states[i]:
                hdf_row_diff = hdf_rows[hdf_index] - hdf_rows[hdf_index -1]  # distance of the interval of the two closest hdf_row_numbers
                m = (ripple[hdf_index]-ripple[hdf_index - 1])/hdf_row_diff
                b = ripple[hdf_index-1] - m*hdf_rows[hdf_index-1]
                sample_number[i] = int(m*hdf_states[i] + b)
            elif (hdf_rows[hdf_index] < hdf_states[i])&(hdf_index + 1 < len(hdf_rows)):
                hdf_row_diff = hdf_rows[hdf_index + 1] - hdf_rows[hdf_index]
                if (hdf_row_diff > 0):
                    m = (ripple[hdf_index + 1] - ripple[hdf_index])/hdf_row_diff
                    b = ripple[hdf_index] - m*hdf_rows[hdf_index]
                    sample_number[i] = int(m*hdf_states[i] + b)
                else:
                    sample_number[i] = ripple[hdf_index]
            else:
                sample_number[i] = ripple[hdf_index]
            
        return sample_number

            
            
    def check_files(self):
        """
        For a proper analysis to be done,
        the hdf, nev, mat, and pkl files are essential.
        The ns5 and ns2 are optional.
        """
        
        if os.path.exists(self.file_prefix + '.hdf'):
            self.has_hdf = True
        if os.path.exists(self.file_prefix + '.ns5'):
            self.has_ns5 = True
        if os.path.exists(self.file_prefix + '.ns2'):
            self.has_ns2 = True
        if os.path.exists(self.file_prefix + '.nev'):
            self.has_nev = True
        if os.path.exists(self.file_prefix + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix + '_KFDecoder.pkl'):
            self.has_decoder = True


    def load_data(self, file_types=None):
        """
        Load selected file types based on user input.
        
        :param file_types: List of file types to load (e.g., ["hdf", "ns5", "nev", "mat", "nev_output", "decoder"]).
                           If None, all available file types are loaded.
        """
        if file_types is None: # If None, load everything.
            file_types = ["hdf", "ns5", "nev", "mat", "nev_output", "decoder"]

        if "hdf" in file_types and self.has_hdf:
            self.hdffile = tables.open_file(self.file_prefix + ".hdf")

        if "ns5" in file_types and self.has_ns5:
            self.ns5file = NsxFile(self.file_prefix + ".ns5")

        if "nev" in file_types and self.has_nev:
            self.nevfile = NSFile(self.file_prefix + ".nev")
            self.spike_entities = [e for e in self.nevfile.get_entities() if e.entity_type == 3]

        if "mat" in file_types and self.has_mat:
            self.matfile = scipy.io.loadmat(self.file_prefix + "_syncHDF.mat")

        if "nev_output" in file_types and self.has_nev_output:
            with open(self.file_prefix + "_nev_output.pkl", "rb") as f:
                self.pklfile = pickle.load(f)

        if "decoder" in file_types and self.has_decoder:
            with open(self.file_prefix + "_KFDecoder.pkl", "rb") as f:
                os.chdir(BMI_FOLDER)  # Necessary to load decoder properly
                self.decfile = pickle.load(f)
                os.chdir(PROJECT_FOLDER)
                
    
    def read_lfp(self):
        
        if self.has_ns2:
            self.ns2file = NsxFile(self.file_prefix + '.ns2')
    
            
    @property
    def n_spike_entities(self):
        return len(self.spike_entities)
    
    
    def extract_waveform(self):
        """
        Extract waveform and spike times from the NEV file.     
        This function should be only used once if there is already a _nev_output.pkl file.
        
        See _generate_nev_output() for details.
        """
            
        def waveform_stats(waveform):
            waveform = np.asanyarray(waveform)
            ptt = [np.max(waveform[i,:]) - np.min(waveform[i,:]) for i in range(len(waveform))]
            mwf = np.mean(waveform, axis=0) # Mean waveform
            std = np.std(waveform, axis=0)
            return ptt, np.mean(std), std, mwf


        rd = dict(name=[], fr=[], ptt=[], std=[], stds=[], wf=[], spks=[])

        print('Processing...')
        for entity in self.spike_entities:
            elec = int(entity.label[4:]) # entity.label is "elecXX"
            
            arrays = dict(a=[],b=[],c=[],d=[])
            spike_times = dict(a=[],b=[],c=[],d=[])
            
            print(f'Electrode {elec}')
            
            for i in range(entity.item_count):
                sort_code = entity.get_segment_data(i)[2]
                if sort_code != 0: # make sure sort code is not zero
                    code = LETTER_CODE[sort_code]
                    arrays[code].append(entity.get_segment_data(i)[1]) # spike waveform info 
                    spike_times[code].append(entity.get_time_by_index(i)) # spike times info
                    
            for code in arrays.keys():
                if len(arrays[code]) > 0:
                   arr = np.array(arrays[code])
                   name = str(elec) + code
                   fr = len(arrays[code])/self.time_in_sec
                   ptt, std, stds, mwf = waveform_stats(arr)
                   
                   rd['name'].append(name)
                   rd['fr'].append(fr)
                   rd['ptt'].append(np.mean(ptt))
                   rd['std'].append(std)
                   rd['stds'].append(stds)
                   rd['wf'].append(mwf)
                   rd['spks'].append( np.array(spike_times[code]) )
                    
        rd = pd.DataFrame(rd)
        rd = rd.set_index('name')
        
        self.waveform = rd
        
        
    def parse_behavior(self):
        """
        The main function to parse all the behavioral metrics.
        
        Note that there are several prefixes regarding the time points.
        1. hdf_XXX: the hdf states in the hdf file. These are the indices of the hdf state instead of the actual time.
        2. time_XXX: the hdf time for each state collected at 60 Hz.
        3. rpp_XXX: the ripple time for each state at 30000 Hz.
        
        Note that the way we find each state is to traverse from reward.
        For a successful trial, it should be
        wait, premove, target, hold, targ_transition, target, hold, target_transition, reward.
        
        This is corresponding to
        wait, start, center, hold-center, check-center, target, hold-target, check-reward, reward.
        """
        print(f'[{self.session}] Parse behavior')
        
        # [Direct read-out]
        self.task_msg = self.hdffile.root.task_msgs[:]['msg']
        self.task_time = self.hdffile.root.task_msgs[:]['time']
        self.decoder_state = self.hdffile.root.task[:]['decoder_state']
        self.target_position = self.hdffile.root.task[:]['target']
        self.spike_counts = self.hdffile.root.task[:]['spike_counts']
        self.error_clamp = self.hdffile.root.task[:]['error_clamp'].flatten()
        self.block_type = self.hdffile.root.task[:]['block_type'].flatten()
        self.perturbation = self.hdffile.root.task[:]['pert'].flatten()

        # [HDF states]
        self.hdf_reward =  np.array([i for i,tsk in enumerate(self.task_msg) if tsk==b'reward']) 
        self.hdf_wait =       self.hdf_reward - 8
        self.hdf_holdcenter = self.hdf_reward -5
        self.hdf_target =     self.hdf_reward - 3
        self.hdf_holdtarget = self.hdf_reward -2
        
        # [HDF state time]
        self.time_reward =     self.task_time[self.hdf_reward]
        self.time_wait =       self.task_time[self.hdf_wait]
        self.time_holdcenter = self.task_time[self.hdf_holdcenter]
        self.time_target =     self.task_time[self.hdf_target]
        self.time_holdtarget = self.task_time[self.hdf_holdtarget]
        
        # [Ripple time]
        self.rpp_reward =     self.hdf_to_sample(self.time_reward, self.matfile)
        self.rpp_wait =       self.hdf_to_sample(self.time_wait, self.matfile)
        self.rpp_holdcenter = self.hdf_to_sample(self.time_holdcenter, self.matfile)
        self.rpp_target =     self.hdf_to_sample(self.time_target, self.matfile)
        self.rpp_holdtarget = self.hdf_to_sample(self.time_holdtarget, self.matfile)
        
        # [Get direct units]
        f = lambda x: str(x[0]) + LETTER_CODE[x[1]]
        self.direct_units = [f(unit) for unit in self.decfile.units]
        
        # [Get rotation angles]
        self.rotation_angle = ROTATION[self.session]
        
        # [Derived metrics]
        self.n_total_trials = len(self.hdf_reward)
        self.avg_moving_time = np.median((self.rpp_holdtarget - self.rpp_target))/30000
        vx, vy = self.decoder_state[:,3,0], self.decoder_state[:,5,0]
        self.velocity_direction = np.arctan2(vy,vx)
        
        
    def get_index(self):
        """
        Find trial information such as block type, error-clamp or not, perturbation or not,
        and the trial numbers.
        """
        pos_x = self.target_position[:,0]
        pos_y = self.target_position[:,2]
        angle = np.arctan2(pos_y,pos_x) * 180 / np.pi
        angle[angle<0] = angle[angle<0] + 360

        self.index = pd.DataFrame({
            'error_clamp': self.error_clamp[self.time_holdcenter],
            'block_type': self.block_type[self.time_holdcenter],
            'direction': angle[self.time_target],
            'perturbation': self.perturbation[self.time_holdcenter],
            'trial_number': np.arange(self.n_total_trials),
            })
        

def read_BMI_data(subj: str) -> tuple[list[str], np.ndarray]:
    """
    Read all the data recorded by the Grapevine NIP system. 
    To accelerate the process, all spikes were preprocessed using _generate_nev_output().
    
    :param subj: the name of the subject. Usually a four-letter word, e.g., airp, braz.
    
    :return unit_id: the session ID + channel ID + unit code. The list will be parsed later using unit_code_pattern().
    :return waveforms: a 2D array of the waveforms with a shape of (# waveforms, # sample per waveform).
    
    """
    unit_id = []
    waveforms = []    
    
    sessions = SESSIONS[subj]

    for session in sessions:
        task = BMI(session)
        task.load_data(['nev_output'])
        for key, val in task.pklfile['wf'].items():
            unit_id.append(session + '_' + key)
            waveforms.append(val)
    
    waveforms = np.array(waveforms)
    
    if len(unit_id) != waveforms.shape[0]:
        raise ValueError('Inconsistent shape between the metadata and the waveforms')
    
    return unit_id, waveforms


def BMI_data_pattern(unit: str):
    """
    Parse the unit_code.
    The input should be like "airp20221203_04_te123_63a".
    What we need is 1) the session (20221203), 2) the unit_code (63a), and 3) the channel number (63).
    """
    
    pattern = r'[a-z]{4}([0-9]{8})_[0-9]{2}_te[0-9]*_(([0-9]*)[a-d])'
    m = re.match(pattern, unit)
    date = m.group(1)
    unit_code = m.group(2)
    channel = int(m.group(3))
    
    return date, unit_code, channel


def make_metadata(subj: str, unit_id: list[str]) -> pd.DataFrame:

    result = pd.DataFrame(columns=['date','unit_code','channel'])
    for i in range(len(unit_id)):
        date, unit_code, channel = BMI_data_pattern(unit_id[i])
        
        if subj == 'airp': 
            # Special processing for airport only.
            if (256 < channel <= 288):
                # Sometimes, we used another front end (not in order) for a separate array.
                # That means, Channel 257 was actually Channel 33.
                # That means we are assigning channel 257 to 33.
                channel -= 224
            elif channel > 288:
                # For channel # > 288, it's recorded using V probes.
                # We discard these since they are acute recordings.
                # Assign these to 0 and filter them out later.
                channel = 0
            
        result.at[i, 'date'] =  date
        result.at[i, 'unit_code'] =  unit_code
        result.at[i, 'channel'] =  channel
        
    result = result[result['channel'] != 0] # Removing channel # = 0.

    return result


def save_data(subj, metadata: pd.DataFrame, waveforms: np.ndarray) -> None:
    
    if not os.path.exists(os.path.join(SCRIPT_FOLDER, 'data', subj)):
        os.makedirs(os.path.join(SCRIPT_FOLDER, 'data', subj))

    metadata.to_csv(os.path.join(SCRIPT_FOLDER, 'data', subj, 'waveforms_metadata.csv'), index=False)
    np.save(os.path.join(SCRIPT_FOLDER, 'data', subj, 'waveforms.npy'), waveforms)
    
    return 


def main():
    
    for subj in SUBJECT:
        unit_ids, waveforms = read_BMI_data(subj)
        metadata = make_metadata(subj, unit_ids)
        save_data(subj, metadata, waveforms)


if __name__ == '__main__':
    main()

