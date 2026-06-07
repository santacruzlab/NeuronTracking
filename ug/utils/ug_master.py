#%% imports
import os
import re
import sys
import glob
import scipy
import tables
import pickle
import random
import datetime

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram
from collections import Counter, defaultdict
from matplotlib_venn import venn2

print('Imported libraries!')

# rewritten with (hopefully) universal support.
UTILS_FOLDER = os.path.dirname(os.path.abspath(__file__))
UG_FOLDER = os.path.dirname(UTILS_FOLDER)
PROJECT_FOLDER = os.path.dirname(UG_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
INPUT_DATA_FOLDER = os.path.join(DATA_FOLDER, 'input_data')
OUTPUT_DATA_FOLDER = os.path.join(DATA_FOLDER, 'output_data')

GITHUB_FOLDER = os.path.dirname(PROJECT_FOLDER)
BMI_FOLDER = os.path.join(GITHUB_FOLDER, 'bmi_python')
NSX_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'blackrock')
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
FIG_FOLDER = os.path.join(PROJECT_FOLDER, 'plots')

sys.path.insert(0,BMI_FOLDER)
sys.path.insert(0,NS_FOLDER)

# os.chdir(BMI_FOLDER)
from riglib.blackrock.brpylib import NsxFile
# os.chdir(BMI_FOLDER)
# from nsfile import NSFile
from riglib.ripple.pyns.pyns.nsfile import NSFile
os.chdir(PROJECT_FOLDER)

print('Imported bmi_python libraries!')

#%% Main and stuff
# Constants
LETTER_CODE = {2.: 'a', 4.: 'b', 8.: 'c', 16.: 'd'}

DUMMY_NUMBER = 1e7 
N_BOOTSTRAP = 1000
USEFUL_N_UNIT = 3
SAVEFIG = True

def __calc_firing_rate(array):
    """
    
    For testing purpose, try the following script.
    
        subj = braz
        bmi = subj.raw_data[subj.sessions[0]]
        
        hdf = bmi.hdffile
        mat = bmi.matfile
        
        fr = []
        
        for key,value in bmi.pklfile['spks'].items():
            print(key)
            fr.append(__calc_firing_rate(value))
    
    """
    window_size = 0.5 # window size in seconds
    step_size = 0.05  # step size in seconds
    box_size = 20
    kernel_size = 1
    
    
    # Define time range for the analysis
    time_bins = np.arange(0, max(array), step_size)
    firing_rate = np.zeros_like(time_bins)
    
    # Compute firing rate for each window
    for i, t in enumerate(time_bins):
        count = np.sum((array >= t) & (array < t + window_size))
        firing_rate[i] = count / window_size  # Rate in Hz (spikes per second)
    
    firing_rate = gaussian(slide_avg(firing_rate, box_size), kernel_size)

    return firing_rate


def complete_sort(similarity, symbol, threshold = -0.0001):

    similarity = similarity.copy()
    if symbol is not None:
        symbol = symbol.copy()
    
    pt = 0
    while pt < len(similarity):
        ref = similarity[pt:, pt:]
        sort_index = ref[0].argsort()[::-1]
        result = ref[sort_index][:, sort_index]
        similarity[pt:, pt:] = result
        
        if symbol is not None:
            ref_sym = symbol[pt:]
            result_sym = ref_sym[sort_index]
            symbol[pt:] = result_sym

        
        try:
            anchor = np.argwhere(np.diff(result)<threshold)[0][0]+1
        except Exception:
            anchor = len(similarity)
        pt += anchor
        
    return similarity, symbol



def gaussian(data, sigma):
    """
    Apply a Gaussian filter to a 1D data array.
    data: np.array - The input data array to smooth.
    sigma: float - Standard deviation of the Gaussian kernel.
    """
    
    def gaussian_kernel(sigma):
        # Determine the size of the kernel (3 standard deviations on each side)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:  # Ensure odd size for symmetry
            kernel_size += 1
            
        # Generate a range of values centered at 0
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        # Calculate the Gaussian function
        kernel = np.exp(-x**2 / (2 * sigma**2))
        # Normalize the kernel so its sum is 1
        kernel = kernel / np.sum(kernel)
        return kernel
    
    # Convolve the data with the Gaussian kernel using 'same' mode
    kernel = gaussian_kernel(sigma)
    smoothed_data = np.convolve(data, kernel, mode='same')
    
    return smoothed_data


def slide_avg(array: np.ndarray, window: int) -> np.ndarray:
    """
    Slide average the array by the specified window.
    """
    avg = np.zeros(array.shape)
    for i in range(len(array)):
        if i < window:
            avg[i] = np.sum(array[:i+1],axis=0)/float(i+1)
        else:
            avg[i] = np.sum(array[i-window+1:i+1],axis=0)/float(window)
    return avg


def parse_unit(unit_name: str):
    """
    Parse the unit ID  >> airp20211202_04_te1598_2a
        session >> airp20211202_04_te1598
        subject >> airp
        date >> 20211202
        channel >> 2
        unit_code >> 2a
    
    """
    pattern = r'(([a-z]{4})([0-9]{8})_[0-9]{2}_te[0-9]*)_(([0-9]*)[a-d])'

    m = re.match(pattern, unit_name)
    session = m.group(1)
    subject = m.group(2)
    date = m.group(3)
    unit_code = m.group(4)
    channel = int(m.group(5))
    
    if subject == 'airp':
        # Special care needed for airp data because V probes were used in some sessions
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
    
    return session, date, channel, unit_code


def cosine_model(theta, MD, PD, meanFR):
    """
    Cosine function for firing rate fitting.
    theta: angles (in radians)
    MD: amplitude of cosine
    PD: phase shift (preferred direction)
    meanFR: baseline firing rate
    """
    return MD * np.cos(theta - PD) + meanFR


def minimal_PD_change(pd_array):
    """
    Modify the PD array so that neighboring PD changes are less than 180 degree.
    
    """
    pd_array = np.array(pd_array)
    for m in range(1,len(pd_array)):
        if pd_array[m] - pd_array[m-1] > 180:
            pd_array[m:] -= 360
        elif pd_array[m] - pd_array[m-1] < -180:
            pd_array[m:] += 360
    
    if np.any(pd_array > 360):
        pd_array  -= 360
    if np.any(pd_array < -360):
        pd_array  += 360
    
    return pd_array


def calc_PD_span(PD, pval: bool = True):
    """
    Calculate the PD span for a given set of preferred directions.
    Parameters:
    - PD: array-like, the preferred directions (in degrees).

    Returns:
    - PD_span: float, the span of the preferred directions (minimum angle covering all PDs).
    """
    
    PD = np.sort(PD)  # Sort the PDs
    dPD = np.diff(PD)  # Differences between consecutive PDs
    dPD = np.append(dPD, PD[0] + 360 - PD[-1])  # Wrap around for circular difference
    PD_span = 360 - np.max(dPD)  # Subtract the largest gap to get the span
    
    if pval:
        n_samples = 2000
        rand_PD = calc_random_span(len(PD), num_samples=n_samples)
        p_value = np.sum(rand_PD <= PD_span) / n_samples
        return PD_span, p_value
    
    return PD_span


def calc_random_span(num_PD, num_samples: int = 100000):
    """
    Compute the p-value for the observed PD span by comparing it to random samples.

    Parameters:
    - observed_PD_span: float, the observed PD span.
    - num_PD: int, the number of preferred directions in the observed data.
    - num_samples: int, the number of random samples to generate.

    Returns:
    - p_value: float, the p-value indicating the significance of the observed PD span.
    """
    random_spans = []
    for _ in range(num_samples):
        # Randomly sample num_PD directions from [0, 360)
        random_PD = np.random.uniform(0, 360, size=num_PD)
        # Compute the PD span for the random sample
        random_spans.append(calc_PD_span(random_PD, pval=False))
    
    # Calculate the proportion of random spans less than or equal to the observed span
    random_spans = np.array(random_spans)
    return random_spans


def read_lfp_later(subj):
    """
    Reading LFP signal after all bmis are initiated.
    This can speed up the data analysis pipeline especially because LFP is not required
    for some analyses.
    """
    for session in subj.raw_data.keys():
        bmi = subj.raw_data[session]
        bmi.check_files()
        if bmi.has_ns2:
            bmi.read_lfp()
        

def band_pass_filter(lfp_signal, fs=1000, low=12, high=30, order=3):
    
    nyq = 0.5 * fs    
    norm_low, norm_high = low/nyq, high/nyq    
    b, a = signal.butter(order, [norm_low, norm_high], btype='band')
    return signal.filtfilt(b, a, lfp_signal)


def calc_spectrum(spike, field, fs):
    """
    Calculate the coherence of spiking and field time sereis.
    
    INPUT
    - spike: array, the estimated firing rate upsampled to the same freuqency as field.
    - field: array, the band-passed or raw LFP.
    - fs: int, the sampling freuqency in Hz.
    
    Output
    - Sxx: the field spectrum
    - Syy: the spike spectrum
    - Sxy: the cross spectrum
    """
    
    assert len(spike) == len(field), ValueError('Field and spike should have the same length.')
    
    N_pts = len(spike)

    # [Preprocess the time series data]
    field_ts = (field-np.mean(field)) * np.hanning(N_pts) # Hanning tapeirng
    spike_ts = spike-np.mean(spike) # But not for the firing rates
    
    # C
    field = np.fft.rfft(field_ts)
    spike = np.fft.rfft(spike_ts)
    
    Sxx = np.real( field * np.conj(field) )
    Syy = np.real( spike * np.conj(spike) )
    Sxy = field * np.conj(spike)
    
    return Sxx, Syy, Sxy

        
class Sync:
    """
    
    General class used to extract non-neural information from Ripple files
    to synchronize with behavioral data saved in linked HDF files.  
    
    Analogsignals for digital events
    Naming convention
    0 - 3  : SMA 1 - 4
    4 - 27 : Pin 1 - 24
    28 - 29: Audio 1 - 2 
    Here we use Pin 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19 (based on Arduino setup)
    
    """
    
    pins_util = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]) + 3
    
    def __init__(self, filename: str):
        
        assert filename.endswith('ns5'), ValueError('Should be an ns5 file.')
        
        self.name = os.path.basename(filename)[:-4]
        self.path = os.path.dirname(filename)
        self.filename = filename
        self.output = None
        self.nsfile = None
        self._read_data()
        
        self.hdf_times = None
        self._extract_rows()
        
        scipy.io.savemat(
            self.name + '_syncHDF.mat',
            self.hdf_times)

    def _read_data(self):
        self.nsfile = NsxFile(self.filename)
        self.output = self.nsfile.getdata()

    def _extract_rows(self):
        """
        Create .mat synchronization file for synchronizing Ripple and 
        behavioral data (saved in .hdf file).
        """

        # Create dictionary to store synchronization data
        hdf_times = dict()
        hdf_times['row_number'] = []             # PyTable row number
        hdf_times['ripple_samplenumber'] = []    # Corresponding Ripple sample number
        hdf_times['ripple_dio_samplerate'] = []  # Sampling frequency of DIO signal recorded by Ripple system
        hdf_times['ripple_recording_start'] = [] # Ripple sample number when behavior recording begins

        signals = self.output['data']
        fs = self.output['samp_per_s']
        msgtype = signals[self.pins_util[8:], :]
        rownum = signals[self.pins_util[:8], :] 

        # Convert to 0 or 1 integers (0 ~ 5000 mV from the recordings)
        # rstart = (signals[22 + 3,:] > 2500).astype(int) # Never used.
        strobe = (signals[20 + 3,:] > 2500).astype(int)
        
        msgtype = np.flip(msgtype > 2500, axis=0)

        #rownum = np.flip((rownum > 2500).astype(int), axis = 0)
        rownum = np.flip(rownum > 2500, axis=0)
        #rownum = rownum.astype(int)

        # Convert the binary digits into arrays
        MSGTYPE = np.zeros(msgtype.shape[1])
        ROWNUMB = np.zeros(rownum.shape[1])
        for tp in range(MSGTYPE.shape[0]):
            MSGTYPE[tp] = int(''.join(str(int(i)) for i in msgtype[:,tp]), 2)
            ROWNUMB[tp] = int(''.join(str(int(i)) for i in rownum[:,tp]), 2)

        find_recording_start = np.ravel(np.nonzero(strobe))[0]
        find_data_rows = np.logical_and(np.ravel(np.equal(MSGTYPE,13)),np.ravel(np.greater(strobe,0)))  
        find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))

        rows = ROWNUMB[find_data_rows_ind]    # row numbers (mod 256)

        prev_row = rows[0]  # placeholder variable for previous row number
        counter = 0         # counter for number of cycles (i.e. number of times we wrap around from 255 to 0) in hdf row numbers

        for ind in range(1,len(rows)):
            row = rows[ind]
            cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
            counter += cycle
            rows[ind] = counter*256 + row
            prev_row = row    

        # Load data into dictionary
        hdf_times['row_number'] = rows
        hdf_times['ripple_samplenumber'] = find_data_rows_ind
        hdf_times['ripple_recording_start'] = find_recording_start
        hdf_times['ripple_dio_samplerate'] = fs
        
        self.hdf_times = hdf_times
        

class BMI:
    
    def __init__(self, session: str, save_folder: str, rotation: int):
        
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
        self.rotation = rotation
        self.file_prefix = os.path.join(PROJECT_FOLDER, 'data', self.session)
        self.file_prefix_hdf = os.path.join(INPUT_DATA_FOLDER, 'hdf', self.session)
        # print(self.file_prefix_hdf)
        self.file_prefix_ripple = os.path.join(save_folder, 'ripple', self.session)
        self.file_prefix_nev_output = os.path.join(INPUT_DATA_FOLDER, 'nev_output', self.session)
        self.file_prefix_mat = os.path.join(INPUT_DATA_FOLDER, 'mat', self.session)
        self.file_prefix_decoder = os.path.join(INPUT_DATA_FOLDER, 'decoder', self.session)

        
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
        self.load_data()
        
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
        
        if os.path.exists(self.file_prefix_hdf + '.hdf'):
            self.has_hdf = True
        if os.path.exists(self.file_prefix_ripple + '.ns5'):
            self.has_ns5 = True
        if os.path.exists(self.file_prefix_ripple + '.ns2'):
            self.has_ns2 = True
        if os.path.exists(self.file_prefix_ripple + '.nev'):
            self.has_nev = True
        if os.path.exists(self.file_prefix_mat + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix_nev_output + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix_decoder + '_KFDecoder.pkl'):
            self.has_decoder = True


    def load_data(self):
        
        if self.has_hdf:
            self.hdffile = tables.open_file(self.file_prefix_hdf + '.hdf')
            
        if self.has_ns5:
            self.ns5file = NsxFile(self.file_prefix_ripple + '.ns5')
                
        if self.has_nev:
            self.nevfile = NSFile(self.file_prefix_ripple + '.nev')
            self.spike_entities = [e for e in self.nevfile.get_entities() if e.entity_type==3]
            
        if self.has_mat:
            self.matfile = scipy.io.loadmat(self.file_prefix_mat + '_syncHDF.mat')
            
        if self.has_nev_output:
            with open(self.file_prefix_nev_output + '_nev_output.pkl', 'rb') as f:
                self.pklfile = pickle.load(f)
        
        if self.has_decoder:
            with open(self.file_prefix_decoder + '_KFDecoder.pkl', 'rb') as f:
                os.chdir(BMI_FOLDER) # Has to be done this way otherwise cannot open.
                self.decfile = pickle.load(f)
                os.chdir(PROJECT_FOLDER)
                
    
    def read_lfp(self):
        
        if self.has_ns2:
            self.ns2file = NsxFile(self.file_prefix_ripple + '.ns2')
    
            
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
        self.rotation_angle = self.rotation
        
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
        
        
class Matched:
    
    def __init__(self, units: list[str]):
        """
        The class for putative matched units.
        The input is a list of the unit codes.  
        
        In general, matched units should be accessed by self.clusters in the Tracking class.
        The matched units information can be read by
        
            self.clusters.neuron.iloc[cluster_ID].df
            
        where the df is made here by make_df(self)
        """
        self.units: list[str] = units
        self.n_unit = len(self.units)

        self.df: pd.DataFrame = None
        self.make_df()
                
        self.start_date: datetime.date = None
        self.end_date: datetime.date = None
        self.all_dates: list[datetime.date] = None
        self.duration = None
        self.get_dates()
        
        
    def __repr__(self):
        return self.units[0]
        

    def make_df(self):  
        
        # The first part is more general information from the unit ID.
        session = []
        date = []
        unit_code = []
        channel = []
        for i in range(self.n_unit):
            s, d, c, u = parse_unit(self.units[i])
            session.append(s)
            date.append(d)
            channel.append(c)
            unit_code.append(u)
            
        self.df = pd.DataFrame({
            'unit': self.units,
            'session': session,
            'date': date,
            'channel': channel,
            'unit_code': unit_code,
            }).sort_values(by='date')
        
        # Will be set through get_clusters() in class Tracking.
        self.df['rotation'] = 0.0
        self.df['is_direct'] = False
        
    def get_dates(self):
        date_dt = [datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])) for d in self.df.date]
        date_dt.sort()
        self.start_date = date_dt[0]
        self.end_date = date_dt[-1]
        self.all_dates = date_dt
        self.duration = abs(self.end_date - self.start_date).days + 1 # Count both ends


class Tracking:

    def __init__(self, subject: str, sessions: list[str], rotation: dict, save_folder: dict):
        """
        Our algorithm assumes that neurons cannot migrate far enough to be captured by a neighboring channel.
        This is a valid assumption for inter-electrode distance > ~500 microns.
        Therefore, we do not need to consider matching units across channels.
        
        We first collect all the units from all channels in all sessions, (>> raw_df)
        extract units that match our criteria (>> data_df) 
        and whose channels have more than 5 units across all sessions (>> useful_df).
        The motivation is that if the channel has less than 5 units, there is not much to be analyzed.
        
        We then calculate similarity for each pair of units in the same channel, (>> sim_df)
        and we use that to determine matched units (>> calc_matched_units).
        
        """

        self.subject = subject
        self.sessions = sessions
        self.dates = [s[4:12] for s in self.sessions]
        self.rotation = rotation
        self.save_folder = save_folder
        
        print(f'[{self.subject}] Start tracking')
        self.raw_data: BMI = None
        self.raw_df = None
        self.data_df = None
        self.grouped_channel = None
        self.useful_channel = None
        self.useful_df = None 
        self.read_sessions(parse=True)
        self.read_data()
        
        print(f'[{self.subject}] Calculating similarities')
        self.sim_df = None
        self.calc_similarity()
        
        print(f'[{self.subject}] Calculating similarity thresholds')
        self.threshold = None
        self.threshold_fussy = None
        self.calc_similarity_threshold()
        
        print(f'[{self.subject}] Obtaining cluster information')
        self.matched_units: list[str] = None
        self.clusters: pd.DataFrame = None
        self.useful_clusters: pd.DataFrame = None
        self.tuning_params: dict = dict()
        self.calc_matched_units()
        self.get_clusters()
        
        # print(f'[{self.subject}] Post-processing analysis')
        # self.calc_tuning()
        # self.calc_PD_metric()
        
        
    @staticmethod
    def calc_waveform_metrics(row) -> pd.Series:
        """        
        For the four important locations, we obtain the location (time sample index) and amplitude (uV).
        1. peak_before: Peak-before trough (index 0 or peak)
        2. trough: Trough
        3. peak_after: Peak-after trough
        4. inflect: Inflection point (if any)
        
        UPDATE 2024/11/14
        Simplified this part to include only peak_after_amp, trough_amp, 
        peak_before_amp, waveform, and inverted. Other metrics are not that useful.
        However, the calculation process is preserved in case it's needed someday.
        """
    
        waveform = row['wf']
        # LEN_WF = len(waveform) # Should be 52 samples
        
        # First determine if the waveform is inverted
        # If inverted, flip waveform. 
        # Then set trough and peak-after trough    
        min_amp = np.min(waveform)
        max_amp = np.max(waveform)
        
        if (min_loc := np.argmin(waveform)) < (max_loc := np.argmax(waveform)): # The normal case
            peak_after_amp = max_amp
            # peak_after_loc = max_loc
            trough_amp = min_amp
            trough_loc = min_loc
            inverted = False
        else: # Inverted case
            peak_after_amp = -min_amp
            # peak_after_loc = min_loc 
            trough_amp = -max_amp
            trough_loc = max_loc
            waveform = -waveform
            inverted = True
        
        # Peak before trough
        if trough_loc > 0:
            peak_before_amp = np.max(waveform[:trough_loc])
            # peak_before_loc = np.argmax(waveform[:trough_loc])
        else:
            peak_before_amp = 0
            # peak_before_loc = 0
        
        # # Inflection point
        # try:
        #     fittedcurve = UnivariateSpline(np.linspace(peak_after_loc, LEN_WF-1, LEN_WF-peak_after_loc), 
        #                                    waveform[peak_after_loc:])
        #     fittedcurve_2d = fittedcurve.derivative(n=2)
        #     fit_curve = fittedcurve_2d(np.linspace(peak_after_loc, LEN_WF-1, LEN_WF-peak_after_loc))
        #     inflect_loc = np.argmin(np.abs(fit_curve)) + peak_after_loc
        #     inflect_amp = waveform[inflect_loc]
        # except Exception:
        #     inflect_loc = None
        #     inflect_amp = None
            
        # # Delta amplitudes (absolute values)
        # amp_drop = peak_before_amp - trough_amp
        # amp_rise = peak_after_amp - trough_amp
        
        # # Amplitude ratios
        # ratio_before_trough =  - peak_before_amp / trough_amp
        # ratio_after_trough =  - peak_after_amp / trough_amp
        
        # # Durations
        # dur_before_trough = trough_loc - peak_before_loc
        # dur_after_trough = peak_after_loc - trough_loc
        
        metrics = {
            "peak_after_amp": peak_after_amp,
            # "peak_after_loc": peak_after_loc, 
            "trough_amp": trough_amp,
            # "trough_loc": trough_loc,
            "peak_before_amp": peak_before_amp,
            # "peak_before_loc": peak_before_loc,
            # "inflect_amp": inflect_amp,
            # "inflect_loc": inflect_loc,
            # "amp_drop": amp_drop,
            # "amp_rise": amp_rise,
            # "ratio_before_trough": ratio_before_trough,
            # "ratio_after_trough": ratio_after_trough,
            # "dur_before_trough": dur_before_trough,
            # "dur_after_trough": dur_after_trough,
            "waveform": waveform,
            "inverted": inverted,
        }
        return pd.Series(metrics)
    

    def read_sessions(self, parse: bool = False):
        """
        Read all the sessions for that subject.
        
        Build self.raw_data, which is a dictionary that can be accessed through
        
            self.raw_data[self.sessions[i]]
        """
        data = dict()
        for session in self.sessions:
            task = BMI(session, self.save_folder[session], self.rotation[session])
            # The parse parameter is False for default, but are set by read_full_data during init.
            # Separating BMI init and parsing data can save time if want to debug BMI class.
            if parse:
                task.parse_behavior()
                task.get_index()
            data[session] = task
            
        self.raw_data = data
        
    
    def read_data(self):
        """
        Read multiple data frames from the raw_data.
        """
        
        # [[raw_df]] - all units from all sessions
        dfs = []
        for datum in self.raw_data.values():
            df = pd.DataFrame(datum.pklfile, columns=['fr','ptt','wf'])
            df['unit'] = datum.session + '_' + df.index
            dfs.append(df)
        self.raw_df = pd.concat(dfs)
        
        session,date,unit_code,channel,is_direct,rotation, = [],[],[],[],[],[]
        for i in range(len(self.raw_df)):
            s, d, c, u = parse_unit(self.raw_df['unit'].iloc[i])
            try:
                direct = u in self.raw_data[s].direct_units
            except:
                print(f"[{s}] - skipped direct units.")
                direct = None
            rot = self.raw_data[s].rotation_angle
            session.append(s)
            date.append(d)
            channel.append(c)
            unit_code.append(u)
            is_direct.append(direct)
            rotation.append(rot)
            
        metrics_df = self.raw_df.apply(self.calc_waveform_metrics, axis=1)
        self.raw_df = pd.concat([self.raw_df, metrics_df], axis=1)
        self.raw_df['session'] = session
        self.raw_df['date'] = date
        self.raw_df['channel'] = channel
        self.raw_df['unit_code'] = unit_code
        self.raw_df['is_direct'] = is_direct
        self.raw_df['rotation'] = rotation
        self.raw_df = self.raw_df[self.raw_df['channel']!=0] # Excluding V probes channels, see parse_unit().
        self.raw_df['suggested'] = True
        self.raw_df.loc[~((self.raw_df['fr']>1) & (self.raw_df['ptt']>=80)), 'suggested'] = False
        
        # [[data_df]] - the units that fits the criteria (fr>1 and peak-to-trough >= 80)
        self.data_df = self.raw_df[self.raw_df['suggested']].reset_index(drop=True)
        
        # [[useful_df]] - the channels that have at least 5 units across all recordings.
        grouped_channel = self.data_df[['waveform','channel']].groupby('channel').count().sort_values(by='waveform',ascending=False)
        self.useful_channel = grouped_channel[grouped_channel['waveform']>=USEFUL_N_UNIT].index
        unuseful_channel = grouped_channel[grouped_channel['waveform']<USEFUL_N_UNIT].index
        self.useful_df = self.data_df.query(f'channel not in {list(unuseful_channel)}').drop(columns=['suggested','wf']).reset_index(drop=True)
        self.useful_df['cluster_ID'] = np.nan # Create a cluster_ID column which will be set in get_clusters(). 
        self.grouped_channel = self.useful_df[['waveform','channel']].groupby('channel').count().sort_values(by='waveform',ascending=False)


    @staticmethod
    def similarity(a, b, metric: str):
        """
        Transformed similarity metrics
        """
        same_array = np.all(a==b)
        match metric:
            case 'euclidean':
                if same_array:
                    return 0
                else:
                    return np.log(np.sqrt(np.sum((a-b)**2)))
            case 'correlation':
                if same_array:
                    return DUMMY_NUMBER
                else:
                    return np.arctanh(np.corrcoef(a, b)[0,1])

    @staticmethod
    def rescale(array, metric: str):
        """
        Rescaled similarity metrics
        """
        maxx = np.max(array[array!=DUMMY_NUMBER])
        minn = np.min(array[array!=DUMMY_NUMBER])
        
        match metric:
            case 'correlation':
                result = (array - minn) / (maxx - minn)
                
            case 'euclidean':
                # For euclidean distance, we would like the smallest number to be 1,
                # And the greatest number to be 0.
                result = (maxx - array) / (maxx - minn)
        
        result[result > 1] = 1
        result[result < 0] = 0
        result[np.isnan(result)] = 0
        return result
    
    
    def calc_similarity(self):
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
        
        # [[sim_df]] - stores the total similarity matrix for each channel
        self.sim_df = pd.DataFrame(columns=['correlation', 'euclidean', 'unit'], index=self.useful_channel)
        
        for ch in self.useful_channel:
            temp_dict = dict() # stores dataframes for each metric 
            ch_df = self.useful_df[self.useful_df['channel']==ch].sort_values(by='date')
            temp_dict['unit'] = ch_df['unit'].values

            for metric in ['correlation', 'euclidean']:
                sim_temp = np.zeros((len(ch_df), len(ch_df))) # stores similarity
                    
                # Compute for each pair of waveforms
                for i in range(len(ch_df)):
                    for j in range(len(ch_df)):
                        sim_temp[i,j] = self.similarity(
                            ch_df['waveform'].iloc[i],
                            ch_df['waveform'].iloc[j],
                            metric
                        )
                sim_temp = self.rescale(sim_temp, metric=metric)
                temp_dict[metric] = sim_temp
                
            self.sim_df.loc[ch] = temp_dict
    
        # total similarity is the avg of correlation and euclidean.
        self.sim_df['total'] = (
            self.sim_df['correlation'] + 
            self.sim_df['euclidean'] )/2
        
        
    def calc_similarity_threshold(self):
        """
        Threshold for determining whether units were from same neurons.
        Refer to the __get_threshold() function.
        Since the function takes a long time to run, we do not rerun it everytime.
        
        If the total similarity between unit A and B > self.threshold,
        then A and B are considered the same units.
        
        self.threshold_fussy is the std of the distribution.
        It's left here as a less strict criterion (see calc_matched_units() method).
        """
        
        match self.subject:
            case 'airp':
                self.threshold = 0.542
                self.threshold_fussy = 0.006
            case 'braz':
                self.threshold = 0.562
                self.threshold_fussy = 0.008
     
                
    @staticmethod
    def find_cluster(T, complete_label):
        """
        Helper function to implement the depth-first search algorithm.
        This DFS algorithm includes units where one of the similarity scores 
        with existing units is greater than the threshold.
        
        For example, if unit [A, B, C] are considered the same units, and we are testing unit D.
        if total similarity T(A,D) < threshold and T(B,D) < threshold, 
        but T(C,D) > threshold, then D will be included in the same cluster.
        
        This algorithm allows the waveforms to be gradually change over time.
        """
        visited = set() # Set to track visited nodes
        cluster_name = [] # List to hold each cluster
        cluster_index = [] # The indices of each unit
    
        for i in range(len(T)):
            if i not in visited:
                stack = [i] # Stack for DFS
                current_cluster_name = [] # List to collect nodes in the current cluster
                current_cluster_index = []
                
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        current_cluster_name.append(complete_label[node])
                        current_cluster_index.append(node)
                        # Add neighbors connected to 'node' by a 'True' entry in T
                        stack.extend([j for j in range(len(T)) 
                                          if T[node, j] and j not in visited])
                cluster_name.append(sorted(current_cluster_name))
                cluster_index.append(sorted(current_cluster_index))
        return cluster_name, cluster_index
    
                    
    def calc_matched_units(self, fussy: bool = False, verify: bool = True):
        """
        Obtain matched units using the total similarity matrices.
        """
        
        
        def find_repeated_indices(lst):
            """
            This function finds the indices that have repeated values. 
            For example, given [1,2,3,4,4,4,4], return [[3,4,5,6]];
            given [1,1,2,3,4,4,4] return [[0,1], [4,5,6]].
            
            The goal is to eliminate some false positives from the calc_matched_units function.
            Since the function is agnostic to the session number,
            it could group the units from the same channel in the same session together.
            For example, unit 22a, 22b, and 22c from the same session can be grouped together.
            
            The function finds the indices of these false positives.
            """
            index_dict = defaultdict(list)
            for i, val in enumerate(lst):
                index_dict[val].append(i)
                
            repeated_indices = [indices for indices in index_dict.values() if len(indices) > 1]
            return repeated_indices
        
        
        matched_units = []
        for u in range(len(self.sim_df)):
            similarity = self.sim_df.total.iloc[u]
            label = self.sim_df.unit.iloc[u]
            complete, complete_label = complete_sort(similarity, label)
            if fussy: # Higher tolerance
                matched = complete > self.threshold - self.threshold_fussy
            else:
                matched = complete > self.threshold
            potential_match_name, potential_match_ind = self.find_cluster(matched, complete_label)
            
            # For these potential matches, we will check whether there are conflicts.
            # Meaning different units in the same channel and session are grouped together.
            # These are false positives that should be removed.
            for k in range(len(potential_match_name)):
                name = potential_match_name[k] # Shorthand
                
                # Grab the dates for all labels
                counts = Counter([parse_unit(n)[1] for n in name]) 
                bug_dates = [key for key, value in counts.items() if value > 1]
                
                if len(bug_dates) > 0: # Then we will deal with this case
                
                    # Find where in that potential match (var: name) has repetition
                    ind_for_potential_match = find_repeated_indices([parse_unit(n)[0] for n in name])
                    
                    # Use a list to collect the indices to be removed
                    ind_to_remove = []
                    for j in range(len(ind_for_potential_match)): # There may be multiple repetition
                        ind_match = ind_for_potential_match[j] # Shorthand 
                        ind_similarity = np.array(potential_match_ind[k])[ind_match]
                        
                        # Keep the one that has the greatest avg similarity score 
                        # among the potential matched units.
                        keep = np.argmax(similarity[ind_similarity].mean(1))
                        ind_match.pop(keep)
                        
                        # The remaining will be removed.
                        ind_to_remove.extend(ind_match)
                        
                    # Collect them all and remove them at once
                    for remove in sorted(ind_to_remove, reverse=True):
                        name.pop(remove)
                        
                matched_units.append(potential_match_name[k])
            
        self.matched_units = matched_units
        
        if verify:
            for i in range(len(self.matched_units)):
                unit = self.matched_units[i]
                unique_dates = np.unique([parse_unit(u)[1] for u in unit])
                if len(unit) > len(unique_dates):
                    print(f'Matched units {i} has conflict units.')
                    print(f'Use self.matched_units[{i}] to debug.')
                
    
    def get_clusters(self):
        """
        Generate a dataframe (self.clusters) where each row is a group of matched units.
        """
        
        clusters = [Matched(i) for i in self.matched_units]
        n_units = [cluster.n_unit for cluster in clusters]
        duration = [cluster.duration for cluster in clusters]
        
        self.clusters = pd.DataFrame({
            'neuron': clusters,
            'n_unit': n_units,
            'duration': duration,
            })
        
        self.clusters['cluster_ID'] = np.arange(len(self.clusters)) # Assign an ID
        for c in range(len(self.clusters)):
            cluster_ID = self.clusters['cluster_ID'].iloc[c]
            cluster_df = self.clusters['neuron'].iloc[c].df
            
            # Assign the ID to useful_df so that we can grab rotation angles and direct/indirect units.
            for u in range(len(cluster_df)):
                unit = cluster_df['unit'].iloc[u]
                row = np.where(self.useful_df.unit==unit)[0]                
                self.useful_df.loc[row,'cluster_ID'] = cluster_ID 
                cluster_df.loc[u, 'rotation'] = self.useful_df.loc[row[0],'rotation']
                cluster_df.loc[u, 'is_direct'] = self.useful_df.loc[row[0],'is_direct']
                
        self.useful_clusters = self.clusters[self.clusters['n_unit'] >= USEFUL_N_UNIT].reset_index(drop=True)
                
                
    def calc_tuning(self, bootstrap: bool = False):
        
        self.tuning_params['block_type'] = 1
        self.tuning_params['error_clamp'] = 0
        self.tuning_params['duration'] = 0.3
                
        def generate_initial_guesses(fr, directions):
            # Step 1: Estimate parameters
            baseline_guess = np.mean(fr)
            amp_guess = (np.max(fr) - np.min(fr)) / 2
            pd_guess = directions[np.argmax(fr)]
            
            # Normalize phase shift to [0, 2*pi]
            pd_guess = pd_guess % (2 * np.pi)
            
            # Step 2: Generate perturbations
            initial_guesses = [
                [amp_guess, pd_guess, baseline_guess],
                [amp_guess * 1.1, pd_guess + np.pi / 8, baseline_guess * 1.1],
                [amp_guess * 0.9, pd_guess - np.pi / 8, baseline_guess * 0.9],
            ]
            
            return initial_guesses
        
        for i in range(len(self.useful_clusters)):
            cluster_ID = self.useful_clusters['cluster_ID'].iloc[i]
            print(f'Processing [{self.subject}] cluster {cluster_ID}')
            cluster_df = self.useful_clusters.neuron.iloc[i].df
            md, pds, meanfr, rss, significant = [],[],[],[],[]
            
            for u in range(len(cluster_df)):
                
                session = cluster_df.session.iloc[u] 
                unit_code = cluster_df.unit_code.iloc[u]
                
                bmi = self.raw_data[session]
                spikes = bmi.pklfile['spks'][unit_code]
                ind = bmi.index
                        
                # [Fit tuning curve]
                trial = ind[(ind['block_type']==1)&(ind['error_clamp']==0)] # All trials in the first block
                directions_rad = np.deg2rad(trial['direction'].values)
                align_pts = bmi.rpp_target[trial['trial_number']] / 30000
                
                duration = self.tuning_params['duration']
                firing_rates = np.array([np.sum((start < spikes) & (spikes<start+duration))/duration \
                                         for start in align_pts])     
                
                # Define multiple initial guesses
                initial_guesses = generate_initial_guesses(firing_rates, directions_rad)
            
                # Initialize default values in case fitting fails
                MD, PD, meanFR, pred_fr = None, None, None, None
            
                # Attempt multiple guesses
                for guess in initial_guesses:
                    try:
                        (MD, PD, meanFR), pcov = curve_fit(
                            cosine_model, directions_rad, firing_rates,
                            p0=guess,
                            bounds=([0, -2 * np.pi, -np.inf], [np.inf, 2 * np.pi, np.inf])
                        )
                        if PD < 0:
                            PD += 2 * np.pi
                        pred_fr = cosine_model(directions_rad, *(MD, PD, meanFR))
                        break  # Exit loop if fitting succeeds
                    except RuntimeError:
                        continue  # Try the next initial guess if fitting fails
        
                # [Bootstrap MD to check it's significantly > 0]
                if bootstrap and MD is not None:
                    md_bootstrap = []
                    for _ in range(N_BOOTSTRAP):
                        # Resample with replacement
                        indices = np.random.choice(np.arange(len(firing_rates)), len(firing_rates), replace=True)
                        directions_resampled = directions_rad[indices]
                        firing_rates_resampled = firing_rates[indices]
                        
                        # Fit model to resampled data
                        try:
                            (md_resampled, _, _), _ = curve_fit(cosine_model, directions_resampled, firing_rates_resampled, p0=guess)
                            md_bootstrap.append(md_resampled)
                        except RuntimeError:
                            continue  # Skip failed fits
                    
                    md_ci = np.percentile(md_bootstrap, [2.5, 97.5]) # Compute confidence intervals
                    
                    # Meaning the CI crosses 0, cannot reject null that it's not tuned.    
                    md_significant = True if (md_ci[0] * md_ci[1] > 0) & (md_ci[0] > 0) else False
                else:
                    md_significant = True # Assume they are significant
                
                md.append(MD)
                pds.append(np.rad2deg(PD))
                meanfr.append(meanFR)
                rss.append(np.sum((firing_rates - pred_fr)**2))
                significant.append(md_significant)
                
            cluster_df['MD'] = md
            cluster_df['PD'] = pds
            cluster_df['meanFR'] = meanfr
            cluster_df['rss'] = rss
            cluster_df['significant'] = significant
    

    def calc_PD_metric(self):
           
        self.useful_clusters[['PD_span',
                              'PD_span_pval',
                              'PD_rate',
                              'PD_intercept',
                              'PD_rate_pval',
                              'PD_intercept_pval',
                              'PD_r2',
                              'days']] = None
        
        for i in range(len(self.useful_clusters)):
            cluster_df = self.useful_clusters.neuron.iloc[i].df
            cluster_df =  cluster_df[cluster_df.significant]
            self.useful_clusters.at[i, 'n_unit'] = len(cluster_df)
            
            modified_PD = minimal_PD_change(cluster_df.PD)    
            date_dt = cluster_df.date.apply(lambda d: datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])))
            days_dt = (date_dt - date_dt.iloc[0]).apply(lambda x: x.days).to_list()
            
            X = sm.add_constant(days_dt)
            model = sm.OLS(modified_PD, X)
            result = model.fit()
            intercetp, rate = result.params
            inter_pval, rate_pval = result.pvalues
            
            PD_span, PD_pval = calc_PD_span(self.useful_clusters.neuron.iloc[i].df.PD)
            
            self.useful_clusters.at[i, 'PD_span'] = PD_span
            self.useful_clusters.at[i, 'PD_span_pval'] = PD_pval
            self.useful_clusters.at[i, 'PD_rate'] = rate
            self.useful_clusters.at[i, 'PD_rate_pvalue'] = rate_pval
            self.useful_clusters.at[i, 'PD_intercept'] = intercetp
            self.useful_clusters.at[i, 'PD_intercept_pvalue'] = inter_pval
            self.useful_clusters.at[i, 'PD_r2'] = result.rsquared
            self.useful_clusters.at[i, 'days'] = days_dt        
            
    
    @property
    def unit_counts(self):
        
        if self.useful_clusters is not None:
            a = len(self.useful_clusters)
            b = len(self.useful_clusters[
                    (self.useful_clusters['PD_r2'] > 0.6) & 
                    (self.useful_clusters['PD_rate_pvalue'] < 0.05)
                ])
            c = len(self.useful_clusters[(self.useful_clusters['PD_span_pval'] < 0.05)])
            d = len(self.useful_clusters[
                    (self.useful_clusters['PD_r2'] > 0.6)  & 
                    (self.useful_clusters['PD_rate_pvalue'] < 0.05) & 
                    (self.useful_clusters['PD_span_pval'] < 0.05)
                ])
            
            pos_pos = d # PD_span_pval < 0.05 and PD_r2 > 0.8
            pos_neg = c-d # PD_span_pval < 0.05 but PD_r2 <= 0.8
            neg_pos = b-d # PD_span_pval >= 0.05 but PD_r2 < 0.8
            neg_neg = a-b-c+d # PD_span_pval >= 0.05 and PD_r2 <= 0.8
            
            summary = {
                'n_total_units': len(self.raw_df),
                'n_suggested_units': len(self.data_df),
                'n_suggested_units_morethan_3': len(self.useful_df),
                'n_clusters': len(self.clusters),
                'n_clusters_units': int(self.clusters.n_unit.sum()),
                'n_useful_clusters': len(self.useful_clusters),
                'n_useful_clusters_units': int(self.useful_clusters.n_unit.sum()),
                'n_PD_metrics': np.array([[pos_pos, pos_neg],
                                          [neg_pos, neg_neg]]) }
            return summary
        else:
            print('Clusters not obtained yet.')
            return None
            
        
    def plot_unit_per_session(self, ax, subject_color):
        
        """
        Plot number of units per session
        Used in Fall 2024 meeting slides page 11.
        """
        y = self.useful_df[['channel','date']].groupby('date').count()
        x = y.index
        
        ax.plot(x, y, 'o--', 
                c=subject_color[self.subject],
                label=self.subject)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=5)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Session')
        ax.set_title('# units for each session')
        ax.legend(frameon=False)
        
        
    def plot_channel_stats(self, ax, subject_color):
        """
        Plot channel statistics
        Used in Fall 2024 meeting slides page 12.
        """
        ax.plot(range(1,len(self.grouped_channel)+1), 
                self.grouped_channel, 
                lw=0.5, 
                color=subject_color[self.subject],
                label=self.subject)
        ax.scatter(range(1,len(self.grouped_channel)+1), 
                   self.grouped_channel, 
                   marker='o', 
                   s=10, 
                   color=subject_color[self.subject])
        plt.legend(frameon=False)
        plt.xlabel('Channel ID sorted by number of waveforms')  
        plt.title('# waveforms in each channel across all sessions')
        plt.ylabel('Counts')
        
    
    def plot_waveform_of_channel(self):
        """
        Plot waveforms in each useful channel
        Used in Fall 2024 meeting slides page 12.
        """
        
        plt.figure(figsize=(10,10))
        for i in range(len(self.useful_channel)):
            ch = self.useful_df[self.useful_df['channel']==self.useful_channel[i]]
            plt.subplot(12, 10, i+1)
            for k in range(len(ch)):
                plt.plot(ch['waveform'].iloc[k], c='k', lw=0.5)
                plt.title(f'Ch {self.useful_channel[i]}, {len(ch)} wf', fontsize=8)
            plt.xticks([])
            plt.yticks([])
        plt.subplots_adjust(hspace=0.5, wspace=0.2)
        plt.show()
        
    
    def plot_waveform_of_all_matched_units(self, example: int = None):
        """
        Plot waveforms in each matched units.
        Used in Fall 2024 meeting slides page 12.
        """
        
        match self.subject:
            case 'airp':
                neuron = self.clusters[self.clusters.n_unit > 2]['neuron']
            case 'braz':
                neuron = self.clusters[self.clusters.n_unit > 4]['neuron'][:120]
        
        plt.figure(figsize=(10,10))
        for n in range(len(neuron)):
            plt.subplot(12, 10, n+1)
            for j, u in enumerate(neuron.iloc[n].units):
                plt.plot(self.useful_df[self.useful_df['unit']==u]['waveform'].iloc[0],
                         c=plt.cm.Greens(j / len(neuron.iloc[n].units)))
            plt.xticks([])
            plt.yticks([])
        plt.show()
        
        if example is not None:
            
            plt.figure(figsize=(3,3))
            for j, u in enumerate(neuron.iloc[example].units):
                plt.plot(self.useful_df[self.useful_df['unit']==u]['waveform'].iloc[0], 
                         c=plt.cm.brg(j / len(neuron.iloc[example].units)), label=u)
            plt.legend(frameon=False, bbox_to_anchor=(1,1), ncols=2)
            plt.xticks([])
            plt.ylabel('Voltage (uV)')
            plt.show()
        
    
    def plot_PD_significance(self):
        
        df = self.useful_clusters
        
        means = df[['PD_span','n_unit']].groupby(by='n_unit').mean()
        
        thres = []
        for l in means.index:
            res = calc_random_span(l, num_samples=10000)
            thres.append(np.percentile(res, 5))
            
        
        plt.figure(figsize=(5,4))
        
        plt.scatter(means.index, thres, marker='_', s=120, c='k', label='Significance level')
        
        for i in range(len(df)):
            
            plt.scatter(df['n_unit'].iloc[i], df['PD_span'].iloc[i], s=5,
                        c='r' if df['PD_span_pval'].iloc[i] < 0.05 else 'b')
            
        plt.xlabel('Number of units in a cluster')
        plt.ylabel('Span of preferred directions (degree)')
        plt.yticks(np.arange(0,361,45))
        plt.ylim([-5,360])
        plt.xticks(np.arange(3,np.max(means.index)+1), rotation=90)
        plt.title(f'{self.subject}')
        plt.legend(frameon=False, loc='upper left')
        if SAVEFIG:
            plt.savefig(os.path.join(FIG_FOLDER, f'[{self.subject}]_PD_span_significance.svg'))
        plt.show()
        
        
    def plot_masked_similarity_matrix(self, example: int = None):
        """
        Plot masked total similarity matrix.
        Dark color means surpassing the threshold (meaning from the same neuron.)
        Used in Fall 2024 meeting slides page 38.
        
        Examples are used in Fall 2024 meeting slides page 39-42.
        """
        plt.figure(figsize=(10,10))
        for i in range(len(self.sim_df)):
            plt.subplot(12, 10, i+1)
            similarity = self.sim_df.total.iloc[i]
            complete, _ = complete_sort(similarity, None)
            plt.pcolormesh(complete>self.threshold, cmap='Blues', vmax=1, vmin=0)
            plt.xticks([])
            plt.yticks([])
            
            for k in range(len(similarity)):
                plt.axvline(k, lw=0.05, c='k')
                plt.axhline(k, lw=0.05, c='k')
        plt.suptitle(f'[{self.subject}] masked total similarity matrix')
        plt.show()
        
        if example is not None:
        
            plt.figure(figsize=(4,4))
            similarity = self.sim_df.total.iloc[example]
            label = self.sim_df.unit.iloc[example]
            sym_map = dict(zip(label, range(len(label))))
            complete, complete_label = complete_sort(similarity, label)
            
            plt.pcolormesh(complete>self.threshold, cmap='Blues', vmax=1, vmin=0)
            plt.xticks(np.arange(len(label))+0.5, list(map(lambda x:sym_map[x], complete_label)), fontsize=10)
            plt.yticks(np.arange(len(label))+0.5, list(map(lambda x:sym_map[x], complete_label)), fontsize=10)
            plt.xlabel('Coded unit ID',fontsize=10)
            plt.ylabel('Coded unit ID',fontsize=10)
            for k in range(len(similarity)):
                plt.axvline(k, lw=0.5, c='k')
                plt.axhline(k, lw=0.5, c='k')
            plt.show()

            # [This plotting method is obsolete, see __plot_matched_units_waveform()]
            # plt.figure(figsize=(5,5))
            # # Show individual waveforms in that specific channel
            # for i, lab in enumerate(label):
            #     c = plt.cm.hsv(i/len(label))
            #     wf = self.useful_df[self.useful_df['unit']==lab]['waveform'].iloc[0]
            #     plt.plot(wf, label=sym_map[lab], c=c)

            # plt.legend(frameon=False, loc='lower right', ncols=2 if len(label) > 7 else 1)
            # plt.xticks([])
            # plt.show()
            
    
    def plot_cluster_PD_rate(self, cluster_ID, rotation_color):
        
        df = self.useful_clusters.query(f'cluster_ID == {cluster_ID}').iloc[0]
        
        rate, intercept, r2, days = df[['PD_rate','PD_intercept','PD_r2','days']]
        pvals = df[['PD_intercept_pvalue','PD_rate_pvalue']].values
        neuron = df.neuron
        PD = minimal_PD_change(neuron.df.PD.copy())
        
        plt.figure(figsize=(5,5))
        plt.scatter(days, PD, c=list(map(lambda x: rotation_color[x], neuron.df.rotation)))
        plt.plot(days, np.array(days) * rate + intercept, c='k', ls='--', lw=2)
        plt.title(f'{neuron} #{cluster_ID}\nR2: {r2:.3f}, slope: {rate:.3f}, slope p-val: {pvals[1]:.3f}')
        plt.xlabel('Days')
        plt.ylabel('Preferred directions (deg)')
        plt.show()
            
    
    def plot_PD_each_block(self, PLOT: bool = False):
        """
        Fit tuning to each block within a cluster.
        Plot the PD across blocks for all units.
        
        Deprecated since we will be focusing on baseline blocks only.        
        """
        def generate_initial_guesses(fr, directions):
            # Step 1: Estimate parameters
            baseline_guess = np.mean(fr)
            amp_guess = (np.max(fr) - np.min(fr)) / 2
            pd_guess = directions[np.argmax(fr)]
            
            # Normalize phase shift to [0, 2*pi]
            pd_guess = pd_guess % (2 * np.pi)
            
            # Step 2: Generate perturbations
            initial_guesses = [
                [amp_guess, pd_guess, baseline_guess],
                [amp_guess * 1.1, pd_guess + np.pi / 8, baseline_guess * 1.1],
                [amp_guess * 0.9, pd_guess - np.pi / 8, baseline_guess * 0.9],
            ]
            
            return initial_guesses
        
        for cluster_ID in range(len(self.useful_clusters)):
            print(cluster_ID)
            error_clamp = 0
            duration = 0.3
            
            cluster_df = self.useful_clusters.neuron.iloc[cluster_ID].df
            
            
            tuning_df = []
            
            for u in range(len(cluster_df)):
                
                session = cluster_df.session.iloc[u] 
                unit_code = cluster_df.unit_code.iloc[u]
                
                bmi = self.raw_data[session]
                spikes = bmi.pklfile['spks'][unit_code]
                ind = bmi.index
                
                md, pref_dir, meanfr, rss, bt = [],[],[],[],[]
            
                for block_type in np.unique(ind.block_type):
                    
                    firing_rates = []
                    for angle in np.arange(0,360,45):
                        trial = ind[(ind['direction']==angle)&
                                    (ind['block_type']==block_type)&
                                    (ind['error_clamp']==error_clamp)]['trial_number']
                        align_pts = bmi.rpp_target[trial] / 30000
                        fr = [np.sum((start<spikes) & (spikes<start+duration)) / duration for start in align_pts]            
                        firing_rates.append(fr)
                    
                    
                    directions_rad = np.deg2rad(np.arange(0,360,45))
                    
                    # There should be at least one trial in each direction.
                    # Otherwise, there will be nan in true_fr
                    if np.all(np.array(list(map(len,firing_rates))) > 0):
                        
                        true_fr = [np.mean(firing_rate) for firing_rate in firing_rates]
                    
                        # Define multiple initial guesses
                        initial_guesses = generate_initial_guesses(true_fr, directions_rad)
                    
                        # Initialize default values in case fitting fails
                        MD, PD, meanFR, pred_fr = None, None, None, None
                    
                        # Attempt multiple guesses
                        for guess in initial_guesses:
                            try:
                                (MD, PD, meanFR), pcov = curve_fit(
                                    cosine_model, directions_rad, true_fr,
                                    p0=guess,
                                    bounds=([0, -2 * np.pi, -np.inf], [np.inf, 2 * np.pi, np.inf])
                                )
                                if PD < 0:
                                    PD += 2 * np.pi
                                pred_fr = cosine_model(directions_rad, *(MD, PD, meanFR))
                                break  # Exit loop if fitting succeeds
                            except RuntimeError:
                                continue  # Try the next initial guess if fitting fails
                    
                        # Check if fitting succeeded
                        if MD is None or PD is None or meanFR is None:
                            print(f"Fitting failed for unit {unit_code}. Skipping...")
                            md.append(np.nan)
                            pref_dir.append(np.nan)
                            meanfr.append(np.nan)
                            rss.append(np.nan)
                            continue
                        
                        # PD modification
                        PD = np.rad2deg(PD)
                        if PD > 180:
                            PD -= 360
                        
                        md.append(MD)
                        pref_dir.append(PD)
                        meanfr.append(meanFR)
                        rss.append(np.sum((true_fr - pred_fr)**2))
                        bt.append(block_type)
            
                tuning_df.append(pd.DataFrame({
                    'block_type': bt,
                    'MD': md,
                    'PD': pref_dir,
                    'meanFR': meanfr,
                    'RSS': rss,
                    'session': session,
                    'rotation': bmi.rotation_angle
                }))
            
            tuning_df = pd.concat(tuning_df).reset_index(drop=True)
            
            tune = tuning_df.groupby('session')
            
            tune_pd = tune['PD'].apply(list)
            labels = tune_pd.index
            rot = tune['rotation'].apply(np.max) # Should be only one
            
            marker_map = {50: 'o', 90: '^', 270: 's', 310: 'x'}
            
            if PLOT and len(cluster_df) > 3:
                
                plt.figure(figsize=(3,3))
                for i in range(len(tune_pd)):
                    
                    pds = tune_pd.iloc[i]
                    
                    # [Could be wrong]
                    # for m in range(1,len(pds)):
                    #     if pds[m] - pds[m-1] > 180:
                    #         pds[m] -= 360
                    #     elif pds[m] - pds[m-1] < -180:
                    #         pds[m] += 360
                    
                    plt.plot(pds,c=plt.cm.rainbow(i/len(tune_pd)))
                    plt.scatter(range(len(pds)),
                                pds,
                                color=plt.cm.rainbow(i/len(tune_pd)),
                                marker=marker_map[rot.iloc[i]],
                                s=10,
                                label=labels[i])
                    
                plt.ylim([-360,360])
                plt.yticks(np.arange(-360, 361, 180))
                plt.legend(frameon=False, bbox_to_anchor=(1,1))
                plt.xticks(range(4),['BL','EP','LP','WO'])
                plt.ylabel('Preferred direction')
                plt.xlabel('Blocks')
                plt.title(f'Cluster ID {cluster_ID}')
                plt.show()
                
    
    def plot_example_cluster_algorithm(self):
                
        def find_waveform(subj: Tracking, unit_name: str):
            return subj.useful_df.query(f'unit == "{unit_name}"')['waveform'].iloc[0]
        
        # Indices for the examples
        if self.subject == 'airp':
            i=-13
        elif self.subject == 'braz':
            i=-4
        
        sub_sim = self.sim_df.iloc[i]
        
        label = sub_sim['unit']
        plt.figure(figsize=(3,6))
        for j, lab in enumerate(label):
            plt.plot(find_waveform(self, lab) - 10*j, 
                     c=plt.cm.rainbow(j/len(label)),
                     label=lab)
        plt.legend(frameon=False, bbox_to_anchor=(1,1), ncols=1)
        plt.xticks([])
        if SAVEFIG:
            plt.savefig(os.path.join(FIG_FOLDER, f'[{self.subject}]_example_cluster_algorithm_[waveform].svg'))
        plt.show()
        
        for metric in ['euclidean','correlation','total']:
            
            similarity = sub_sim[metric]
            complete, complete_label = complete_sort(similarity, label)
            data = np.tril(similarity)
            plt.figure(figsize=(4,4))
            plt.pcolormesh(data, cmap='Greys')
            plt.title(metric)
            plt.colorbar()
            plt.xticks(np.arange(len(data))+0.5, sub_sim['unit'], rotation=90)
            plt.yticks(np.arange(len(data))+0.5, sub_sim['unit'])
            if SAVEFIG:
                plt.savefig(os.path.join(FIG_FOLDER, f'[{self.subject}]_example_cluster_algorithm_[{metric}].svg'))
            plt.show()
        
        similarity = sub_sim['total']
        matched = similarity > self.threshold
        
        data = np.tril(matched)
        plt.figure(figsize=(4,4))
        plt.pcolormesh(data, cmap='Greys')
        plt.title('Masked')
        plt.colorbar()
        plt.xticks(np.arange(len(data))+0.5, sub_sim['unit'], rotation=90)
        plt.yticks(np.arange(len(data))+0.5, sub_sim['unit'])
        if SAVEFIG:
            plt.savefig(os.path.join(FIG_FOLDER, f'[{self.subject}]_example_cluster_algorithm_[masked].svg'))
        plt.show()

#%% Unit-specific analyses
class Spacial:

    def __init__(self, subject: str, sessions: list[str], rotation: dict, save_folder: dict):

        self.subject = subject
        self.sessions = sessions
        self.dates = [s[4:12] for s in self.sessions]
        self.rotation = rotation
        self.save_folder = save_folder
        
        print(f'[{self.subject}] Begin Spacial Analysis')
        self.raw_data: BMI = None
        self.raw_df = None
        self.data_df = None
        self.grouped_channel = None
        self.useful_channel = None
        self.useful_df = None 
        self.read_sessions(parse=True)
        self.read_data()
    
    @staticmethod
    def calc_waveform_metrics(row) -> pd.Series:
        """        
        For the four important locations, we obtain the location (time sample index) and amplitude (uV).
        1. peak_before: Peak-before trough (index 0 or peak)
        2. trough: Trough
        3. peak_after: Peak-after trough
        4. inflect: Inflection point (if any)
        
        UPDATE 2024/11/14
        Simplified this part to include only peak_after_amp, trough_amp, 
        peak_before_amp, waveform, and inverted. Other metrics are not that useful.
        However, the calculation process is preserved in case it's needed someday.
        """
    
        waveform = row['wf']
        # LEN_WF = len(waveform) # Should be 52 samples
        
        # First determine if the waveform is inverted
        # If inverted, flip waveform. 
        # Then set trough and peak-after trough    
        min_amp = np.min(waveform)
        max_amp = np.max(waveform)
        
        if (min_loc := np.argmin(waveform)) < (max_loc := np.argmax(waveform)): # The normal case
            peak_after_amp = max_amp
            # peak_after_loc = max_loc
            trough_amp = min_amp
            trough_loc = min_loc
            inverted = False
        else: # Inverted case
            peak_after_amp = -min_amp
            # peak_after_loc = min_loc 
            trough_amp = -max_amp
            trough_loc = max_loc
            waveform = -waveform
            inverted = True
        
        # Peak before trough
        if trough_loc > 0:
            peak_before_amp = np.max(waveform[:trough_loc])
            # peak_before_loc = np.argmax(waveform[:trough_loc])
        else:
            peak_before_amp = 0
            # peak_before_loc = 0
        
        # # Inflection point
        # try:
        #     fittedcurve = UnivariateSpline(np.linspace(peak_after_loc, LEN_WF-1, LEN_WF-peak_after_loc), 
        #                                    waveform[peak_after_loc:])
        #     fittedcurve_2d = fittedcurve.derivative(n=2)
        #     fit_curve = fittedcurve_2d(np.linspace(peak_after_loc, LEN_WF-1, LEN_WF-peak_after_loc))
        #     inflect_loc = np.argmin(np.abs(fit_curve)) + peak_after_loc
        #     inflect_amp = waveform[inflect_loc]
        # except Exception:
        #     inflect_loc = None
        #     inflect_amp = None
            
        # # Delta amplitudes (absolute values)
        # amp_drop = peak_before_amp - trough_amp
        # amp_rise = peak_after_amp - trough_amp
        
        # # Amplitude ratios
        # ratio_before_trough =  - peak_before_amp / trough_amp
        # ratio_after_trough =  - peak_after_amp / trough_amp
        
        # # Durations
        # dur_before_trough = trough_loc - peak_before_loc
        # dur_after_trough = peak_after_loc - trough_loc
        
        metrics = {
            "peak_after_amp": peak_after_amp,
            # "peak_after_loc": peak_after_loc, 
            "trough_amp": trough_amp,
            # "trough_loc": trough_loc,
            "peak_before_amp": peak_before_amp,
            # "peak_before_loc": peak_before_loc,
            # "inflect_amp": inflect_amp,
            # "inflect_loc": inflect_loc,
            # "amp_drop": amp_drop,
            # "amp_rise": amp_rise,
            # "ratio_before_trough": ratio_before_trough,
            # "ratio_after_trough": ratio_after_trough,
            # "dur_before_trough": dur_before_trough,
            # "dur_after_trough": dur_after_trough,
            "waveform": waveform,
            "inverted": inverted,
        }
        return pd.Series(metrics)
        
    def read_sessions(self, parse: bool = False):
        """
        Read all the sessions for that subject.
        
        Build self.raw_data, which is a dictionary that can be accessed through
        
            self.raw_data[self.sessions[i]]
        """
        data = dict()
        for session in self.sessions:
            task = BMI(session, self.save_folder[session], self.rotation[session])
            # The parse parameter is False for default, but are set by read_full_data during init.
            # Separating BMI init and parsing data can save time if want to debug BMI class.
            if parse:
                task.parse_behavior()
                task.get_index()
            data[session] = task
            
        self.raw_data = data
        
    
    def read_data(self):
        """
        Read multiple data frames from the raw_data.
        """
        
        # [[raw_df]] - all units from all sessions
        dfs = []
        for datum in self.raw_data.values():
            df = pd.DataFrame(datum.pklfile, columns=['fr','ptt','wf'])
            df['unit'] = datum.session + '_' + df.index
            dfs.append(df)
        self.raw_df = pd.concat(dfs)
        
        session,date,unit_code,channel,is_direct,rotation, = [],[],[],[],[],[]
        for i in range(len(self.raw_df)):
            s, d, c, u = parse_unit(self.raw_df['unit'].iloc[i])
            try:
                direct = u in self.raw_data[s].direct_units
            except:
                print(f"[{s}] - skipped direct units.")
                direct = None
            rot = self.raw_data[s].rotation_angle
            session.append(s)
            date.append(d)
            channel.append(c)
            unit_code.append(u)
            is_direct.append(direct)
            rotation.append(rot)
            
        metrics_df = self.raw_df.apply(self.calc_waveform_metrics, axis=1)
        self.raw_df = pd.concat([self.raw_df, metrics_df], axis=1)
        self.raw_df['session'] = session
        self.raw_df['date'] = date
        self.raw_df['channel'] = channel
        self.raw_df['unit_code'] = unit_code
        self.raw_df['is_direct'] = is_direct
        self.raw_df['rotation'] = rotation
        self.raw_df = self.raw_df[self.raw_df['channel']!=0] # Excluding V probes channels, see parse_unit().
        self.raw_df['suggested'] = True
        self.raw_df.loc[~((self.raw_df['fr']>1) & (self.raw_df['ptt']>=80)), 'suggested'] = False
        
        # [[data_df]] - the units that fits the criteria (fr>1 and peak-to-trough >= 80)
        self.data_df = self.raw_df[self.raw_df['suggested']].reset_index(drop=True)
        
        # [[useful_df]] - the channels that have at least 5 units across all recordings.
        grouped_channel = self.data_df[['waveform','channel']].groupby('channel').count().sort_values(by='waveform',ascending=False)
        self.useful_channel = grouped_channel[grouped_channel['waveform']>=USEFUL_N_UNIT].index
        unuseful_channel = grouped_channel[grouped_channel['waveform']<USEFUL_N_UNIT].index
        self.useful_df = self.data_df.query(f'channel not in {list(unuseful_channel)}').drop(columns=['suggested','wf']).reset_index(drop=True)
        self.grouped_channel = self.useful_df[['waveform','channel']].groupby('channel').count().sort_values(by='waveform',ascending=False)

    def unit_sfc(self, channels: list[int] = None, rand: bool = False):
    
        start_sec, end_sec = -1, 1
        fs = 1000 # Sampling frequency

        N_pts = int(0.6*fs) # Use N_pts//2 data points to compute spectrum
        N_freqs = N_pts//2 + 1 # So we have N_freqs of frequency for the spectrums
        f = np.fft.rfftfreq(N_pts, 1/fs)
                        
        # Divide the aligned period (from start_sec to end_sec around align_pts) into N_timesteps sections
        N_timesteps = 100
        t = np.linspace(start_sec, end_sec, N_timesteps)
        
        if channels is None:
            channels = list(self.useful_channel) # If no specific channels are provided, use all useful channels.
            
        with h5py.File(os.path.join(OUTPUT_DATA_FOLDER, f'{self.subject}_sfc.h5'), 'a') as h5file:
            h5file.attrs['start_sec'] = start_sec
            h5file.attrs['end_sec'] = end_sec
            h5file.attrs['fs'] = fs
            h5file.attrs['freqs'] = f
            h5file.attrs['timesteps'] = t
            
            for session in self.sessions: # For each selected channel
                print(session)

                bmi = self.raw_data[session]
                ns2 = bmi.ns2file
                ind = bmi.index
                
                session_df = self.useful_df[self.useful_df['session'] == session]

                for channel in session_df['channel'].unique(): 

                    if channel not in channels:
                        continue
                    
                    lfp = ns2.getdata()['data'][channel] # Read LFP data
                    unit_codes = session_df[(session_df['channel'] == channel) & session_df['is_direct'] == True]['unit_code'].unique()
                    for unit_code in unit_codes:
                        spike_times = bmi.pklfile['spks'].get(unit_code) # Spike times

                        # [Estimate firing rate] - upsampled to the same rate as LFP
                        fr = __calc_firing_rate(spike_times)
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
                        
                        # Create a matrix to store all the detailed align points
                        fine_align_pts = np.zeros((N_trials, N_timesteps), dtype=int)
                        for t in range(N_trials):
                            fine_align_pts[t] = np.linspace(align_pts[t] + start_sec * fs, 
                                                            align_pts[t] + end_sec * fs, 
                                                            N_timesteps, 
                                                            dtype=int)
                        
                        coherogram = np.zeros((N_timesteps, N_freqs))
                        for ts in range(N_timesteps): # Iterate through each time step
                            Sxx = np.zeros(int(N_pts/2+1)) # Field spectrum.
                            Syy = np.zeros(int(N_pts/2+1)) # Spike spectrum.
                            Sxy = np.zeros(int(N_pts/2+1), dtype=complex) # Cross spectrum.
                        
                            for t in range(N_trials):
                                pt = fine_align_pts[t, ts] # The align point
                                
                                # Take N_pts around pt to calculate coherence
                                field_raw = lfp[pt-N_pts//2: pt+N_pts//2]
                                spike_raw = fr[pt-N_pts//2: pt+N_pts//2]
                                sxx, syy, sxy = calc_spectrum(spike_raw, field_raw, fs=1000)
                                
                                # Directly adding the averaged values
                                Sxx += (sxx / N_trials)
                                Syy += (syy / N_trials)
                                Sxy += (sxy / N_trials)
                            
                            cohr = abs(Sxy) / np.sqrt(Syy) / np.sqrt(Sxx)
                            coherogram[ts] = cohr
                        
                        path = f'{session}/ch_{channel}/unit_{unit_code}'
                        grp = h5file.require_group(path)
                        if "coherogram" in grp:
                            del grp["coherogram"]  # Delete existing dataset if it exists
                        grp.create_dataset("coherogram", data=coherogram, compression="gzip", compression_opts=4, chunks=True)


#%% SFC

def sfc_of_tracked_neuron(subj: Tracking, example: pd.Series, title: str, rand: bool = False):
    """
    Plot trial-averaged SFC for each session in a tracked neuron.
    
    example: usage like airp.useful_clusters.iloc[0]
    title: for the title and the saved filename
    rand: if this is to randomized aligned points.   
    
    """
    
    start_sec, end_sec = -1, 1
    
    # plt.figure(figsize=(4,4)) # Each line is data from a neuron
    
    coherograms = []
    sessions = []
    
    for n in range(example.n_unit): # For each neuron in a cluster
        session, unit_code, channel = example.neuron.df[['session','unit_code','channel']].iloc[n]
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
            fr = __calc_firing_rate(spike_times)
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
            
            coherogram = np.zeros((N_timesteps, N_freqs))
            for ts in range(N_timesteps): # Iterate through each time step
                Sxx = np.zeros(int(N_pts/2+1)) # Field spectrum.
                Syy = np.zeros(int(N_pts/2+1)) # Spike spectrum.
                Sxy = np.zeros(int(N_pts/2+1), dtype=complex) # Cross spectrum.
            
                for t in range(N_trials):
                    pt = fine_align_pts[t, ts] # The align point
                    
                    # Take N_pts around pt to calculate coherence
                    field_raw = lfp[pt-N_pts//2: pt+N_pts//2]
                    spike_raw = fr[pt-N_pts//2: pt+N_pts//2]
                    sxx, syy, sxy = calc_spectrum(spike_raw, field_raw, fs=1000)
                    
                    # Directly adding the averaged values
                    Sxx += (sxx / N_trials)
                    Syy += (syy / N_trials)
                    Sxy += (sxy / N_trials)
                
                cohr = abs(Sxy) / np.sqrt(Syy) / np.sqrt(Sxx)
                coherogram[ts] = cohr
            
            coherograms.append(coherogram)

        except:
            pass
        
        
    return np.array(coherograms), np.array(sessions)


