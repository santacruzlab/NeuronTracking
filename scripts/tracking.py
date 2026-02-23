import os
import re
import sys
import glob
import scipy
import tables
import pickle
import random
import datetime

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
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SCRIPT_FOLDER)
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
RAWDATA_FOLDER = os.path.join(PROJECT_FOLDER, 'rawdata')
GITHUB_FOLDER = os.path.dirname(PROJECT_FOLDER)
BMI_FOLDER = os.path.join(GITHUB_FOLDER, 'bmi_python')
NSX_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'blackrock')
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
FIG_FOLDER = os.path.join(PROJECT_FOLDER, 'plots')
NEV_OUTPUT_FOLDER = r"F:\cole\neuron_tracking_nev_outputs\neuron_tracking_pkl_files"

sys.path.insert(0,BMI_FOLDER)
sys.path.insert(0,NS_FOLDER)
# os.chdir(BMI_FOLDER)
from riglib.blackrock.brpylib import NsxFile
# os.chdir(BMI_FOLDER)
# from nsfile import NSFile
from riglib.ripple.pyns.pyns.nsfile import NSFile
os.chdir(SCRIPT_FOLDER)
from sessions import AIRPORT_SESSIONS, BRAZOS_SESSIONS, AIRPORT_SESSIONS_1, AIRPORT_SESSIONS_2, AIRPORT_ROTATION, BRAZOS_ROTATION
os.chdir(PROJECT_FOLDER)

print('Imported bmi_python libraries!')

# Constants
LETTER_CODE = {2.: 'a', 4.: 'b', 8.: 'c', 16.: 'd'}
ROTATION_CLR = {50: 'blue', 90: 'red', 270: 'green', 310: 'orange'}
SUBJECT = ['airp', 'braz']
SESSIONS = dict(zip(SUBJECT, [AIRPORT_SESSIONS, BRAZOS_SESSIONS]))
ROTATION = dict(zip(AIRPORT_SESSIONS + BRAZOS_SESSIONS, 
                    AIRPORT_ROTATION + BRAZOS_ROTATION))
#SAVE_FOLDER = dict(zip([AIRPORT_SESSIONS_1, AIRPORT_SESSIONS_2, BRAZOS_SESSIONS], [r"X:\storage\rawdata", r"Y:\storage\rawdata", r"W:\storage\rawdata"]))
SAVE_FOLDER = {}
storage_paths = [r"X:\storage\rawdata", r"Y:\storage\rawdata", r"W:\storage\rawdata"] # santacruz2, santacruz1, santacruz3
for i, sessions in enumerate([ AIRPORT_SESSIONS_1, AIRPORT_SESSIONS_2, BRAZOS_SESSIONS ]):
    for session in sessions:
        SAVE_FOLDER[session] = storage_paths[i]

# SAVE_FOLDER = dict(zip([AIRPORT_SESSIONS_1, AIRPORT_SESSIONS_2, BRAZOS_SESSIONS], [r"K:\storage\rawdata", r"L:\storage\rawdata", r"J:\storage\rawdata"]))
SUBJECT_COLOR = dict(zip(SUBJECT, ['g', 'b']))
DUMMY_NUMBER = 1e7 
N_BOOTSTRAP = 1000
USEFUL_N_UNIT = 3
SAVEFIG = True

def _generate_nev_output(
        all_sessions: list[str] = AIRPORT_SESSIONS + BRAZOS_SESSIONS,
        count: bool = True):
    """
    
    The main fit_tuning to generate the nev_output.pkl file for each session.
    
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
        self.file_prefix_hdf = os.path.join(SAVE_FOLDER[self.session], 'hdf', self.session)
        self.file_prefix_ripple = os.path.join(SAVE_FOLDER[self.session], 'ripple', self.session)
        self.file_prefix_nev_output = os.path.join(NEV_OUTPUT_FOLDER, self.session)

        
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
        if os.path.exists(self.file_prefix + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix_nev_output + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix + '_KFDecoder.pkl'):
            self.has_decoder = True


    def load_data(self):
        
        # if self.has_hdf:
        #     self.hdffile = tables.open_file(self.file_prefix_hdf + '.hdf')
            
        if self.has_ns5:
            self.ns5file = NsxFile(self.file_prefix_ripple + '.ns5')
                
        if self.has_nev:
            self.nevfile = NSFile(self.file_prefix_ripple + '.nev')
            self.spike_entities = [e for e in self.nevfile.get_entities() if e.entity_type==3]
            
        if self.has_mat:
            self.matfile = scipy.io.loadmat(self.file_prefix + '_syncHDF.mat')
            
        if self.has_nev_output:
            with open(self.file_prefix_nev_output + '_nev_output.pkl', 'rb') as f:
                self.pklfile = pickle.load(f)
        
        if self.has_decoder:
            with open(self.file_prefix + '_KFDecoder.pkl', 'rb') as f:
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

    def __init__(self, subject: str):
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
        self.sessions = SESSIONS[self.subject]
        self.dates = [s[4:12] for s in self.sessions]
        
        print(f'[{self.subject}] Start tracking')
        self.raw_data: BMI = None
        self.raw_df = None
        self.data_df = None
        self.grouped_channel = None
        self.useful_channel = None
        self.useful_df = None 
        self.read_sessions() # parse=True)
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
        
        print(f'[{self.subject}] Post-processing analysis')
        self.calc_tuning()
        self.calc_PD_metric()
        
        
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
            task = BMI(session)
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
            direct = u in self.raw_data[s].direct_units
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
            
        
    def plot_unit_per_session(self, ax):
        
        """
        Plot number of units per session
        Used in Fall 2024 meeting slides page 11.
        """
        y = self.useful_df[['channel','date']].groupby('date').count()
        x = y.index
        
        ax.plot(x, y, 'o--', 
                c=SUBJECT_COLOR[self.subject],
                label=self.subject)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=5)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Session')
        ax.set_title('# units for each session')
        ax.legend(frameon=False)
        
        
    def plot_channel_stats(self, ax):
        """
        Plot channel statistics
        Used in Fall 2024 meeting slides page 12.
        """
        ax.plot(range(1,len(self.grouped_channel)+1), 
                self.grouped_channel, 
                lw=0.5, 
                color=SUBJECT_COLOR[self.subject],
                label=self.subject)
        ax.scatter(range(1,len(self.grouped_channel)+1), 
                   self.grouped_channel, 
                   marker='o', 
                   s=10, 
                   color=SUBJECT_COLOR[self.subject])
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
            
    
    def plot_cluster_PD_rate(self, cluster_ID):
        
        df = self.useful_clusters.query(f'cluster_ID == {cluster_ID}').iloc[0]
        
        rate, intercept, r2, days = df[['PD_rate','PD_intercept','PD_r2','days']]
        pvals = df[['PD_intercept_pvalue','PD_rate_pvalue']].values
        neuron = df.neuron
        PD = minimal_PD_change(neuron.df.PD.copy())
        
        plt.figure(figsize=(5,5))
        plt.scatter(days, PD, c=list(map(lambda x: ROTATION_CLR[x], neuron.df.rotation)))
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


def __plot_transformation_rescale(subj: Tracking):
    
    """
    Check euclidean distribution and corrcoef distribution.
    Used in Fall 2024 meeting slides page 24-25.
    """
    
    def rescale(array, metric: str, nine: bool = False):
        """
        0-99 scaling in van Beest 2024 Nat Methods
        If nine == True, use 0-99 scaling.
        """
                
        if nine:
            maxx = np.percentile(array, 99)
        else:   
            maxx = np.max(array)
        minn = np.min(array)
        
        match metric:
            case 'correlation':
                result = (array - minn) / (maxx - minn)
            case 'euclidean':
                result = (maxx - array) / (maxx - minn)
    
        # Clipping    
        result[result < 0] = 0
        result[result > 1] = 1
        result[np.isnan(result)] = 0 
        return result
    
    calc_euclidean = lambda a,b: np.sqrt(np.sum((a-b)**2))
    calc_corrcoef = lambda a,b: np.corrcoef(a, b)[0,1]
    
    euclidean = []
    corrcoef = []
    
    for ch in subj.useful_channel:
        ch_df = subj.useful_df[subj.useful_df['channel']==ch].sort_values(by='date')
        
        for i in range(len(ch_df)):
            for j in range(len(ch_df)):
                euclidean.append(calc_euclidean(
                    ch_df['waveform'].iloc[i], ch_df['waveform'].iloc[j]))
                corrcoef.append(calc_corrcoef(
                    ch_df['waveform'].iloc[i], ch_df['waveform'].iloc[j]))

    corrcoef = np.array(corrcoef)
    euclidean = np.array(euclidean)
    
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    # Plot raw metrics
    ax[0].hist(euclidean, bins=100, color=SUBJECT_COLOR[subj.subject])
    ax[1].hist(corrcoef, bins=100, color=SUBJECT_COLOR[subj.subject])
    
    ax[0].set_title('Euclidean distance similarity', fontsize=10)
    ax[1].set_title('Pearson correlation similarity', fontsize=10)
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Counts')
    ax[0].set_xlabel('Euclidean distance')
    ax[1].set_xlabel('Pearson correlation')
    fig.subplots_adjust(wspace=0.6)
    
    corrcoef = corrcoef[np.abs(corrcoef-1) > 1e-10]
    corrcoef = np.arctanh(corrcoef)
    
    euclidean = euclidean[np.abs(euclidean-0) > 1e-10]
    euclidean = np.log(euclidean)
    
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    # Plot transformed metrics
    ax[0].hist(euclidean, bins=100, color=SUBJECT_COLOR[subj.subject])
    ax[1].hist(corrcoef, bins=100, color=SUBJECT_COLOR[subj.subject])
    ax[0].set_title('Euclidean distance similarity', fontsize=10)
    ax[1].set_title('Pearson correlation similarity', fontsize=10)
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Counts')
    ax[0].set_xlabel('Log transformed\nEuclidean distance')
    ax[1].set_xlabel('Fisher transformed\nPearson correlation')
    fig.subplots_adjust(wspace=0.6)
    
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    # Plot rescaled transformed metrics (using 0-99 scaling)
    ax[0].hist(rescale(euclidean, 'euclidean', nine=True), bins=100, color=SUBJECT_COLOR[subj.subject])
    ax[1].hist(rescale(corrcoef, 'correlation', nine=True), bins=100, color=SUBJECT_COLOR[subj.subject])
    ax[0].set_title('Euclidean distance similarity', fontsize=10)
    ax[1].set_title('Pearson correlation similarity', fontsize=10)
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Counts')
    ax[0].set_xlabel('Log transformed\nEuclidean distance')
    ax[1].set_xlabel('Fisher transformed\nPearson correlation')
    fig.subplots_adjust(wspace=0.6)
    
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    # Plot rescaled transformed metrics (using min-max scaling)
    ax[0].hist(rescale(euclidean, 'euclidean'), bins=100, color=SUBJECT_COLOR[subj.subject])
    ax[1].hist(rescale(corrcoef, 'correlation'), bins=100, color=SUBJECT_COLOR[subj.subject])
    ax[0].set_title('Euclidean distance similarity', fontsize=10)
    ax[1].set_title('Pearson correlation similarity', fontsize=10)
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Counts')
    ax[0].set_xlabel('Log transformed\nEuclidean distance')
    ax[1].set_xlabel('Fisher transformed\nPearson correlation')
    fig.subplots_adjust(wspace=0.6)


def __plot_sorting_grouping(subjs):
    
    """
    Plot the sorting and grouping quality using brazos no.56 channel.
    Used in Fall 2024 meeting slides page 27-29.
    """
    
    ch = braz.sim_df.iloc[56]
    
    similarity = ch['total']
    sym = ch['unit']
    sym_map = dict(zip(sym, range(len(sym))))
    complete, complete_sym = complete_sort(similarity, sym)
    
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    # Compare total similarity matrix before and after sorting
    ax[0].pcolormesh(similarity, cmap='Blues', vmax=0.7, vmin=0)
    ax[0].set_xticks(np.arange(len(similarity))+0.5)
    ax[0].set_xticklabels(list(map(lambda x:sym_map[x], sym)), rotation=90, fontsize=5)
    ax[0].set_yticks(np.arange(len(similarity))+0.5)
    ax[0].set_yticklabels(list(map(lambda x:sym_map[x], sym)), fontsize=5)
    ax[1].pcolormesh(complete, cmap='Blues', vmax=0.7, vmin=0)
    ax[1].set_xticks(np.arange(len(similarity))+0.5)
    ax[1].set_xticklabels(list(map(lambda x:sym_map[x], complete_sym)), rotation=90, fontsize=5)
    ax[1].set_yticks(np.arange(len(similarity))+0.5)
    ax[1].set_yticklabels(list(map(lambda x:sym_map[x], complete_sym)), fontsize=5)
    plt.show()
    
    
    plt.figure(figsize=(6,2))
    # Plot traces of potential matched units
    plt.subplot(131)
    for unit in complete_sym[5:10]:
        plt.plot(braz.useful_df[braz.useful_df.unit==unit]['waveform'].iloc[0], c='orange')
        
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(132)
    for unit in complete_sym[18:32]:
        plt.plot(braz.useful_df[braz.useful_df.unit==unit]['waveform'].iloc[0], c='green')
    
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    for unit in complete_sym[18:32]:
        plt.plot(braz.useful_df[braz.useful_df.unit==unit]['waveform'].iloc[0], c='green')
    
    for unit in complete_sym[13:18]:
        plt.plot(braz.useful_df[braz.useful_df.unit==unit]['waveform'].iloc[0], c='grey')
    
    for unit in complete_sym[32:37]:
        plt.plot(braz.useful_df[braz.useful_df.unit==unit]['waveform'].iloc[0], c='grey')
    
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    for subj in subjs:
        plt.figure(figsize=(10,10))
        # Plot sorted total similarities for all useful channels in each subject.
        for i in range(len(subj.sim_df)):
            plt.subplot(12, 10, i+1)
            ch = subj.sim_df.iloc[i]
            similarity = ch['total']
            sym = ch['unit']
            sym_map = dict(zip(sym, range(len(sym))))
            complete, complete_sym = complete_sort(similarity, sym, threshold=-0.0001)
            
            # plt.pcolormesh(similarity, cmap='Blues', vmax=0.7, vmin=0) # Unsorted
            plt.pcolormesh(complete, cmap='Blues', vmax=0.7, vmin=0) # Sorted
            plt.yticks([])
            plt.xticks([])
        plt.show()
        
def __plot_units_by_first_appearance(subjs):
    subjs=[airp,braz]
    plt.figure(figsize=(5,4))
    for i, subj in enumerate(subjs):
        plt.subplot(1,2,i+1)
        first_date = np.min([subj.useful_clusters.neuron.iloc[_].start_date for _ in range(len(subj.useful_clusters.neuron))])
        
        dates = np.array([
            [(subj.useful_clusters.neuron.iloc[i].start_date-first_date).days, 
             (subj.useful_clusters.neuron.iloc[i].end_date-first_date).days - 
                 (subj.useful_clusters.neuron.iloc[i].start_date-first_date).days+1] 
            for i in range(len(subj.useful_clusters.neuron))])
        dates = dates[dates[:, 0].argsort()]
        
        for i,d in enumerate(dates):
            plt.broken_barh([d], [i,2], fc=SUBJECT_COLOR[subj.subject])
            
        plt.xlabel('Days')
        plt.ylabel('Units')
        # plt.yticks(np.arange(0, len(dates), 100),fontsize=5)
    plt.subplots_adjust(wspace=0.3)
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, '[all]_units_by_first_appearance.svg'))
    plt.show()
    
    
def __plot_matched_units_statistics():

    fig, ax = plt.subplots(figsize=(4,4))
    for subj in [airp, braz]:
        sns.scatterplot(data=subj.useful_clusters, x='duration', y='n_unit', 
                        lw=1, fc='none', ec=SUBJECT_COLOR[subj.subject], 
                        ax=ax, alpha=0.5, label=subj.subject)
    ax.set_xlim([0,120])
    ax.set_ylim([0,35])
    ax.legend(frameon=False)
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel('# unit in a cluster')
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, '[all]_cluster_statistics_scatter.svg'))
    plt.show()
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(4,4))
    for subj in [airp, braz]:
        ax[0].hist(subj.useful_clusters['duration'], bins=np.arange(0,120,5), density=True,
                   ec='k', color=SUBJECT_COLOR[subj.subject])
        ax[1].hist(subj.useful_clusters['n_unit'], bins=np.arange(0,35,2), density=True,
                   ec='k', color=SUBJECT_COLOR[subj.subject])
    ax[0].set_xlabel('Duration (days)')
    ax[1].set_xlabel('# unit in a cluster')
    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Density')
    fig.subplots_adjust(hspace=0.4)
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, '[all]_cluster_statistics_hist.svg'))
    plt.show()

        
        
def __plot_firing_rate_estimation(subj: Tracking):
    
    bmi = subj.raw_data[subj.sessions[0]]
    spikes =  next(iter(bmi.pklfile['spks'].values()))
    
    window_size = 0.5 # window size in seconds
    step_size = 0.05  # step size in seconds
    
    # Define time range for the analysis
    time_bins = np.arange(0, max(spikes), step_size)
    firing_rate = np.zeros_like(time_bins)
    
    # Compute firing rate for each window
    for i, t in enumerate(time_bins):
        count = np.sum((spikes >= t) & (spikes < t + window_size))
        firing_rate[i] = count / window_size  # Rate in Hz (spikes per second)
        
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(time_bins, slide_avg(firing_rate, 20),label='Box', c='b', lw=0.5)
    ax.plot(time_bins, gaussian(firing_rate, 6),label='Gauss', c='g', lw=0.5)
    ax.plot(time_bins, gaussian(slide_avg(firing_rate, 20), 6),label='Gauss & Box', c='r', lw=0.5)
    ax.eventplot(spikes, lineoffsets=4.5, color='k')
    ax.set_xlim([0,25])
    ax.set_ylim([-0.5,7.5])
    ax.set_ylabel('Estimated firing rate\n(sp/s)')
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel('Time (sec)')
    

def __plot_tuning_example():

    subj = airp
    cluster_ID = 554
    u = 0

    cluster_df = subj.clusters.neuron.iloc[cluster_ID].df
    
    session = cluster_df.session.iloc[u] 
    unit_code = cluster_df.unit_code.iloc[u]
    unit = cluster_df.unit.iloc[u]
    
    bmi = subj.raw_data[session]
    spikes = bmi.pklfile['spks'][unit_code]
    ind = bmi.index
    
    tuning = []
    for angle in np.arange(0,360,45):
        trial = ind[(ind['direction']==angle)&
                    (ind['block_type']==1)&
                    (ind['error_clamp']==0)]['trial_number']
        align_pts = bmi.rpp_target[trial] / 30000
        fr = [np.sum((start<spikes) & (spikes < start+0.5)) / 0.5 for start in align_pts]            
        tuning.append(fr)
    
    directions_rad = np.deg2rad(np.arange(0,360,45))
    true_fr = [np.mean(tune) for tune in tuning]
    
    (MD, PD, meanFR), pcov = curve_fit(
        cosine_model, directions_rad, true_fr, 
        p0=[10, 0, 10],
        bounds=([0, -2*np.pi, -np.inf], [np.inf, 2*np.pi, np.inf])
    )
    
    if PD < 0:
        PD += 2*np.pi
    
    pred_fr = cosine_model(directions_rad, *(MD, PD, meanFR))
    
    # [Tuning curve for each unit]
    theta_fine = np.linspace(0, 2 * np.pi, 360)
    fitted_rates = cosine_model(theta_fine, *(MD, PD, meanFR))
    
    plt.figure(figsize=(4,4))
    for i, angle in enumerate(np.arange(0,360,45)):
        plt.errorbar(angle, np.mean(tuning[i]), 
                     yerr=np.std(tuning[i])/np.sqrt(len(tuning[i])),
                     fmt='o', 
                     color=plt.cm.rainbow(u/len(cluster_df)))
    plt.plot(np.rad2deg(theta_fine), fitted_rates, 
             c='k',lw=2,ls='--',
             label=unit)
    plt.xlabel('Direction (degree)',fontsize=10)
    plt.xticks(np.arange(0,361,90),fontsize=10)
    plt.yticks(fontsize=10)
    # plt.legend(frameon=False, fontsize=10, loc='upper left')
    plt.ylabel('Firing rate (sp/s)',fontsize=10)
    plt.title(f'MD: {MD:.2f}; PD: {np.rad2deg(PD):.2f}\nFR: {meanFR:.2f}, SSE: {np.sum((true_fr - pred_fr)**2):.2f}',fontsize=10)
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, '[airp]_tuning_example.svg'))
    plt.show()

        

def __plot_matched_units_waveform(subj, cluster_ID):

    matched_units = subj.useful_df[subj.useful_df['cluster_ID']==cluster_ID].sort_values(by='date').reset_index()
    
    block = subj.tuning_params.get('block_type')
    duration = subj.tuning_params.get('duration')
    title = f'[{subj.subject}] cluster {cluster_ID}\n Block {block} and duration {duration}'
    
    plt.figure(figsize=(5,5))
    for u in range(len(matched_units)):
        plt.plot(matched_units['waveform'].iloc[u],
                 c=plt.cm.rainbow(u/len(matched_units)),
                 label=matched_units['unit'].iloc[u])
    plt.title(title)
    plt.legend(frameon=False, bbox_to_anchor=(1,1), ncols=2)
    plt.xticks([])
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_[{cluster_ID}]_matched_units_waveform.svg'))
    plt.show()


def __plot_matched_units_tuning_arrow(subj, cluster_ID):
    
    cluster_df = subj.useful_clusters[subj.useful_clusters['cluster_ID']==cluster_ID]['neuron'].iloc[0].df
    title = f'[{subj.subject}] cluster {cluster_ID}'
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4,4))
    ax.set_rmax(1)
    ax.set_title(title)
    for i in range(len(cluster_df)):
        pds = cluster_df.PD.iloc[i]
                
        ax.annotate(
            '', xy=(np.deg2rad(pds), 1), xytext=(0, 0),
            arrowprops=dict(facecolor=plt.cm.rainbow(i/len(cluster_df)), 
                            edgecolor=plt.cm.rainbow(i/len(cluster_df)), 
                            shrink=0, width=1, headwidth=5, headlength=5))
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_[{cluster_ID}]_matched_units_tuning_polar.svg'))
    plt.show()


def __plot_PD_random_span(n_PD: int = 10, n_samples: int = 10000):
    
    dist = calc_random_span(n_PD,n_samples)
    thres = np.percentile(dist, 5) # 0.05 significance level
    
    plt.figure(figsize=(4,4))
    plt.hist(dist, bins=30, density=True)
    plt.axvline(thres, c='r', ls='--', label=np.round(thres,3))
    plt.legend(frameon=False, loc='upper left')
    plt.xlim([np.min(dist)-30, None])
    plt.title(f'{n_PD} PDs, random dist over {n_samples} samples')
    plt.xlabel('PD span (angle)')
    plt.ylabel('Prob density')
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, f'[none]_PD_random_span_[{n_PD}]_[{n_samples}].svg'))
    plt.show()
    

def __plot_rsquared_distribution(subj):
    """
    Used in Spring 2025 meeting slides
    """
    df = subj.useful_clusters[['PD_r2','n_unit','PD_rate_pvalue']].copy()
    df['corr'] = (df['PD_r2'] >= 0.6) 
    df['sig'] = (df['PD_rate_pvalue'] < 0.05)
    
    
    fig,ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(data=df, x='PD_r2', y='n_unit', hue='corr', style='sig', ax=ax, legend=None)
    ax.set_xlabel('R_squared')
    ax.set_ylabel('# units in a cluster')
    ax.set_yticks(np.arange(3, np.max(df.n_unit)+1,2))
    plt.title(subj.subject)
    # plt.legend(frameon=False, loc='upper left')
    plt.show()
    
    
def __plot_metric_stability():
    
    metric_label_pair = {
        # 'peak_after_amp': 'Peak amplitude (uV)',
        # 'trough_amp': 'Trough amplitude (uV)',
        'fr': 'Firing rate (sp/s)',
        }
    
    df = []
    for subj in [airp, braz]:
        for used_metric in metric_label_pair.keys():
            data = []
            for i in range(len(subj.useful_clusters)):
                units = subj.useful_clusters.neuron.iloc[i].units
                first = subj.useful_df[subj.useful_df['unit']==units[0]][used_metric].iloc[0]
                last = subj.useful_df[subj.useful_df['unit']==units[-1]][used_metric].iloc[0]
                data.append((last - first) / first * 100)
            df.append(pd.DataFrame({
                'subject': subj.subject,
                'metric': used_metric,
                'value': data
            }))
    df = pd.concat(df).reset_index()
    df = df[df['value'] < 600]
    
    fig, ax = plt.subplots(figsize=(2,4))
    sns.violinplot(data=df, x='metric', y='value', split=True, hue='subject', gap=.1,
                   density_norm="width", inner=None, ax=ax, cut=0)
    # plt.xticks(np.arange(2), ['Peak amp','Trough amp'])
    plt.xlabel(None)
    plt.ylabel('Relative change (%)')
    plt.legend(loc='upper left', frameon=False)
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, f'[all]_stability_[{used_metric}].svg'))
    plt.show()
    
    for subj in [airp, braz]:
        for metric in df.metric.unique():
            dd = df.query(f'(metric=="{metric}") & (subject=="{subj.subject}")')
            res = pg.ttest(dd.value, 0).squeeze()
            pval = res['p-val']
            effect_size = res['cohen-d']
            print(subj.subject, metric, pval, effect_size)
    

def __plot_units_each_day():
    
    max_d = 0
    plt.figure(figsize=(3,3))
    for subj in [airp,braz]:
        name = subj.subject 
        data = subj.useful_df[['channel','date']].groupby('date').count()
        counts = data.values.flatten()
        counts = counts / np.max(counts) * 100
        dates = list(map(lambda d: datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8])), data.index))
        days = [(dates[i]-dates[0]).days for i in range(len(dates))]
        max_d = max(np.max(days), max_d)
        plt.plot(days, counts, c=SUBJECT_COLOR[name],lw=0.5, label=name)
        plt.scatter(days, counts, s=6, color=SUBJECT_COLOR[name])
    
    plt.xlabel('Days')
    plt.ylabel('Percent of neuron counts (%)')
    plt.ylim([0,100])
    plt.xticks(np.arange(0,max_d,10), rotation=90)
    plt.legend(frameon=False, loc='upper right')
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, '[all]_n_units_each_day.svg'))
    plt.show()


def __plot_dPD():
    for subj in [airp, braz]:
        res = []
        for i in range(len(subj.useful_clusters)):
            df = subj.useful_clusters.iloc[i]
            neuron = df.neuron
            PD = minimal_PD_change(neuron.df.PD.copy()) 
            for i in np.diff(PD):
                res.append(i)
        # rand_diff = np.array([np.diff(np.random.random(2)*360)[0] for _ in range(len(res))])
        # rand_diff[rand_diff > 180] -= 360
        # rand_diff[rand_diff < -180] += 360
        
        plt.figure(figsize=(4,4))
        plt.hist(res, bins=np.arange(-180,181,8), label=f'{subj.subject}')#, alpha=0.3, ec='k')
        # plt.hist(rand_diff, bins=np.arange(-180,181,8), label='randomized', alpha=0.3, ec='k')
        plt.xlabel('∆PD (deg)')
        plt.ylabel('Counts')
        plt.legend(frameon=False, loc='upper left')
        plt.xticks([-180,-90,0,90,180])
        if SAVEFIG:
            plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_dPD.svg'))
        plt.show()


def __plot_significant_PD_span_counts():
    
    airp_sig = sum(airp.useful_clusters.PD_span_pval < 0.05)
    airp_xsig = sum(airp.useful_clusters.PD_span_pval > 0.05)
    braz_sig = sum(braz.useful_clusters.PD_span_pval < 0.05)
    braz_xsig = sum(braz.useful_clusters.PD_span_pval > 0.05)
    
    plt.figure(figsize=(4,4))
    plt.bar([0,0.4,1,1.4], [airp_sig,airp_xsig,braz_sig,braz_xsig],width=0.4,ec='k')
    plt.xticks([0.2,1.2],['airp','braz'])
    plt.ylabel('Counts')
    if SAVEFIG:
        plt.savefig(os.path.join(FIG_FOLDER, '[all]_significant_PD_span_counts.svg'))
    plt.show()


def __plot_STA():
    
    subj = braz
    for use_cluster in range(len(subj.useful_clusters)):    
        example = subj.useful_clusters.iloc[use_cluster]
        
        plt.figure(figsize=(4,4))
        for dur in range(example.n_unit):
            session, unit_code, channel = example.neuron.df[['session','unit_code','channel']].iloc[dur]
            
            bmi = subj.raw_data[session]
            if not bmi.has_ns2:
                continue
            ns2 = bmi.ns2file
            spike_times = bmi.pklfile['spks'].get(unit_code)
            try:
                lfp = ns2.getdata()['data'][channel]
                fs = 1000
        
                beta = band_pass_filter(lfp, fs, 2, 30, 5)
                window = (-0.3,0.3) # +/- 1 second
                samples_window = (int(window[0] * fs), int(window[1] * fs))
                segments = []
                for spike in spike_times: # Using just the first 5000 spikes should be sufficient.
                    spike_idx = int(spike * fs)
                    if spike_idx + samples_window[0] >= 0 and spike_idx + samples_window[1] < len(lfp):
                        segments.append(beta[spike_idx + samples_window[0]:spike_idx + samples_window[1]])
                
                sta = np.mean(segments, axis=0)
                plt.plot(sta, c=plt.cm.rainbow(dur/example.n_unit), label=f'Day {example.days[dur]}')
            except:
                pass
        plt.legend(frameon=False,bbox_to_anchor=(1,1))
        plt.xticks([0,300,600],[-300,0,300])
        plt.xlabel('Time (msec)')
        plt.ylabel('Voltage (uV)')
        if SAVEFIG:
            plt.savefig(os.path.join(FIG_FOLDER, 'STA', f'[{subj.subject}]_sta_cluster_[{example.cluster_ID}].svg'))
        plt.show()


def __sanitycheck_firing_rate_estimation():
    """
    Making sure that fr is representative of the spike_times.    
    """
        
    session, unit_code, channel = airp.useful_clusters.neuron.iloc[0].df[['session','unit_code','channel']].iloc[0]
    
    bmi = airp.raw_data[session]
    ns2 = bmi.ns2file

    spike_times = bmi.pklfile['spks'].get(unit_code)
    fr = __calc_firing_rate(spike_times) # 20 Hz
    lfp = ns2.getdata()['data'][channel]
    
    fr_time = np.arange(0, len(fr) / 20, 1 / 20)
    lfp_time = np.arange(0, len(lfp) / 1000, 1 / 1000)
    interp_func = interp1d(fr_time, fr, kind='linear', fill_value='extrapolate')
    fr = interp_func(lfp_time)

    data = spike_times[100:500]
    plt.eventplot(data)
    plt.plot(lfp_time[int(data[0] * 1000):int(data[-1] * 1000)],
             fr[int(data[0] * 1000):int(data[-1] * 1000)])
    plt.show()
    
    

def __deprecated_plv():

    """
    Phase-locking values. Not used in the paper.
    """
    n=0
    fs=1000
    example = braz.useful_clusters.iloc[0]
    
    session, unit_code, channel = example.neuron.df[['session','unit_code','channel']].iloc[n]
    print(session)
    bmi = braz.raw_data[session]
    
    ns2 = bmi.ns2file
    ind = bmi.index
    
    spike_times = bmi.pklfile['spks'].get(unit_code)
    lfp = ns2.getdata()['data'][channel]
    beta = band_pass_filter(lfp, fs=1000, low=12, high=30, order=5)
    
    hilb_field = signal.hilbert(beta)
    lfp_phase = np.angle(hilb_field)
    
    # The indices where 
    spike_indices = (spike_times * fs).astype(int)  # Convert spike times to indices
    
    trial = ind[(ind['block_type']==1)&(ind['error_clamp']==0)] # All trials in the first block
    align_pts = np.array(bmi.rpp_target[trial['trial_number']] / 30, dtype=int)
    
    phases = []
    for ts in align_pts:
        start, end = (ts - fs*0.5).astype(int), (ts + fs*0.5).astype(int)    
        phases.extend(lfp_phase[spike_indices[(spike_indices > start) & (spike_indices < end)]])
    
    phases = np.array(phases)
    phases = phases[~np.isnan(phases)]
    
    np.abs(np.mean(np.exp(1j * phases)))
    
    
    # for i in range(len(subj.useful_clusters)):
    i=0
    example = braz.useful_clusters.iloc[i]
    
    for n in range(example.n_unit):
        session, unit_code, channel = example.neuron.df[['session','unit_code','channel']].iloc[n]
        print(session)
        bmi = braz.raw_data[session]
        
        ns2 = bmi.ns2file
        ind = bmi.index
        
        spike_times = bmi.pklfile['spks'].get(unit_code)
        lfp = ns2.getdata()['data'][channel]
        beta = band_pass_filter(lfp, fs=1000, low=12, high=30, order=5)
        
        hilb_field = signal.hilbert(beta)
        lfp_phase = np.angle(hilb_field)
        
        # The indices where 
        spike_indices = (spike_times * fs).astype(int)  # Convert spike times to indices
        
        trial = ind[(ind['block_type']==1)&(ind['error_clamp']==0)] # All trials in the first block
        align_pts = np.array(bmi.rpp_target[trial['trial_number']] / 30, dtype=int)
        
        plv = []
        
        for ts in align_pts:
        
            start, end = (ts - fs*0.5).astype(int), (ts + fs*0.5).astype(int)
            
            # Choose spike indices between start and end
            spike_indices[(spike_indices > start) & (spike_indices < end)]
            
            plv.append(np.abs(np.mean(np.exp(1j * lfp_phase[spike_indices[(spike_indices > start) & (spike_indices < end)]]))))
            
        plv_rand = []
        for ts in (np.random.random(336) * 942419).astype(int):
        
            start, end = (ts - fs*0.5).astype(int), (ts + fs*0.5).astype(int)
            spike_indices[(spike_indices > start) & (spike_indices < end)]# Choose spike indices between start and end
            plv_rand.append(np.abs(np.mean(np.exp(1j * lfp_phase[spike_indices[(spike_indices > start) & (spike_indices < end)]]))))
            
        plt.figure(figsize=(4,4))
        plt.hist(plv, bins=np.arange(0,0.6,0.02), ec='k', alpha=0.3, label='Aligned to movement')
        plt.hist(plv_rand, bins=np.arange(0,0.6,0.02), ec='k', alpha=0.3, label='Randomized')
        plt.xlabel('Phase locking values')
        plt.ylabel('Counts')
        plt.legend(frameon=False)
        plt.show()

    

def __get_threshold(subj: Tracking, pct: float, PLOT: bool):
    
    """
    Calculate the similarity threshold for determining matched units.

    This function generates a null distribution of similarity scores by randomly pairing 
    waveforms from different channels across sessions. The similarity metrics are 
    aggregated, rescaled, and used to compute a total similarity score for each pair. 
    A threshold is then determined based on the specified percentile of the null distribution.

    Parameters:
        subj (Tracking): The Tracking object containing the data and methods for 
                         similarity calculations.
        pct (float): The percentile value (e.g., 95 for 95th percentile) used to 
                     define the similarity threshold.

    Returns:
        None: The function updates the threshold attribute of the Tracking object.
    """


    # Obtain the amount of units for each channel in each session.
    # useful_channel is used to ensure there are at least 5 wavefroms
    subj_ch_session = np.zeros((len(subj.useful_channel), len(subj.dates)))
    for i, ch in enumerate(subj.useful_channel):
        for j, date in enumerate(subj.dates):
            subj_ch_session[i,j] = len(subj.useful_df[(subj.useful_df.date==date) & 
                                                      (subj.useful_df.channel==ch)])
    
    loc = np.zeros((len(subj.dates), 2))
    loc[:,1] = np.arange(len(subj.dates))
    
    threshold = []
    
    for it in range(200):
        
        print(f'{it} iteration')
        run = 0
        
        dist = []
        n_unit = []
        
        while len(dist) < 5000:
            
            run += 1        
            loc[:,0] = random.sample(range(len(subj.useful_channel)), len(subj.dates))
            
            used_pair = []
            for l in loc:
                if subj_ch_session[int(l[0]), int(l[1])] != 0:
                    used_pair.append((int(subj.useful_channel[int(l[0])]), 
                                      subj.dates[int(l[1])]))
            
            sim_temp = np.zeros((2, len(used_pair), len(used_pair)))
            for m, metric in enumerate(['correlation', 'euclidean']):
                for i, (ch1, date1) in enumerate(used_pair):
                    for j, (ch2, date2) in enumerate(used_pair):
                        
                        wf_s1 = subj.useful_df[(subj.useful_df['channel']==ch1)&(subj.useful_df['date']==date1)]['waveform']
                        wf_s2 = subj.useful_df[(subj.useful_df['channel']==ch2)&(subj.useful_df['date']==date2)]['waveform']
                        
                        # Randomly pick one if > 1 unit in that channel.
                        wf1 = wf_s1.iloc[random.randint(0, len(wf_s1)-1) if len(wf_s1) > 1 else 0]
                        wf2 = wf_s2.iloc[random.randint(0, len(wf_s2)-1) if len(wf_s2) > 1 else 0]
                        
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
            #     plt.xticks(np.arange(len(subj.dates)), subj.dates, rotation=90, fontsize=5)
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

airp = Tracking('airp')
braz = Tracking('braz')

read_lfp_later(airp)
read_lfp_later(braz)

#%% PLOTS

__plot_units_each_day()

fig,ax = plt.subplots(figsize=(4,4))
airp.plot_channel_stats(ax)
braz.plot_channel_stats(ax)
ax.axhline(3, ls='--', c='k')
if SAVEFIG:
    plt.savefig(os.path.join(FIG_FOLDER, '[all]_n_units_each_channel.svg'))
plt.show()

airp.plot_example_cluster_algorithm()
braz.plot_example_cluster_algorithm()

__plot_units_by_first_appearance(subjs=[airp,braz])
__plot_matched_units_statistics()
__plot_metric_stability()


cluster_ID = airp.useful_clusters.query('n_unit == 21').cluster_ID.values[0]
airp.plot_cluster_PD_rate(cluster_ID)
__plot_matched_units_tuning_arrow(airp, cluster_ID)
__plot_matched_units_waveform(airp, cluster_ID)

__plot_dPD()
__plot_STA()

#%% PLOTS - [Not very useful for paper]

airp.plot_waveform_of_channel()
braz.plot_waveform_of_channel()

__plot_transformation_rescale(airp)
__plot_transformation_rescale(braz)

airp.plot_masked_similarity_matrix(example=39)
braz.plot_masked_similarity_matrix()

__plot_sorting_grouping(subjs=[airp,braz])


airp.plot_waveform_of_all_matched_units()
braz.plot_waveform_of_all_matched_units(example=115)

__plot_firing_rate_estimation(airp)

__plot_tuning_example()
    
x = braz.useful_clusters.n_unit.argmax()
cluster_ID = braz.useful_clusters.iloc[x].cluster_ID
__plot_matched_units_tuning_arrow(braz, cluster_ID)
__plot_matched_units_waveform(braz,cluster_ID)

#%% UNFINISHED


#%% ISI


for subj in [airp, braz]:

    subj.useful_clusters['stable_isi'] = None
    
    for use_example in range(len(subj.useful_clusters)):
        example = subj.useful_clusters.iloc[use_example]
        example_cluster = example.neuron.df[['session','unit_code','channel']]
        # fig, ax = plt.subplots(figsize=(4,4))
        
        CV = []
        
        for i in range(len(example_cluster)):
            session, unit_code, channel = example_cluster.iloc[i]
            
            bmi = subj.raw_data[session]
            spike_times = bmi.pklfile['spks'].get(unit_code)
            
            isi = np.diff(spike_times) * 1000
            
            CV.append(np.std(isi) / np.mean(isi))
            
            # sns.kdeplot(isi, ax=ax,log_scale=True, label=f'Day {example.days[i]}', color=plt.cm.rainbow(i/example.n_unit))
            
        # plt.xlabel('ISI (ms)')
        # plt.legend(frameon=False)
        # plt.title(f'[braz]_ISI_[{use_example}]')
        # if SAVEFIG:
        #     plt.savefig(os.path.join(FIG_FOLDER, f'[braz]_ISI_[{use_example}].svg'))
        # plt.show()
        
        res = pg.ttest(CV, CV[0]) 
        subj.useful_clusters.at[use_example, 'stable_isi'] = res['p-val'].iloc[0]
        
        
airp_stable = sum(airp.useful_clusters.stable_isi > 0.05)
airp_unstable = sum(airp.useful_clusters.stable_isi < 0.05)
braz_stable = sum(braz.useful_clusters.stable_isi > 0.05)
braz_unstable = sum(braz.useful_clusters.stable_isi < 0.05)

plt.figure(figsize=(4,4))
plt.bar([0,0.4,1,1.4], [airp_stable, airp_unstable, braz_stable, braz_unstable], width=0.4, ec='k')
plt.xticks([0.2,1.2],['airp','braz'])
plt.ylabel('Counts')
plt.title(f'airp: {airp_stable / (airp_stable + airp_unstable) * 100 :.2f}%, braz: {braz_stable / (braz_stable + braz_unstable) * 100 :.2f}%')
if SAVEFIG:
    plt.savefig(os.path.join(FIG_FOLDER, '[all]_stable_isi.svg'))
plt.show()
    
#%% ISI example

example = airp.useful_clusters.iloc[49]
example_cluster = example.neuron.df[['session','unit_code','channel']]
# fig, ax = plt.subplots(figsize=(4,4))

CV = []

for i in range(len(example_cluster)):
    session, unit_code, channel = example_cluster.iloc[i]
    
    bmi = subj.raw_data[session]
    spike_times = bmi.pklfile['spks'].get(unit_code)
    
    isi = np.diff(spike_times) * 1000
    
    CV.append(np.std(isi) / np.mean(isi))


plt.figure(figsize=(4,4))
plt.scatter(example.days, CV, color='k')
plt.plot(example.days, CV, c='k')
plt.xlabel('Days')
plt.ylabel('Coefficient of Variation')
plt.ylim([0,2])
if SAVEFIG:
    plt.savefig(os.path.join(FIG_FOLDER, '[airp]_stable_isi_example.svg'))
plt.show()

#%% Drifting ISI examples

ids = 49
subj = airp
example = subj.useful_clusters.iloc[ids]
example_cluster = example.neuron.df[['session','unit_code','channel']]
# fig, ax = plt.subplots(figsize=(4,4))

CV = []

for i in range(len(example_cluster)):
    session, unit_code, channel = example_cluster.iloc[i]
    
    bmi = subj.raw_data[session]
    spike_times = bmi.pklfile['spks'].get(unit_code)
    
    isi = np.diff(spike_times) * 1000
    
    CV.append(np.std(isi) / np.mean(isi))


plt.figure(figsize=(4,4))
plt.scatter(example.days, CV, color='k')
plt.plot(example.days, CV, c='k')
plt.xlabel('Days')
plt.ylabel('Coefficient of Variation')
plt.ylim([0,2])
plt.title(f'{pg.ttest(CV, CV[0])["p-val"]}:.3f; {example.n_unit}')
# if SAVEFIG:
#     plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_unstable_isi_example_[{ids}].svg'))
plt.show()


#%% SFC

def sfc_of_tracked_neuron(example: pd.Series, title: str, rand: bool = False):
    """
    Plot trial-averaged SFC for each session in a tracked neuron.
    
    example: usage like airp.useful_clusters.iloc[0]
    title: for the title and the saved filename
    rand: if this is to randomized aligned points.   
    
    """
    
    start_sec, end_sec = -1, 1
    
    # plt.figure(figsize=(4,4)) # Each line is data from a neuron
    
    coherences = []
    
    for n in range(example.n_unit): # For each neuron in a cluster
        session, unit_code, channel = example.neuron.df[['session','unit_code','channel']].iloc[n]
        print(session)
        
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
                    
            # Averaged the signal
            f_start= np.argmin(np.abs(f-2))
            f_end = np.argmin(np.abs(f-6))
            
            coherence = coherogram[:, f_start:f_end].mean(1)
            coherence_std = coherogram[:, f_start:f_end].std(1) / np.sqrt(coherogram.shape[0])
            
            # plt.plot(np.linspace(start_sec,end_sec,N_timesteps), coherence, label=f'Day {example.days[n]}', c=plt.cm.rainbow(n/example.n_unit))
            # plt.fill_between(np.linspace(start_sec,end_sec,N_timesteps),
            #                  coherence + coherence_std, coherence - coherence_std,  
            #                  color=plt.cm.rainbow(n/example.n_unit), alpha=0.2)
            coherences.append(coherence)

        except:
            pass
        
        
    return np.array(coherences)
        
    # plt.legend(frameon=False)
    # plt.xlabel('Time from movement to target (sec)')
    # plt.ylabel('Coherence')
    # plt.title(title)
    # sub_folder = 'SFC_rand' if rand else 'SFC'
    # if SAVEFIG:
    #     plt.savefig(os.path.join(FIG_FOLDER, sub_folder, f'{title}.svg'))
    # plt.ylim([0, 0.4])
    # plt.show()
    
# for subj in [airp, braz]: # For each subject
subj=braz

subj.useful_clusters['stable_sfc'] = None

for k in range(len(subj.useful_clusters)): # Go through each useful cluster
    
    print(f'Cluster [{k}]')
    example = subj.useful_clusters.iloc[k]
    title = f'[{subj.subject}]_SFC_[{k}]'
    res = sfc_of_tracked_neuron(example, title)
        
    if res.ndim > 1:
        avg_coh = res[:,50:75].max(1)
        sig = pg.ttest(avg_coh,avg_coh[0])['p-val'].iloc[0]
        print(sig)
        
    else:
        sig = 1
    subj.useful_clusters.at[k, 'stable_sfc'] = sig





#%%

subj = braz


phases, amps = [],[]
for k in range(len(subj.useful_clusters)):  
    print(f'Cluster [{k}]')
    example = subj.useful_clusters.iloc[k]
    
    phase_this_neuron = []
    amp_this_neuron = []
    
    phase_ref = None
    amp_ref = None
    
    for n in range(example.n_unit):
        session, unit_code, channel = example.neuron.df[['session','unit_code','channel']].iloc[n]
        
        bmi = subj.raw_data[session]
        ns2 = bmi.ns2file
        spike_times = bmi.pklfile['spks'].get(unit_code)
        try:
            lfp = ns2.getdata()['data'][channel]
            fs = 1000
    
            beta = band_pass_filter(lfp, fs, 2, 30, 3)
            window = (-0.3,0.3) # +/- 1 second
            samples_window = (int(window[0] * fs), int(window[1] * fs))
            segments = []
            for spike in spike_times: # Using just the first 5000 spikes should be sufficient.
                spike_idx = int(spike * fs)
                if spike_idx + samples_window[0] >= 0 and spike_idx + samples_window[1] < len(lfp):
                    segments.append(beta[spike_idx + samples_window[0]: spike_idx + samples_window[1]])
            
            sta = np.mean(segments, axis=0)
            sta = slide_avg(sta, 20)
            hilb_field = signal.hilbert(sta)
            lfp_phase = np.angle(hilb_field)
            phase = lfp_phase[300]
            amp = np.max(sta) - np.min(sta)
            
            if phase_ref is None:
                phase_ref = phase
                amp_ref = amp
            
            phase_this_neuron.append(phase - phase_ref)
            amp_this_neuron.append(amp / amp_ref)
            
        except:
            pass
    
    phases.extend(phase_this_neuron)
    amps.extend(amp_this_neuron)

#%%

post_amp = np.log10(np.array(amps))
post_phases = np.rad2deg(phases)

post_phases[post_phases > 180] -= 360
post_phases[post_phases < -180] += 360

data = np.vstack((post_amp, post_phases)).T

plt.figure(figsize=(4,4))
plt.scatter(post_amp, post_phases, color='k', fc='none', ec='k', s=15, lw=0.5)
plt.xlabel('Normalized log-scaled amplitude')
plt.ylabel('Phase difference')
plt.title(subj.subject)
if SAVEFIG:
    plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_stable_sta.svg'))
plt.show()

pg.multivariate_ttest(data, [0,0])




#%% Percentages

airp.useful_clusters[['stable_isi','PD_span_pval']]

stable_isi = airp.useful_clusters[['stable_isi']]>0.05
stable_pd = airp.useful_clusters[['PD_span_pval']]<0.05

sum(stable_isi & stable_pd)

#%%

subj=airp
df = subj.useful_clusters

AB = len(df[(df['stable_isi'] > 0.05) & (df['PD_span_pval'] < 0.05)])
Ab = len(df[(df['stable_isi'] > 0.05) & (df['PD_span_pval'] > 0.05)])
aB = len(df[(df['stable_isi'] < 0.05) & (df['PD_span_pval'] < 0.05)])
ab = len(df[(df['stable_isi'] < 0.05) & (df['PD_span_pval'] > 0.05)])


plt.figure(figsize=(4,4))
venn2([Ab,aB,AB])
if SAVEFIG:
    plt.savefig(os.path.join(FIG_FOLDER, f'[{subj.subject}]_venn.svg'))
plt.show()


#%%

for subj in [airp, braz]:
    res = []
    for i in range(len(subj.useful_clusters)):
        df = subj.useful_clusters.iloc[i]
        neuron = df.neuron
        PD = minimal_PD_change(neuron.df.PD.copy()) 
        for i in np.diff(PD):
            res.append(i)
    # rand_diff = np.array([np.diff(np.random.random(2)*360)[0] for _ in range(len(res))])
    # rand_diff[rand_diff > 180] -= 360
    # rand_diff[rand_diff < -180] += 360
    
    print(pg.ttest(res, 0).squeeze())