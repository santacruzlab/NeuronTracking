"""
SFC (Spike-Field Coherence) calculation for a single neuron.

Usage:
    Configure the parameters in the "USER PARAMETERS" section below,
    then run the script. It will compute and plot the SFC for the specified
    session / unit / channel combination.
"""

# ── imports ──────────────────────────────────────────────────────────────────
import os
import sys
import pickle
import scipy
import tables
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# ── path setup (mirror the original project layout) ──────────────────────────
SCRIPT_FOLDER  = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SCRIPT_FOLDER)
BMI_FOLDER     = os.path.join(os.path.dirname(PROJECT_FOLDER), 'bmi_python')
NS_FOLDER      = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')

sys.path.insert(0, BMI_FOLDER)
sys.path.insert(0, NS_FOLDER)


#Fill in these file locations based on your file structure
HDF_FOLDER     = r""
MAT_FOLDER     = r""
NEV_OUTPUT_FOLDER = r""

# ── lazy imports that depend on sys.path ─────────────────────────────────────
from riglib.blackrock.brpylib import NsxFile          # noqa: E402  (path set above)

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS  ← edit these for each run
# ─────────────────────────────────────────────────────────────────────────────
SESSION    = "airp20220211_05_te1812"   # session prefix (no extension) ex: airp20220211_05_te1812
UNIT_CODE  = "2a"                      # unit code, e.g. "2a"
CHANNEL    = 22                         # LFP channel index (0-based in ns2 data array)

# Storage path for this session's Ripple (ns2) files (name the folder containing the ripple folder here)
SAVE_FOLDER = r""    

# SFC window around movement-onset (seconds)
START_SEC  = -1.0
END_SEC    =  1.0

# Frequency band for the summary coherence trace (Hz)
FREQ_LOW   = 2
FREQ_HIGH  = 6

# Number of time-steps in the coherogram
N_TIMESTEPS = 100

# Block / error-clamp filters
BLOCK_TYPE   = 1
ERROR_CLAMP  = 0

# Ripple sample rate
RIPPLE_FS = 30_000

SAVEFIG = True
FIG_FOLDER = os.path.join(PROJECT_FOLDER, 'plots') #where the coherogram and SFC plot will be saved
# ─────────────────────────────────────────────────────────────────────────────


# ── helper functions (self-contained copies from tracking.py) ───────

def gaussian(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to a 1-D data array."""
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode='same')


def slide_avg(array: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window average."""
    avg = np.zeros_like(array)
    for i in range(len(array)):
        if i < window:
            avg[i] = array[:i + 1].sum() / (i + 1)
        else:
            avg[i] = array[i - window + 1: i + 1].sum() / window
    return avg


def calc_firing_rate(spike_times: np.ndarray,
                     window_size: float = 0.5,
                     step_size: float  = 0.05,
                     box_size: int     = 20,
                     kernel_size: float = 1.0) -> np.ndarray:
    """
    Estimate firing rate from spike times (seconds).
    Returns an array sampled at 1/step_size Hz.
    """
    time_bins    = np.arange(0, spike_times.max(), step_size)
    firing_rate  = np.zeros_like(time_bins)
    for i, t in enumerate(time_bins):
        count = np.sum((spike_times >= t) & (spike_times < t + window_size))
        firing_rate[i] = count / window_size
    firing_rate = gaussian(slide_avg(firing_rate, box_size), kernel_size)
    return firing_rate


def band_pass_filter(lfp_signal: np.ndarray,
                     fs: int   = 1000,
                     low: float = 12,
                     high: float = 30,
                     order: int  = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, lfp_signal)


def calc_spectrum(spike: np.ndarray, field: np.ndarray, fs: int):
    """
    Return (Sxx, Syy, Sxy): field auto-spectrum, spike auto-spectrum,
    and field-spike cross-spectrum.
    """
    assert len(spike) == len(field)
    N_pts     = len(spike)
    field_ts  = (field - field.mean()) * np.hanning(N_pts)
    spike_ts  = spike - spike.mean()
    F         = np.fft.rfft(field_ts)
    S         = np.fft.rfft(spike_ts)
    Sxx = np.real(F * np.conj(F))
    Syy = np.real(S * np.conj(S))
    Sxy = F * np.conj(S)
    return Sxx, Syy, Sxy


# ── file-loading helpers ──────────────────────────────────────────────────────

def load_hdf(session: str) -> tables.File:
    path = os.path.join(HDF_FOLDER, session + '.hdf')
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF not found: {path}")
    return tables.open_file(path)


def load_mat(session: str) -> dict:
    path = os.path.join(MAT_FOLDER, session + '_syncHDF.mat')
    if not os.path.exists(path):
        raise FileNotFoundError(f"MAT not found: {path}")
    return scipy.io.loadmat(path)


def load_pkl(session: str) -> dict:
    path = os.path.join(NEV_OUTPUT_FOLDER, session + '_nev_output.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"PKL not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_ns2(session: str, save_folder: str) -> NsxFile:
    path = os.path.join(save_folder, 'ripple', session + '.ns2')
    if not os.path.exists(path):
        raise FileNotFoundError(f"NS2 not found: {path}")
    return NsxFile(path)


def hdf_to_sample(hdf_states: np.ndarray, hdf_times: dict) -> np.ndarray:
    """Convert HDF row indices to Ripple sample numbers (linear interpolation)."""
    sample_number = np.zeros(hdf_states.size)
    hdf_rows = hdf_times['row_number'][0]
    ripple   = hdf_times['ripple_samplenumber'][0]

    for i in range(len(hdf_states)):
        idx = np.argmin(np.abs(hdf_rows - hdf_states[i]))
        if np.abs(hdf_rows[idx] - hdf_states[i]) == 0:
            sample_number[i] = ripple[idx]
        elif hdf_rows[idx] > hdf_states[i]:
            diff = hdf_rows[idx] - hdf_rows[idx - 1]
            m = (ripple[idx] - ripple[idx - 1]) / diff
            b = ripple[idx - 1] - m * hdf_rows[idx - 1]
            sample_number[i] = int(m * hdf_states[i] + b)
        elif idx + 1 < len(hdf_rows):
            diff = hdf_rows[idx + 1] - hdf_rows[idx]
            if diff > 0:
                m = (ripple[idx + 1] - ripple[idx]) / diff
                b = ripple[idx] - m * hdf_rows[idx]
                sample_number[i] = int(m * hdf_states[i] + b)
            else:
                sample_number[i] = ripple[idx]
        else:
            sample_number[i] = ripple[idx]

    return sample_number


# ── main SFC function ─────────────────────────────────────────────────────────

def compute_sfc_single_neuron(
        session:    str,
        unit_code:  str,
        channel:    int,
        save_folder: str,
        start_sec:  float = START_SEC,
        end_sec:    float = END_SEC,
        freq_low:   float = FREQ_LOW,
        freq_high:  float = FREQ_HIGH,
        n_timesteps: int  = N_TIMESTEPS,
        block_type: int   = BLOCK_TYPE,
        error_clamp: int  = ERROR_CLAMP,
        rand:       bool  = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the spike-field coherogram for a single neuron.

    Parameters
    ----------
    session     : session prefix string
    unit_code   : unit identifier, e.g. "22a"
    channel     : LFP channel index (0-based) in the ns2 data array
    save_folder : root folder containing ``ripple/<session>.ns2``
    start_sec   : window start relative to alignment point (s)
    end_sec     : window end   relative to alignment point (s)
    freq_low    : lower bound of summary frequency band (Hz)
    freq_high   : upper bound of summary frequency band (Hz)
    n_timesteps : number of time-steps across the window
    block_type  : task block to include
    error_clamp : error-clamp flag filter (0 = non-clamped)
    rand        : if True, randomise alignment points (null distribution)

    Returns
    -------
    coherogram  : (n_timesteps, n_freqs) array of coherence values
    coherence   : (n_timesteps,) band-averaged coherence trace
    f           : (n_freqs,) frequency axis (Hz)
    """

    # ── load data ────────────────────────────────────────────────────────────
    print(f"[{session}] Loading data…")
    hdffile  = load_hdf(session)
    matfile  = load_mat(session)
    pklfile  = load_pkl(session)
    ns2      = load_ns2(session, save_folder)

    # ── parse behavioural events ─────────────────────────────────────────────
    task_msg  = hdffile.root.task_msgs[:]['msg']
    task_time = hdffile.root.task_msgs[:]['time']

    hdf_reward     = np.array([i for i, m in enumerate(task_msg) if m == b'reward'])
    hdf_target     = hdf_reward - 3
    time_target    = task_time[hdf_target]
    rpp_target     = hdf_to_sample(time_target, matfile)

    # block / clamp index
    error_clamp_arr = hdffile.root.task[:]['error_clamp'].flatten()
    block_type_arr  = hdffile.root.task[:]['block_type'].flatten()
    target_pos      = hdffile.root.task[:]['target']

    hdf_holdcenter     = hdf_reward - 5
    time_holdcenter    = task_time[hdf_holdcenter]

    pos_x = target_pos[:, 0]
    pos_y = target_pos[:, 2]
    angle = np.arctan2(pos_y, pos_x) * 180 / np.pi
    angle[angle < 0] += 360

    import pandas as pd
    index = pd.DataFrame({
        'error_clamp': error_clamp_arr[time_holdcenter],
        'block_type':  block_type_arr[time_holdcenter],
        'direction':   angle[time_target],
        'trial_number': np.arange(len(hdf_reward)),
    })

    # ── select trials ────────────────────────────────────────────────────────
    trial      = index[(index['block_type'] == block_type) &
                       (index['error_clamp'] == error_clamp)]
    align_pts  = np.array(rpp_target[trial['trial_number']] / 30, dtype=int)

    if rand:
        align_pts = (np.random.random(len(align_pts)) * align_pts[-1]).astype(int)

    # ── load LFP and estimate firing rate ────────────────────────────────────
    print(f"[{session}] Reading LFP channel {channel}…")
    lfp = ns2.getdata()['data'][channel]
    fs  = 1000  # ns2 sampling rate (Hz)

    spks = pklfile['spks']

# Convert unit_code to string if necessary
    unit_code = str(unit_code)

    if unit_code not in spks:
        print(f"[{session}] Unit {unit_code} not found in spike file.")
        print(f"Available units: {list(spks.keys())[:10]} ...")
        hdffile.close()
        return None, None, None

    spike_times = spks[unit_code]

    fr      = calc_firing_rate(spike_times)
    fr_rate = 20  # Hz (1 / step_size = 1 / 0.05)
    fr_time = np.arange(0, len(fr) / fr_rate, 1 / fr_rate)
    lfp_time = np.arange(0, len(lfp) / fs, 1 / fs)
    interp_func = interp1d(fr_time, fr, kind='linear', fill_value='extrapolate')
    fr = interp_func(lfp_time)

    # ── build coherogram ─────────────────────────────────────────────────────
    N_trials  = len(align_pts)
    N_pts     = int(0.6 * fs)           # window used for each spectrum estimate
    N_freqs   = N_pts // 2 + 1
    f         = np.fft.rfftfreq(N_pts, 1 / fs)

    fine_align_pts = np.zeros((N_trials, n_timesteps), dtype=int)
    for t in range(N_trials):
        fine_align_pts[t] = np.linspace(
            align_pts[t] + start_sec * fs,
            align_pts[t] + end_sec   * fs,
            n_timesteps, dtype=int)

    coherogram = np.zeros((n_timesteps, N_freqs))

    print(f"[{session}] Computing coherogram ({n_timesteps} time steps, "
          f"{N_trials} trials)…")
    for ts in range(n_timesteps):
        Sxx = np.zeros(N_freqs)
        Syy = np.zeros(N_freqs)
        Sxy = np.zeros(N_freqs, dtype=complex)

        for t in range(N_trials):
            pt = fine_align_pts[t, ts]
            # guard against index out of bounds
            if pt - N_pts // 2 < 0 or pt + N_pts // 2 >= len(lfp):
                continue
            field_raw = lfp[pt - N_pts // 2: pt + N_pts // 2]
            spike_raw = fr[pt - N_pts // 2: pt + N_pts // 2]
            sxx, syy, sxy = calc_spectrum(spike_raw, field_raw, fs)
            Sxx += sxx / N_trials
            Syy += syy / N_trials
            Sxy += sxy / N_trials

        coherogram[ts] = np.abs(Sxy) / np.sqrt(Syy) / np.sqrt(Sxx)

    # ── band-average ─────────────────────────────────────────────────────────
    f_start   = np.argmin(np.abs(f - freq_low))
    f_end     = np.argmin(np.abs(f - freq_high))
    coherence = coherogram[:, f_start:f_end].mean(axis=1)

    hdffile.close()
    return coherogram, coherence, f


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_sfc(coherogram: np.ndarray,
             coherence:  np.ndarray,
             f:          np.ndarray,
             session:    str,
             unit_code:  str,
             start_sec:  float = START_SEC,
             end_sec:    float = END_SEC,
             freq_low:   float = FREQ_LOW,
             freq_high:  float = FREQ_HIGH,
             savefig:    bool  = SAVEFIG,
             fig_folder: str   = FIG_FOLDER):
    """Two-panel figure: coherogram (left) and band-averaged trace (right)."""

    t_axis = np.linspace(start_sec, end_sec, coherogram.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # --- left: full coherogram ---
    ax = axes[0]
    im = ax.pcolormesh(t_axis, f, coherogram.T,
                       cmap='viridis', shading='auto')
    ax.axvline(0, color='w', lw=1, ls='--')
    ax.axhline(freq_low,  color='r', lw=0.8, ls='--', label=f'{freq_low} Hz')
    ax.axhline(freq_high, color='r', lw=0.8, ls='--', label=f'{freq_high} Hz')
    ax.set_ylim([0, 50])
    ax.set_xlabel('Time from movement onset (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Coherogram')
    fig.colorbar(im, ax=ax, label='Coherence')
    ax.legend(frameon=False, fontsize=8)

    # --- right: band-averaged coherence ---
    ax = axes[1]
    ax.plot(t_axis, coherence, color='k', lw=1.5)
    ax.axvline(0, color='gray', lw=1, ls='--')
    ax.set_xlabel('Time from movement onset (s)')
    ax.set_ylabel(f'Mean coherence ({freq_low}–{freq_high} Hz)')
    ax.set_title('Band-averaged SFC')

    fig.suptitle(f'SFC  |  {session}  |  unit {unit_code}', fontsize=11)
    fig.tight_layout()

    if savefig:
        os.makedirs(fig_folder, exist_ok=True)
        fname = os.path.join(fig_folder, f'SFC_{session}_{unit_code}.svg')
        fig.savefig(fname)
        print(f"Figure saved: {fname}")

    plt.show()
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    coherogram, coherence, f = compute_sfc_single_neuron(
        session     = SESSION,
        unit_code   = UNIT_CODE,
        channel     = CHANNEL,
        save_folder = SAVE_FOLDER,
    )

    plot_sfc(
        coherogram = coherogram,
        coherence  = coherence,
        f          = f,
        session    = SESSION,
        unit_code  = UNIT_CODE,
    )

    