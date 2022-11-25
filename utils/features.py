"""Functions to extract features from the EEG data"""

from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
from itertools import combinations
import numpy as np

BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 16), 'Beta': (16, 32), 'Gamma': (32, 45)}
BP_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'TotalAbsPower']


def get_names(ch_names):
    return np.array([f'{ch_name}_{band}' for ch_name in ch_names for band in BP_NAMES])


# Since the YASA package destroyed my venvs, this is adapted from the package
def bandpower_all(data, sf, window_sec, ch_names, bands=None, method='welch', relative=False):
    if bands is None:
        bands = [(0.5, 4, 'Delta'),
                 (4, 8, 'Theta'), (8, 16, 'Alpha'),
                 (16, 32, 'Beta'), (32, 45, 'Gamma')]
    data = data.T
    data = data.to_numpy()
    # Compute the modified periodogram (Welch)
    if method == 'welch':
        nperseg = window_sec * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]
    nchan = psd.shape[0]

    ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))

    bp = np.zeros((nchan, len(bands)), dtype=np.float64)
    psd = psd[:, idx_good_freq]
    total_power = simps(psd, dx=res)
    total_power = total_power[..., np.newaxis]

    # Check if there are negative values in PSD
    if (psd < 0).any():
        msg = (
            "There are negative values in PSD. This will probably result in incorrect "
            "bandpower values.")
        print(msg)

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[:, i] = simps(psd[:, idx_band], dx=res)

    if relative:
        bp /= total_power

    # reformatting to fit the input shape for the river models
    bp_dict = {}
    bp = np.append(bp, total_power, axis=1)
    names = get_names(ch_names)
    for name, val in zip(names, bp.flatten()):
        bp_dict[name] = val

    return bp_dict, psd, freqs


def band_ratio(dd, channel, band_a, band_b):
    """Ratio of two bands, df already contains the calculated bandpower of all bands (for one or multiple sensors)"""
    return dd[f'{channel}_{band_a}'] / dd[f'{channel}_{band_b}']


def band_ratio_all(bp_dict, ch_names):
    """Ratio of bands, respectively"""
    df_ratios = {}
    for channel in ch_names:
        for bands in combinations(BP_NAMES[:-1], 2):
            col_name = f'Ratio_{channel}_{bands[0][0]}{bands[1][0]}'
            df_ratios[col_name] = band_ratio(bp_dict, channel, bands[0], bands[1])

    return df_ratios
