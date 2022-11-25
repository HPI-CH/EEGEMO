"""Provides different filters for EEG data"""

from scipy.signal import butter, iirnotch, filtfilt


def notch(fs, f0=50, Q=50):
    b, a = iirnotch(f0, Q, fs)
    return a, b


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return a, b


def apply_filter(df, a, b):
    df_filtered = df.apply(lambda col: filtfilt(b, a, col), axis=0)
    return df_filtered


# Averages to common reference
def avg_common_ref(df):
    df_avg = df.sub(df.mean(axis=1), axis=0)
    return df_avg


def avg_ref(df):
    datac = df.copy()
    mean = datac.mean(axis=1)
    averaged = datac.apply(lambda v: v - mean)

    return averaged
