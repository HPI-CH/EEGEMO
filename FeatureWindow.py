from collections import deque
from utils.filter import notch, apply_filter, butter_bandpass, avg_ref
import pandas as pd

from utils import features


class FeatureWindow:
    def __init__(self, window_sec, sf, ch_names, live):
        self.y = int
        self.x = {}
        self.xQue = deque([])
        self.sampling_frequency = sf
        self.window_sec = window_sec
        self.size = sf * window_sec
        self.eeg_channels = ch_names
        self.live = live

    def next(self, x, y):
        if len(self.xQue) < self.size - 1:
            self.xQue.appendleft(x)
        elif len(self.xQue) == self.size - 1:
            self.xQue.appendleft(x)
            self.y = y
            df = pd.DataFrame(self.xQue, columns=self.eeg_channels)
            if self.live:
                # filter data
                self.prep_data(df)
            else:
                # do feature extraction
                self.extract_features(df)
            return True
        else:
            self.xQue.clear()
            self.xQue.appendleft(x)
        return False

    def prep_data(self, df):
        df_eeg = df.filter(self.eeg_channels)
        a, b = notch(self.sampling_frequency)
        df_notch = apply_filter(df_eeg, a, b)
        c, d = butter_bandpass(0.5, 45, fs=self.sampling_frequency)
        df_notch_band = apply_filter(df_notch, c, d)
        df_ref = avg_ref(df_notch_band)

        self.extract_features(df_ref)

    def extract_features(self, df):
        bp, psd, freqs = features.bandpower_all(df[[*self.eeg_channels]], self.sampling_frequency,
                                                window_sec=self.window_sec,
                                                ch_names=self.eeg_channels, relative=True)
        ratios = features.band_ratio_all(bp, self.eeg_channels)

        feature_dict = {**bp, **ratios}

        self.x = feature_dict
