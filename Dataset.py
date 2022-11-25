import datetime as dt
import pandas as pd
import numpy as np


class Dataset:
    """Interface for dataset classes"""
    eeg_channels = []
    column_names = []
    sampling_frequency = int
    length = int

    def __init__(self, dirpath):
        self.dirpath = dirpath

    def add_window(self, t, data):
        data['window'] = data.index // (self.sampling_frequency * t)
        return data

    def add_timestamp(self, data):
        random_date = dt.datetime(2022, 3, 25, 13, 21)
        data['timeStamp'] = data.index // self.sampling_frequency
        data['timeStamp'] = data['timeStamp'].apply(lambda v: random_date + dt.timedelta(seconds=v))
        return data

    def format_labels(self, labels, data_length):
        labels_val = list()
        labels_ar = list()

        for i in range(len(data_length)):
            labels_val = np.append(labels_val, np.full(data_length[i], labels['valence'].loc[i]))
            labels_ar = np.append(labels_ar, np.full(data_length[i], labels['arousal'].loc[i]))

        df_val = pd.DataFrame(labels_val, columns=['Valence'])
        df_ar = pd.DataFrame(labels_ar, columns=['Arousal'])
        df_labels = pd.concat([df_val, df_ar], axis=1)

        return df_labels
