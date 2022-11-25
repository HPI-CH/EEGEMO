from Dataset import Dataset
import pandas as pd
import os
from scipy.io import loadmat


class Amigos(Dataset):
    eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    column_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "ECG Right",
                    "ECG Left",
                    "GSR"]
    sampling_frequency = 128
    length = 40
    name = "Amigos"
    out_path = os.path.join('./output_data', 'amigos')

    def __init__(self, dirname):
        super().__init__(dirname)

    def load_labels(self, user_id):
        # Labels
        all_labels = pd.read_excel((os.path.join(self.dirpath, "SelfAsessment.xlsx")),
                                   sheet_name="Experiment_1", header=1, usecols="A:C,P:S",
                                   names=["UserID", "VideoID", "Rep_Index", "arousal", "valence", "dominance",
                                          "liking"])

        user_labels = all_labels.loc[all_labels['UserID'] == user_id].sort_values(by=['Rep_Index'])
        user_labels = user_labels.reset_index()

        # Split into high/low arousal and valence (threshold 5)
        user_labels['valence'] = user_labels['valence'].apply(lambda x: 1 if x >= 5 else 0)
        user_labels['arousal'] = user_labels['arousal'].apply(lambda x: 1 if x >= 5 else 0)

        return user_labels

    def load_eeg(self, user_id):
        user_id_str = str(user_id) if user_id > 9 else f"0{str(user_id)}"
        eeg_prep = loadmat((os.path.join(self.dirpath, f"Data_Preprocessed_P{user_id_str}",
                                         f"Data_Preprocessed_P{user_id_str}.mat")))

        return eeg_prep

    def cut_to_window_size(self, df, window_sec):
        # remove first x samples so that n*(window_sec*sampling_frequency) per video remain
        n = len(df.index) % (self.sampling_frequency*window_sec)
        df.drop(index=df.index[:n], axis=0, inplace=True)
        return df

    def prepare_data(self, user_id, window_sec):
        eeg_data = self.load_eeg(user_id)
        labels = self.load_labels(user_id)
        # data for all short videos
        df_eeg = pd.DataFrame(columns=self.eeg_channels)
        sample_size = list()
        # remove the last 4 because I only looked at the short videos (the last 4 are the long ones)
        for vid_id, dataset in enumerate(eeg_data['joined_data'][0][:-4]):
            data = [[row.flat[0] for row in line] for line in dataset]
            df = pd.DataFrame(data, columns=self.column_names)
            df = df.drop(columns=["ECG Right", "ECG Left", "GSR"])
            df['Label'] = vid_id
            df = self.cut_to_window_size(df, window_sec)
            # add window column
            df = self.add_window(window_sec, df)
            df_eeg = pd.concat([df_eeg, df], axis=0, ignore_index=True)
            sample_size.append(len(df.index))

        # add timestamp for elapsed time
        df_eeg = self.add_timestamp(df_eeg)

        df_labels = self.format_labels(labels, sample_size)

        return df_eeg, df_labels
