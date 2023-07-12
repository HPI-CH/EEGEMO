from Dataset import Dataset
import pandas as pd
import os


class MyData(Dataset):
    eeg_channels = []
    column_names = []
    sampling_frequency = 256
    length = 11
    crown_channels = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
    muse_channels = ["TP9", "AF7", "AF8", "TP10"]
    name = "My Dataset"
    out_path = os.path.join('./output_data', 'mydata')

    def __init__(self, dirname, device):
        super().__init__(dirname)
        self.device, self.eeg_channels = self.set_device_and_channels(device)

    def set_device_and_channels(self, device):
        if device == 'm':
            return "muse", self.muse_channels
        elif device == 'c':
            return "crown", self.crown_channels
        else:
            raise ValueError('Unrecognized value for device. Device can either be Muse or Crown')

    def load_labels_from_excel(self, user_id):
        # Labels
        all_labels = pd.read_excel((os.path.join(self.dirpath, "SelfAssessment.xlsx")), header=1, usecols="A:H",
                                   names=["UserID", "SessionID", "Device", "VideoID", "Rep_Index", "arousal", "valence",
                                          "familiarity"])

        user_labels = all_labels.loc[all_labels['UserID'] == user_id].sort_values(by=['SessionID', 'Rep_Index'])
        user_labels = user_labels.reset_index()
        user_labels.astype({'valence': 'float', 'arousal': 'float'}).dtypes

        # Split into high/low arousal and valence (threshold 0.5)
        user_labels['valence'] = user_labels['valence'].apply(lambda x: 1 if x > 0.5 else 0)
        user_labels['arousal'] = user_labels['arousal'].apply(lambda x: 1 if x > 0.5 else 0)

        print(user_labels)

        return user_labels

    def cut_to_window_size(self, df, window_sec):
        # # remove first x samples so that n*(window_sec*sampling_frequency) per video remain
        df_cut = pd.DataFrame(columns=self.eeg_channels)
        for vid_id in df['Label'].unique():
            df_vid = df.loc[df['Label'] == vid_id]
            df_vid.reset_index(drop=True, inplace=True)
            n = len(df_vid.index) % (self.sampling_frequency * window_sec)
            df_copy = df_vid.copy()
            df_copy.drop(index=df_copy.index[:n], axis=0, inplace=True)
            df_cut = pd.concat([df_cut, df_copy], axis=0, ignore_index=True)
        return df_cut

    def prepare_data(self, user_id, window_sec):
        path = os.path.join(self.dirpath, f"subject_{str(user_id)}", f"{user_id}_part1_{self.device}_prep.csv")

        try:
            df_eeg = pd.read_csv(path)
        except FileNotFoundError:
            path = os.path.join(self.dirpath, f"subject_{str(user_id)}", f"{user_id}_part2_{self.device}_prep.csv")
            df_eeg = pd.read_csv(path)

        df_eeg = df_eeg.loc[df_eeg['Label'].str.contains("vid")]
        df_eeg = self.cut_to_window_size(df_eeg, window_sec)
        df_labels = df_eeg.filter(['Valence', 'Arousal'], axis=1)
        df_eeg.drop(columns=["Valence", "Arousal"], inplace=True)
        df_labels['Valence'] = df_labels['Valence'].apply(lambda x: 1 if x >= 0.5 else 0)
        df_labels['Arousal'] = df_labels['Arousal'].apply(lambda x: 1 if x >= 0.5 else 0)

        df_eeg = self.add_window(window_sec, df_eeg)

        return df_eeg, df_labels
