"""Script to cut the EEG data into parts corresponding to the stimuli timestamps."""

import argparse
import math
import os
import pandas as pd

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

LOG_KEYWORDS = ["start_relax_logging",
                "end_relax_logging",
                "start_baseline_logging",
                "end_baseline_logging",
                "start_vid",
                "end_vid"
                ]

AFFECTS = {"Valence", "Arousal"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cuts the csv data which was recorded during an Experiment."
    )
    parser.add_argument(
        "eegFile", type=str, help="EEG input filename including path")
    parser.add_argument(
        "logFile", type=str, help="Log input filename including path")
    parser.add_argument(
        "labelFile", type=str, help="Label input filename including path")
    parser.add_argument("subjectId", type=str, help="subject ID")
    parser.add_argument("device", type=str, help="c(rown) or m(use)", choices=['c', 'm'])
    parser.add_argument("--out", type=str, help="output path"
                                                "Default: '<path_to_file>/preprocessed/'")
    return vars(parser.parse_args())


def read_timestamps(log_path):
    timestamps = OrderedDict()

    with open(log_path, 'r') as log_file:
        log_file = log_file.readlines()

    for line in log_file:
        for keyword in LOG_KEYWORDS:
            if keyword in line:
                rest, timestamp = line.split(": ")
                timestamp = datetime.fromtimestamp(float(timestamp.strip()))
                if "vid" not in keyword:
                    timestamps[keyword] = timestamp
                else:
                    x, id_logging = rest.split("_vid")
                    video_keyword = f"{keyword}{id_logging}"
                    timestamps[video_keyword] = timestamp

    return list(timestamps.items())


def find_label(timestamps, ts):
    for i in range(0, len(timestamps), 2):
        if timestamps[i][1] <= ts <= timestamps[i + 1][1]:
            return timestamps[i][0].split("_")[1]
    return "Other"


def add_seq_label(eeg_path, timestamps):
    df_eeg = pd.read_csv(eeg_path, on_bad_lines='skip')
    df_eeg = df_eeg.astype({'timeStamp': 'datetime64[ns]'})

    df_eeg["Label"] = df_eeg["timeStamp"].apply(lambda ts: find_label(timestamps, ts))

    return df_eeg


def add_affect_label(label_path, df_eeg):
    df_av = pd.read_csv(label_path, on_bad_lines='skip')

    df_eeg["Valence"] = df_eeg["Label"].apply(lambda label: get_affect_response(label, "Valence", df_av))
    df_eeg["Arousal"] = df_eeg["Label"].apply(lambda label: get_affect_response(label, "Arousal", df_av))

    return df_eeg


def get_affect_response(label, affect, df_av):
    if affect not in AFFECTS:
        raise ValueError("get_affect_response: affect must be one of %r." % AFFECTS)
    if "vid" in label:
        vid_id = label.split("d")[1]
        res1, res2 = df_av.loc[df_av['ID'] == float(vid_id), f"slider{affect}_1.response"].iloc[0], df_av.loc[df_av['ID'] == float(vid_id), f"slider{affect}_2.response"].iloc[0]
        return res1 if math.isnan(res2) else res2
    return None


def write_output(eeg_path, out_path, subject_id, df):
    if not out_path:
        dirname = os.path.dirname(eeg_path)
        out_path = os.path.join(dirname, "preprocessed", subject_id)
        # create output folder in case it does not exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    file = os.path.basename(eeg_path)
    base, tail = file.split("_cleaned")
    file_name = f"{base}_labelled.csv"
    file_out = os.path.join(out_path, file_name)
    df.to_csv(file_out, index=False)


def main():
    args = parse_args()
    eeg_path = Path(args["eegFile"])
    log_path = Path(args["logFile"])
    label_path = Path(args["labelFile"])
    subject_id = args["subjectId"]
    device = args["device"]
    out_path = args["out"]

    timestamps = read_timestamps(log_path)
    print(timestamps)

    df_seqs = add_seq_label(eeg_path, timestamps)
    df_labelled = add_affect_label(label_path, df_seqs)

    write_output(eeg_path, out_path, subject_id, df_labelled)


if __name__ == '__main__':
    main()
