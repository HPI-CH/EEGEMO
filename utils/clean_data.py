"""Script to clean the data collected with the Muse or Crown device."""

import argparse
import pandas as pd

from datetime import datetime
from pathlib import Path

MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]
CROWN_CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleans the csv data which was recorded with Muse or Crown."
    )
    parser.add_argument(
        "filename", type=str, help="Input filename including path")
    parser.add_argument("device", type=str, help="c(rown) or m(use)", choices=['c', 'm'])
    parser.add_argument("--out", type=str, help="Output filename including path."
                                                "Default: '<path_to_file>/<filename>_cleaned'")
    return vars(parser.parse_args())


def clean(df, channels):
    # remove corrupted timestamps
    df['timeStamp'] = df['timeStamp'].apply(lambda ts: remove_corrupted_time(ts))
    # remove rows with corrupted cells
    df_eeg = df[channels].applymap(
        lambda val: pd.NaT if (len(str(val).split(" ")) > 1) else val)
    df = df.drop(channels, axis=1)
    df_con = pd.concat([df, df_eeg], axis=1)

    df_con = df_con.dropna()
    df_con = df_con.reset_index(drop=True)

    # timestamp formatting
    df_con['timeStamp'] = df_con['timeStamp'].apply(lambda ts: datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f'))

    return df_con


def remove_corrupted_time(ts):
    try:
        if len(str(ts).split(" ")) > 2:
            return pd.NaT
        else:
            return ts
    except TypeError:
        return pd.NaT


def clean_muse(df):
    df_clean = clean(df, MUSE_CHANNELS)
    # add second timestamp
    df_fin = add_elapsed_time(df_clean, "timeStamp")

    return df_fin


def clean_crown(df):
    # convert unix time stamp
    df['timeStampCrown'] = df['timeStampCrown'].apply(
        lambda ts: pd.to_datetime(ts, unit='ms'))
    # sort by timeStamp
    df = df.sort_values(by='timeStampCrown')

    df_clean = clean(df, CROWN_CHANNELS)
    # add second timestamp
    df_fin = add_elapsed_time(df_clean, "timeStampCrown")

    return df_fin


def add_elapsed_time(df, col):
    df['time_elapsed'] = df['timeStamp'].apply(lambda ts: ts - df[f"{col}"].iloc[0])
    df = df.iloc[1:, :]
    df['time_elapsed'] = df['time_elapsed'].apply(lambda ts: (time_formatting(ts)).strftime('%M:%S'))
    return df


def time_formatting(ts):
    try:
        return datetime.strptime(str(ts).split()[-1], '%H:%M:%S.%f')
    except ValueError:
        return datetime.strptime(str(ts).split()[-1], '%H:%M:%S')


def write_output(file_path, out_path, df):
    if out_path:
        file_out = out_path
    else:
        file_out = f"{file_path.with_suffix('')}_cleaned"
        file_out = Path(file_out).with_suffix('.csv')

    df.to_csv(file_out, index=False)


def main():
    args = parse_args()
    file_path = Path(args["filename"])
    device = args["device"]
    out_path = args["out"]

    df_eeg = pd.read_csv(file_path, on_bad_lines='skip')
    df_eeg = df_eeg.dropna(subset=['timeStamp'])
    df_eeg = df_eeg.drop(columns=['no'])

    if device == "m":
        df_cleaned = clean_muse(df_eeg)
    else:
        df_cleaned = clean_crown(df_eeg)

    print(df_cleaned.dtypes)
    write_output(file_path, out_path, df_cleaned)


if __name__ == '__main__':
    main()
