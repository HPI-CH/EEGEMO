"""Script to make two .xlsx files which contain the stimuli paths for the two experiment parts.
    The 4x4 videos (4 categories) get chosen randomly into two groups of 8 so that there are
    the names of 2 videos of each category in each file"""

import argparse
import pandas
from pathlib import Path
from random import shuffle

CATEGORIES = ["HVHA", "HVLA", "LVHA", "LVLA"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Creates two .xlsx files which contain the stimuli paths for the two experiment parts"
    )
    parser.add_argument("filename", type=str, help="Input filename including path")

    return vars(parser.parse_args())


def write_output(file_in, df, part):
    file_out = f"{file_in.with_suffix('')}_part{part}"
    file_out = Path(file_out).with_suffix('.xlsx')

    df.to_excel(file_out, index=False)


def main():
    args = parse_args()
    file_path = Path(args["filename"])
    column_names = ["ID", "vidName", "affect", "source_dataset", "source_movie"]
    vids1 = pandas.DataFrame(columns=column_names)
    vids2 = pandas.DataFrame(columns=column_names)
    indices = [0, 1, 2, 3]

    df = pandas.read_excel(file_path)

    for cat in CATEGORIES:
        df_affect = df.loc[df['affect'] == cat]
        df_affect = df_affect.reset_index(drop=True)
        shuffle(indices)
        print(df_affect.iloc[indices[:2]])
        vids1 = vids1.append(df_affect.iloc[indices[:2]])
        vids2 = vids2.append(df_affect.iloc[indices[-2:]])

    vids1 = vids1.reset_index(drop=True)
    vids2 = vids2.reset_index(drop=True)

    write_output(file_path, vids1, 1)
    write_output(file_path, vids2, 2)


if __name__ == '__main__':
    main()
