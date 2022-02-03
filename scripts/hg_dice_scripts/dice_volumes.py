import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from dice_config import Configuration
from dice_stats import calculate_pval
from scipy.stats.stats import pearsonr

from scripts.fs_lut import fs_lut


def extract_synthseg_vols(config, file_name):
    flag = 1 if "mri" in os.path.basename(file_name) else 0
    df = pd.read_csv(file_name, skiprows=flag, header=0)

    # TODO: this flag is annoying, either see if I can change it myself or
    # ask BB for help
    if flag:
        df = df.rename(columns={"Unnamed: 0": "subjects"})

    df["subjects"] = df["subjects"].str.slice(0, 7)
    df = df.set_index("subjects")

    df.index.name = None

    df = combine_pairs(df, config.LABEL_PAIRS)
    df = df.drop(
        columns=[column for column in df.columns if "(" not in column])

    return df


def print_correlation_pairs(config, *args, flag=None, suffix=""):
    x, y, z = args
    common_labels = x.index.intersection(y.index).intersection(z.index)
    common_labels = sorted(set(common_labels) - set(config.IGNORE_SUBJECTS))

    x = x.loc[common_labels]
    y = y.loc[common_labels]
    z = z.loc[common_labels]

    col_names = x.columns

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(
            os.path.join(config.SYNTHSEG_RESULTS, "volumes",
                         "volume_correlations" + "_" + suffix),
            "a+",
    ) as f:
        sys.stdout = f  # Change the standard output to the file we created.

        print(f"{flag} RECONSTRUCTIONS (n = {len(x)})")
        print("{:^15}{:^15}{:^15}{:^15}".format("label", "SAMSEG", "SYNTHSEG",
                                                "p-value"))
        print("=" * 65)
        print("CORRELATIONS")
        print("=" * 65)
        for col_name, name in zip(col_names, config.LABEL_PAIR_NAMES):
            try:
                a = pearsonr(x[col_name], y[col_name])[0]
                b = pearsonr(x[col_name], z[col_name])[0]
                k = pearsonr(y[col_name], z[col_name])[0]
                _, alpha = calculate_pval(b, a, k, len(x[col_name]))
                print(f"{name:^15}{a:^15.3f}{b:^15.3f}{alpha:^15.6f}")
            except ValueError as e:
                a, b, alpha = 0, 0, 0
                print(f"{name:^15}{a:^15.3f}{b:^15.3f}{alpha:^15.6f}")
        print("=" * 65)

        print("MEAN ABSOLUTE RESIDUALS")
        print("=" * 45)
        for col_name, name in zip(col_names, config.LABEL_PAIR_NAMES):
            a = np.mean(np.abs(x[col_name] - y[col_name]) / x[col_name]) * 100
            b = np.mean(np.abs(x[col_name] - z[col_name]) / x[col_name]) * 100

            print(f"{name:^15}{a:^15.3f}{b:^15.3f}")
        print("=" * 45)

        print("MEAN RESIDUALS")
        print("=" * 45)
        for col_name, name in zip(col_names, config.LABEL_PAIR_NAMES):
            a = np.mean((x[col_name] - y[col_name]) / x[col_name]) * 100
            b = np.mean((x[col_name] - z[col_name]) / x[col_name]) * 100

            print(f"{name:^15}{a:^15.3f}{b:^15.3f}")
        print("=" * 45)
        print()
        sys.stdout = original_stdout  # Reset the standard output to its original value

    return


def combine_pairs(df, pair_list):
    for label_pair in pair_list:
        label_pair = tuple(str(item) for item in label_pair)
        df[f"{label_pair}"] = df[label_pair[0]] + df[label_pair[1]]
        df = df.drop(columns=list(label_pair))

    return df


def extract_samseg_volumes(config, folder_path):
    # TODO: worst written function; clean this mess
    df_list = []

    folders = sorted(glob.glob(os.path.join(folder_path, "*")))

    for folder in folders:
        folder_name = os.path.basename(folder)

        try:
            subject_id = re.findall("\d+(?:-|_)\d+",
                                    folder_name)[0].replace("_", "-")
        except IndexError:
            continue

        try:
            df = pd.read_csv(
                os.path.join(folder, "samseg.stats"),
                header=None,
                names=["label", "volume", "units"],
            )
        except FileNotFoundError:
            continue

        # drop column 'units'
        df = df.drop(columns=["units"])

        # remove '# measure' from 'label' column
        df["label"] = df["label"].str.replace(r"# Measure ", "")

        # map 'label' to 'idx'
        df["idx"] = df["label"].map(fs_lut()[1])

        # drop 'label' column
        df = df.drop(columns=["label"])

        # drop 'nan' rows
        df = df[df["idx"].notna()]

        # make 'idx' the new index
        df = df.set_index("idx").sort_index()

        df = df.rename(columns={"volume": subject_id})

        df.index.name = None

        df_list.append(df)

    try:
        df1 = pd.concat(df_list, axis=1)
        df2 = df1.T
    except ValueError:
        return []

    df2 = combine_pairs(df2, config.LABEL_PAIRS)
    hard_samseg_df = df2.drop(
        columns=[column for column in df2.columns if "(" not in column])

    return hard_samseg_df


def print_correlations(config, x, y, file_name=None):
    if file_name is None:
        raise Exception("Please enter a file name to print correlations")
    col_names = x.columns

    corr_dict = dict()
    for col_name in col_names:
        corr_dict[col_name] = pearsonr(x[col_name], y[col_name])[0]

    with open(os.path.join(config.SYNTHSEG_RESULTS, file_name),
              "w",
              encoding="utf-8") as fp:
        json.dump(corr_dict, fp, sort_keys=True, indent=4)


# def write_correlations_to_file0(config, suffix=None):
#     print("Extracting SYNTHSEG Volumes")
#     mri_synthseg_vols = extract_synthseg_vols(config, config.mri_synthseg_vols_file)
#     hard_synthseg_vols = extract_synthseg_vols(config, config.hard_synthseg_vols_file)
#     soft_synthseg_vols = extract_synthseg_vols(config, config.soft_synthseg_vols_file)

#     print("Extracting SAMSEG Volumes")
#     hard_samseg_vols = extract_samseg_volumes(config, config.HARD_SAMSEG_STATS)
#     soft_samseg_vols = extract_samseg_volumes(config, config.SOFT_SAMSEG_STATS)

#     print("Writing Correlations to File")
#     print_correlation_pairs(
#         mri_synthseg_vols,
#         hard_samseg_vols,
#         hard_synthseg_vols,
#         flag="HARD",
#         suffix=suffix,
#     )

#     print_correlation_pairs(
#         mri_synthseg_vols,
#         soft_samseg_vols,
#         soft_synthseg_vols,
#         flag="SOFT",
#         suffix=suffix,
#     )


def get_volumes(config, item):
    source = item["source"]
    if hasattr(config, source):
        source = config.__dict__.get(source)
    else:
        print(f"{source} not found")
        return None

    if item["type"] == "samseg":
        volumes = extract_samseg_volumes(config, source)
    elif item["type"] == "synthseg":
        volumes = extract_synthseg_vols(config, source)

    return volumes


# Note: separating volume extraction, printing and corrlation analysis.
# The goal is to ignore subjects in the last phase rather than the first phase
# (which is the case at the moment)
# DONE: separating printing
# TODO: ignoring subjects in correlation analysis
def write_volumes_to_file(config, item_list):
    file_name = os.path.join(config.SYNTHSEG_RESULTS, "volumes",
                             "volumes.xlsx")
    for item in item_list:
        try:
            volumes = get_volumes(config, item)
        except:
            continue
        
        if volumes is None:
            continue

        mode = "a" if os.path.exists(file_name) else "w"
        with pd.ExcelWriter(file_name, engine="openpyxl", mode=mode) as writer:
            volumes.to_excel(writer, sheet_name=item["tag"])

    return


def write_correlations_to_file(config, item_list, tags, flag=None, suffix=""):

    # TODO: tags must be unique in the list and also in the item list
    if not flag:
        raise Exception()

    # for loop tp preserve order
    filtered_item_list = []
    for tag in tags:
        filtered_item_list.append(
            *[item for item in item_list if item["tag"] == tag])

    vols_list = [get_volumes(config, item) for item in filtered_item_list]

    if len(list(filter(lambda x: x is not None, vols_list))) == 3:
        print_correlation_pairs(
            config,
            *vols_list,
            flag=flag.upper(),
            suffix=suffix,
        )

    return
