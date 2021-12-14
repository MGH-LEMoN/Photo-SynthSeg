#!/usr/bin/env python
# coding: utf-8
"""
Author: HG
Acknowledgements: JEI, BB, and HT
Notes: Contains code to reproduce dice plots from https://arxiv.org/abs/2009.05596
Running Instructions:
1. https://github.com/hvgazula/SynthSeg/tree/photos
2. Create environment using the requirements.txt
3. source activate /space/calico/1/users/Harsha/venvs/synthseg-venv &&
4. export PYTHONPATH='/space/calico/1/users/Harsha/SynthSeg'

TODO:
1. Write a generic function for reading volumes from SAMSEG stats files
2. Write a generic function for reading volumes from SynthSeg volume files
"""
import glob
import json
import math
import os
import re
import sys
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ext.lab2im import utils
from matplotlib import rcParams
from nipype.interfaces.freesurfer import MRIConvert
from scipy.stats import norm
from scipy.stats.stats import pearsonr
from scripts.fs_lut import fs_lut
from SynthSeg.evaluate import fast_dice

rcParams.update({"figure.autolayout": True})

sns.set(
    style="whitegrid",
    rc={
        "text.usetex": True,
        "font.family": "serif",
    },
)

# use this dictionary to gather files from source to destination
file_gather_dict = {
    "mri_scan": {
        "source": "UW_MRI_SCAN",
        "destination": "MRI_SCANS",
        "expr": ["*.rotated.mgz"],
        "message": "Original Scans",
    },
    "image_ref": {
        "source": "UW_MRI_SCAN",
        "destination": "MRI_SCANS_REF",
        "expr": ["*.rotated.masked.mgz"],
        "message": "Image Masks",
    },
    "hard_ref": {
        "source": "UW_MRI_SCAN",
        "destination": "HARD_REF",
        "expr": ["*.rotated.binary.mgz"],
        "message": "Hard References",
    },
    "hard_recon": {
        "source": "UW_HARD_RECON",
        "destination": ["HARD_RECONS3C", "HARD_RECONS"],
        "expr": ["ref_mask", "*recon.mgz"],
        "message": "Hard Reconstructions",
    },
    "soft_recon": {
        "source": "UW_SOFT_RECON",
        "destination": ["SOFT_RECONS3C", "SOFT_RECONS"],
        "expr": ["ref_soft_mask", "*recon.mgz"],
        "message": "Soft Reconstrucions",
    },
    "hard_warped_ref": {
        "source": "UW_HARD_RECON",
        "destination": "HARD_REF_WARPED",
        "expr": [],
        "message": "Hard Warped References",
    },
    "soft_warped_ref": {
        "source": "UW_SOFT_RECON",
        "destination": "SOFT_REF_WARPED",
        "expr": ["ref_soft_mask", "registered_reference.mgz"],
        "message": "Soft Warped References",
    },
    "hard_samseg": {
        "source": "UW_HARD_SAMSEG",
        "destination": "HARD_SAMSEG",
        "expr": ["*seg.mgz"],
        "message": "Hard SAMSEG",
    },
    "soft_samseg": {
        "source": "UW_SOFT_SAMSEG",
        "destination": "SOFT_SAMSEG",
        "expr": ["*seg.mgz"],
        "message": "Soft SAMSEG",
    },
    "hard_gt_labels": {
        "source": "UW_HARD_RECON",
        "destination": "HARD_MANUAL_LABELS_MERGED",
        "expr": ["ref_mask", "*elastix.mgz"],
        "message": "Hard Ground Truth",
    },
    "soft_gt_labels": {
        "source": "UW_SOFT_RECON",
        "destination": "SOFT_MANUAL_LABELS_MERGED",
        "expr": ["ref_soft_mask", "*elastix.mgz"],
        "message": "Soft Ground Truth",
    },
}

samseg_gather_dict = {
    "hard_samseg": {
        "source": "UW_HARD_SAMSEG",
        "destination": "HARD_SAMSEG",
        "expr": ["*seg.mgz"],
        "message": "Hard SAMSEG",
    },
    "soft_samseg": {
        "source": "UW_SOFT_SAMSEG",
        "destination": "SOFT_SAMSEG",
        "expr": ["*seg.mgz"],
        "message": "Soft SAMSEG",
    },
}

mri_convert_items = [
    {
        "source": "MRI_SCANS_SYNTHSEG_REG_RES",
        "reference": "HARD_SAMSEG",
        "target": "MRI_SYNTHSEG_IN_SAMSEG_SPACE",
    },
    {
        "source": "HARD_SYNTHSEG",
        "reference": "HARD_SAMSEG",
        "target": "HARD_SYNTHSEG_IN_SAMSEG_SPACE",
    },
    {
        "source": "HARD_SYNTHSEG",
        "reference": "MRI_SCANS_SYNTHSEG_REG_RES",
        "target": "HARD_SYNTHSEG_IN_MRISEG_SPACE",
    },
    {
        "source": "MRI_SCANS_SYNTHSEG_REG_RES",
        "reference": "SOFT_SAMSEG",
        "target": "MRI_SYNTHSEG_IN_SAMSEG_SPACE",
    },
    {
        "source": "SOFT_SYNTHSEG",
        "reference": "SOFT_SAMSEG",
        "target": "SOFT_SYNTHSEG_IN_SAMSEG_SPACE",
    },
]

dice2d_dict = [
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SYNTHSEG_IN_SAMSEG_SPACE",
        "output_name": "hard_manual_vs_hard_synth_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 0,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space",
    },
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SYNTHSEG_IN_SAMSEG_SPACE",
        "output_name": "hard_manual_vs_hard_synth_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 1,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space",
    },
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SAMSEG",
        "output_name": "hard_manual_vs_hard_sam_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 0,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "HARD_MANUAL_LABELS_MERGED",
        "target": "HARD_SAMSEG",
        "output_name": "hard_manual_vs_hard_sam_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 1,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SYNTHSEG_IN_SAMSEG_SPACE",
        "output_name": "hard_manual_vs_hard_synth_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 0,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SYNTHSEG_IN_SAMSEG_SPACE",
        "output_name": "soft_manual_vs_soft_synth_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 1,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SAMSEG",
        "output_name": "soft_manual_vs_soft_sam_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 0,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
    {
        "source": "SOFT_MANUAL_LABELS_MERGED",
        "target": "SOFT_SAMSEG",
        "output_name": "soft_manual_vs_soft_sam_in_sam_space",
        "slice_bool": 1,
        "merge_bool": 1,
        "message":
        "Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space",
    },
]


def dice_plot_from_df(config, df, out_file_name, flag):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax = sns.boxplot(x="struct",
                     y="score",
                     hue="type",
                     data=df,
                     palette="Greys_r")

    # Setting width and color of outer box
    [i.set_linewidth(1.5) for i in ax.spines.values()]
    [i.set_edgecolor("k") for i in ax.spines.values()]

    # Set y-ticks
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.tick_params(
        axis="y",
        direction="in",
        which="both",
        left="on",
        right="on",
        length=5,
        width=1.25,
    )
    plt.grid(axis="y", which="minor")

    ax.set_ylim(-0.0, 1.0)
    ax.set_xlim(-1, 9)
    [
        ax.axvline(x + 0.5, color="k", linestyle=":", lw=0.5)
        for x in ax.get_xticks()
    ]
    [ax.axvline(x, 0, 0.020, color="k", lw=1) for x in ax.get_xticks()]
    [ax.axvline(x, 0.98, 1, color="k", lw=1) for x in ax.get_xticks()]

    # Adding title
    plt.title(f"2D Dice Scores (For {flag} reconstruction)", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("Dice Overlap", fontsize=20, fontweight="bold")
    # LABEL_PAIR_NAMES = [fr"\textbf{{{item}}}" for item in LABEL_PAIR_NAMES]
    ax.set_xticklabels(config.LABEL_PAIR_NAMES,
                       rotation=45,
                       color="k",
                       fontweight="bold",
                       ha="right")

    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    # Working with Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(handles=handles,
              labels=labels,
              fontsize=20,
              frameon=True,
              edgecolor="black")

    plt.savefig(os.path.join(config.SYNTHSEG_RESULTS, out_file_name))


def construct_dice_plots_from_files(config, file1, file2, merge_flag,
                                    hard_or_soft, out_name):
    data1 = extract_scores(config, file1, merge_flag)
    data2 = extract_scores(config, file2, merge_flag)

    df = create_single_dataframe(config, data1, data2)
    dice_plot_from_df(config, df, out_name, hard_or_soft)


def extract_scores(config, in_file_name, merge=0):
    # TODO: Look into this function again and cleanup
    hard_dice_json = os.path.join(config.SYNTHSEG_RESULTS, in_file_name)
    with open(hard_dice_json, "r") as fp:
        hard_dice = json.load(fp)

    if merge:
        dice_pair_dict = dict()
        for label_idx1, label_idx2 in config.LABEL_PAIRS:
            dice_pair_dict[label_idx1] = []

        for subject in hard_dice:
            for label_idx1, _ in config.LABEL_PAIRS:
                dice_pair = hard_dice[subject].get(str(label_idx1), 0)

                # if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
                dice_pair_dict[label_idx1].append(dice_pair)

        data = []
        for label_idx in dice_pair_dict:
            data.append(dice_pair_dict[label_idx])
    else:
        dice_pair_dict = dict()
        for label_pair in config.LABEL_PAIRS:
            dice_pair_dict[label_pair] = []

        for subject in hard_dice:
            for label_pair in config.LABEL_PAIRS:
                dice_pair = [
                    hard_dice[subject].get(str(label), 0)
                    for label in label_pair
                ]

                # if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
                dice_pair_dict[label_pair].append(dice_pair)

        data = []
        for label_pair in dice_pair_dict:
            data.append(np.mean(dice_pair_dict[label_pair], 1))

    return data


def create_single_dataframe(config, data1, data2):
    ha1 = pd.DataFrame(data1, index=config.LABEL_PAIRS)
    ha2 = pd.DataFrame(data2, index=config.LABEL_PAIRS)

    ha1 = ha1.stack().reset_index()
    ha1 = ha1.rename(
        columns=dict(zip(ha1.columns, ["struct", "subject", "score"])))
    ha1["type"] = "samseg"

    ha2 = ha2.stack().reset_index()
    ha2 = ha2.rename(
        columns=dict(zip(ha2.columns, ["struct", "subject", "score"])))
    ha2["type"] = "synthseg"

    ha = pd.concat([ha1, ha2], axis=0, ignore_index=True)

    return ha


def write_plots(config):
    construct_dice_plots_from_files(
        config,
        "hard_manual_vs_hard_sam_in_sam_space_no-merge.json",
        "hard_manual_vs_hard_synth_in_sam_space_no-merge.json",
        0,
        "hard",
        "hard_reconstruction_no-merge.png",
    )

    construct_dice_plots_from_files(
        config,
        "hard_manual_vs_hard_sam_in_sam_space_merge.json",
        "hard_manual_vs_hard_synth_in_sam_space_merge.json",
        1,
        "hard",
        "hard_reconstruction_merge.png",
    )

    construct_dice_plots_from_files(
        config,
        "soft_manual_vs_soft_sam_in_sam_space_no-merge.json",
        "soft_manual_vs_soft_synth_in_sam_space_no-merge.json",
        0,
        "soft",
        "soft_reconstruction_no-merge.png",
    )

    construct_dice_plots_from_files(
        config,
        "soft_manual_vs_soft_sam_in_sam_space_merge.json",
        "soft_manual_vs_soft_synth_in_sam_space_merge.json",
        1,
        "soft",
        "soft_reconstruction_merge.png",
    )


def merge_labels_in_image(config, x, y):
    merge_required_labels = []
    for (id1, id2) in config.LABEL_PAIRS:
        x[x == id2] = id1
        y[y == id2] = id1

        merge_required_labels.append(id1)

    merge_required_labels = np.array(merge_required_labels)

    return x, y, merge_required_labels


def fisherZ(r):
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def calculate_pval(r12, r13, r23, n):

    z12 = fisherZ(r12)
    z13 = fisherZ(r13)
    z23 = fisherZ(r23)

    r1sq = ((r12 + r13) / 2.0) * ((r12 + r13) / 2.0)
    variance = (1.0 / ((1 - r1sq) *
                       (1 - r1sq))) * (r23 * (1.0 - 2.0 * r1sq) - 0.5 * r1sq *
                                       (1 - 2.0 * r1sq - (r23 * r23)))
    variance2 = np.sqrt((2.0 - 2.0 * variance) / (n - 3.0))

    p = (z12 - z13) / variance2
    alpha = norm.sf(p)

    return p, alpha


def print_correlation_pairs(config, x, y, z, flag=None, suffix=None):
    common_labels = x.index.intersection(y.index).intersection(z.index)
    x = x.loc[common_labels]
    y = y.loc[common_labels]
    z = z.loc[common_labels]

    col_names = x.columns

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(
            os.path.join(config.SYNTHSEG_RESULTS,
                         "volume_correlations" + "_" + suffix),
            "a+",
    ) as f:
        sys.stdout = f  # Change the standard output to the file we created.

        print(f"{flag} RECONSTRUCTIONS")
        print("{:^15}{:^15}{:^15}{:^15}".format("label", "SAMSEG", "SYNTHSEG",
                                                "p-value"))
        print("=" * 65)
        print("CORRELATIONS")
        print("=" * 65)
        for col_name, name in zip(col_names, config.LABEL_PAIR_NAMES):
            a = pearsonr(x[col_name], y[col_name])[0]
            b = pearsonr(x[col_name], z[col_name])[0]
            k = pearsonr(y[col_name], z[col_name])[0]
            _, alpha = calculate_pval(b, a, k, len(x[col_name]))

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


def extract_samseg_volumes(config, folder_path, flag):
    df_list = []

    hard_folder_list = sorted(glob.glob(os.path.join(folder_path, "*")))

    for folder in hard_folder_list:
        _, folder_name = os.path.split(folder)

        if flag == "hard":
            subject_id = folder_name.split(".")[0]
        elif flag == "soft":
            subject_id = folder_name.split("_")[0]
        else:
            raise Exception("Incorrect Flag")

        if subject_id in config.IGNORE_SUBJECTS:
            continue

        df = pd.read_csv(
            os.path.join(folder, "samseg.stats"),
            header=None,
            names=["label", "volume", "units"],
        )

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

    df1 = pd.concat(df_list, axis=1)
    df2 = df1.T

    df2 = combine_pairs(df2, config.LABEL_PAIRS)
    hard_samseg_df = df2.drop(
        columns=[column for column in df2.columns if "(" not in column])

    return hard_samseg_df


def extract_synthseg_vols(config, file_name, flag):
    skiprows = 1 if flag == "mri" else None
    df = pd.read_csv(file_name, skiprows=skiprows, header=0)

    if flag == "mri":
        df = df.rename(columns={"Unnamed: 0": "subjects"})

    df["subjects"] = df["subjects"].str.slice(0, 7)
    df = df.set_index("subjects")

    df.index.name = None

    df = combine_pairs(df, config.LABEL_PAIRS)
    df = df.drop(
        columns=[column for column in df.columns if "(" not in column])
    df = df.drop(labels=config.IGNORE_SUBJECTS)

    return df


def write_correlations_to_file(config, suffix=None):
    print("Extracting SYNTHSEG Volumes")
    mri_synthseg_vols = extract_synthseg_vols(config,
                                              config.mri_synthseg_vols_file,
                                              "mri")
    hard_synthseg_vols = extract_synthseg_vols(config,
                                               config.hard_synthseg_vols_file,
                                               "hard")
    soft_synthseg_vols = extract_synthseg_vols(config,
                                               config.soft_synthseg_vols_file,
                                               "soft")

    print("Extracting SAMSEG Volumes")
    hard_samseg_vols = extract_samseg_volumes(config, config.HARD_SAMSEG_STATS,
                                              "hard")
    soft_samseg_vols = extract_samseg_volumes(config, config.SOFT_SAMSEG_STATS,
                                              "soft")

    print("Writing Correlations to File")
    print_correlation_pairs(
        config,
        mri_synthseg_vols,
        hard_samseg_vols,
        hard_synthseg_vols,
        flag="HARD",
        suffix=suffix,
    )

    print_correlation_pairs(
        config,
        mri_synthseg_vols,
        soft_samseg_vols,
        soft_synthseg_vols,
        flag="SOFT",
        suffix=suffix,
    )


def calculate_dice(config, folder1, folder2, file_name, slice=False, merge=0):
    folder1_list, folder2_list = files_at_path(folder1), files_at_path(folder2)

    folder1_list, folder2_list = return_common_subjects(
        folder1_list, folder2_list)

    final_dice_scores = dict()
    for file1, file2 in zip(folder1_list, folder2_list):
        if not id_check(config, file1, file2):
            continue

        subject_id = os.path.split(file1)[-1][:7]

        x = utils.load_volume(file1)
        y = utils.load_volume(file2)

        assert x.shape == y.shape, "Shape Mismatch"

        if slice:
            slice_idx = np.argmax((x > 1).sum(0).sum(0))

            x = x[:, :, slice_idx].astype("int")
            y = y[:, :, slice_idx].astype("int")

        required_labels = config.required_labels
        if merge:
            x, y, required_labels = merge_labels_in_image(config, x, y)

        dice_coeff = fast_dice(x, y, required_labels)
        required_labels = required_labels.astype("int").tolist()
        final_dice_scores[subject_id] = dict(zip(required_labels, dice_coeff))

    merge_tag = "merge" if merge else "no-merge"

    with open(
            os.path.join(config.SYNTHSEG_RESULTS,
                         f"{file_name}_{merge_tag}.json"),
            "w",
            encoding="utf-8",
    ) as fp:
        json.dump(final_dice_scores, fp, sort_keys=True, indent=4)


def run_mri_convert(in_file, ref_file, out_file):
    mc = MRIConvert()
    mc.terminal_output = "none"
    mc.inputs.in_file = in_file
    mc.inputs.out_file = out_file
    if ref_file:
        mc.inputs.reslice_like = ref_file
    mc.inputs.out_type = "mgz"
    mc.inputs.out_datatype = "float"
    mc.inputs.resample_type = "nearest"

    mc.run()


def perform_overlay(config):
    mri_scans_reg = utils.list_images_in_folder(config.HARD_REF_WARPED)
    mri_resampled_segs = utils.list_images_in_folder(
        config.MRI_SCANS_SYNTHSEG_RESAMPLED)

    mri_scans_reg, mri_resampled_segs = return_common_subjects(
        mri_scans_reg, mri_resampled_segs)

    os.makedirs(config.MRI_SCANS_SYNTHSEG_REG_RES, exist_ok=True)

    print("Creating...")
    for scan_reg, mri_resampled_seg in zip(mri_scans_reg, mri_resampled_segs):
        id_check(config, scan_reg, mri_resampled_seg)

        _, scan_reg_aff, scan_reg_head = utils.load_volume(scan_reg,
                                                           im_only=False)
        mrs_im = utils.load_volume(mri_resampled_seg)

        _, file_name = os.path.split(mri_resampled_seg)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + ".reg" + file_ext
        out_file = os.path.join(config.MRI_SCANS_SYNTHSEG_REG_RES, out_file)

        # We can now combine the segmentation voxels with the registered header.
        utils.save_volume(mrs_im, scan_reg_aff, scan_reg_head, out_file)

        # this new file should overlay with the 3D photo reconstruction


def id_check(config, *args):
    fn_list = set([os.path.split(item)[-1][:7] for item in args])

    assert len(fn_list) == 1, "File MisMatch"

    if config.IGNORE_SUBJECTS:
        if fn_list.intersection(set(config.IGNORE_SUBJECTS)):
            return 0
    return 1


def return_common_subjects(*args):
    if len(set([len(item) for item in args])) > 1:

        args = [{
            os.path.split(input_file)[-1][:7]: input_file
            for input_file in file_list
        } for file_list in args]

        lst = [set(lst.keys()) for lst in args]

        # One-Liner to intersect a list of sets
        common_names = sorted(lst[0].intersection(*lst))

        args = [[lst[key] for key in common_names] for lst in args]

    return args


def perform_registration(config, input_path, reference_path, output_path):
    input_files = files_at_path(input_path)
    reference_files = files_at_path(reference_path)

    input_files, reference_files = return_common_subjects(
        input_files, reference_files)

    os.makedirs(output_path, exist_ok=True)

    print("Creating...")
    for input_file, reference_file in zip(input_files, reference_files):
        id_check(config, input_file, reference_file)

        _, file_name = os.path.split(input_file)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + ".res" + file_ext
        out_file = os.path.join(output_path, out_file)

        run_mri_convert(input_file, reference_file, out_file)


def run_make_target(config, flag):
    os.system(f"make -C {config.SYNTHSEG_PRJCT} predict-{flag}")


def files_at_path(path_str):
    return sorted(glob.glob(os.path.join(path_str, "*")))


def list_files(source, expr):
    if len(expr) == 1:
        file_list = glob.glob(os.path.join(source, *expr))

        if not file_list:
            file_list = glob.glob(os.path.join(source, "*", expr[-1]))

    else:
        file_list = glob.glob(os.path.join(source, "*", *expr[:-1], expr[-1]))

    return file_list


def copy_files_from_source(source, destination, expr):
    """[summary]

    Args:
        source ([type]): [description]
        destination ([type]): [description]
        expr ([type]): [description]

    Raises:
        Exception: [description]
    """
    os.makedirs(destination, exist_ok=True)

    file_list = list_files(source, expr)

    for src_scan_file in file_list:
        _, file_name = os.path.split(src_scan_file)

        if not re.findall("\d+", file_name):
            a = re.findall("\d+", src_scan_file)
            a.remove("1")
            if a:
                file_name = "-".join([*a, file_name])
            else:
                continue

        print(file_name)

        if file_name.startswith("NP"):
            dest_scan_file = file_name[2:].replace("_", "-")
        else:
            dest_scan_file = file_name.replace("_", "-")
        dst_scan_file = os.path.join(destination, dest_scan_file)

        copyfile(src_scan_file, dst_scan_file)

    return


def copy_relevant_files(config, some_dict):

    for key in some_dict.keys():
        if some_dict[key]["expr"]:
            print(f'Copying {some_dict[key]["message"]}')
            src = config.__getattribute__(some_dict[key]["source"])

            destinations = some_dict[key]["destination"]
            if isinstance(destinations, list):
                for destination in destinations:
                    dest = config.__getattribute__(destination)
                    copy_files_from_source(src, dest, some_dict[key]["expr"])
            else:
                dest = config.__getattribute__(destinations)
                copy_files_from_source(src, dest, some_dict[key]["expr"])


def calculate_dice_for_dict(config, item_list):
    for item in item_list:
        source = config.__dict__.get(item["source"], None)
        target = config.__dict__.get(item["target"], None)

        output_file = item["output_file"]
        slice_bool = item["slice_bool"]
        merge_bool = item["merge_bool"]

        if not source:
            raise Exception(f'Source folder {item["source"]} does not exist')

        if not target:
            raise Exception(
                f'Reference folder {item["reference"]} does not exist')

        if not output_file:
            raise Exception("Output File Name Cannot be Empty")

        calculate_dice(config,
                       source,
                       target,
                       output_file,
                       slice=slice_bool,
                       merge=merge_bool)


def move_volumes_into_target_spaces(config, item_list):
    for item in item_list:
        source = config.__dict__.get(item["source"], None)
        reference = config.__dict__.get(item["reference"], None)
        target = config.__dict__.get(item["target"], None)

        if not source:
            raise Exception(f'Source folder {item["source"]} does not exist')

        if not reference:
            raise Exception(
                f'Reference folder {item["reference"]} does not exist')

        if not target:
            print(f"Target folder DNE: But adding now")
            target_folder_name = item["target"].lower().replace("_", ".")
            setattr(
                config,
                item["target"],
                f"{config.SYNTHSEG_RESULTS}/{config.PROJECT_ID}.{target_folder_name}",
            )
            target = config.__dict__.get(item["target"], None)

        print(f'Putting {" ".join(item["target"].split("_"))}')
        perform_registration(config, source, reference, target)


class Configuration:
    def __init__(self, project_dir, project_suffix, out_folder):
        self.SYNTHSEG_PRJCT = project_dir
        self.PROJECT_ID = project_suffix
        self.SYNTHSEG_RESULTS = f"{self.SYNTHSEG_PRJCT}/recon-results/{out_folder}"

        self.UW_HARD_RECON = "/cluster/vive/UW_photo_recon/Photo_data/"
        self.UW_SOFT_RECON = "/cluster/vive/UW_photo_recon/Photo_data/"
        self.UW_MRI_SCAN = "/cluster/vive/UW_photo_recon/FLAIR_Scan_Data"

        self.MRI_SCANS = f"{self.SYNTHSEG_RESULTS}/UW.mri.scans"
        self.MRI_SCANS_REF = f"{self.SYNTHSEG_RESULTS}/UW.mri.scans.ref"
        self.MRI_SCANS_SYNTHSEG = f"{self.SYNTHSEG_RESULTS}/UW.mri.synthseg"

        self.UW_SOFT_SAMSEG = f"{self.SYNTHSEG_RESULTS}/UW.soft.samseg"
        self.UW_HARD_SAMSEG = f"{self.SYNTHSEG_RESULTS}/UW.hard.samseg"

        self.MRI_SCANS_SYNTHSEG_RESAMPLED = self.MRI_SCANS_SYNTHSEG + ".resampled"
        self.MRI_SCANS_SYNTHSEG_REG_RES = (self.MRI_SCANS_SYNTHSEG_RESAMPLED +
                                           ".registered")

        self.HARD_REF = f"{self.SYNTHSEG_RESULTS}/UW.hard.ref"
        self.HARD_REF_WARPED = f"{self.SYNTHSEG_RESULTS}/UW.hard.ref.warped"
        self.HARD_RECONS3C = f"{self.SYNTHSEG_RESULTS}/UW.hard.recon3c"
        self.HARD_RECONS = f"{self.SYNTHSEG_RESULTS}/UW.hard.recon"
        self.HARD_RECON_SYNTHSEG = f"{self.SYNTHSEG_RESULTS}/UW.hard.synthseg"
        self.HARD_SAMSEG = f"{self.SYNTHSEG_RESULTS}/UW.hard.samseg.segmentations"
        self.HARD_MANUAL_LABELS_MERGED = (
            f"{self.SYNTHSEG_RESULTS}/UW.hard.manual.labels")

        self.SOFT_RECONS3C = f"{self.SYNTHSEG_RESULTS}/UW.soft.recon3c"
        self.SOFT_RECONS = f"{self.SYNTHSEG_RESULTS}/UW.soft.recon"
        self.SOFT_REF_WARPED = f"{self.SYNTHSEG_RESULTS}/UW.soft.ref.warped"
        self.SOFT_RECON_SYNTHSEG = f"{self.SYNTHSEG_RESULTS}/UW.soft.synthseg"
        self.SOFT_SAMSEG = f"{self.SYNTHSEG_RESULTS}/UW.soft.samseg.segmentations"
        self.SOFT_MANUAL_LABELS_MERGED = (
            f"{self.SYNTHSEG_RESULTS}/UW.soft.manual.labels")

        # Note: All of these are in photo RAS space (just resampling based on reference)
        self.MRI_SYNTHSEG_IN_SAMSEG_SPACE = (self.MRI_SCANS_SYNTHSEG_REG_RES +
                                             ".in_samseg_space")
        self.MRI_SYNTHSEG_IN_SOFTSAMSEG_SPACE = (
            self.MRI_SCANS_SYNTHSEG_REG_RES + ".in_softsamseg_space")
        self.HARD_SYNTHSEG_IN_SAMSEG_SPACE = (self.HARD_RECON_SYNTHSEG +
                                              ".in_samseg_space")
        self.HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE = (self.HARD_RECON_SYNTHSEG +
                                                    ".in_mri_space")
        self.SOFT_SYNTHSEG_IN_SAMSEG_SPACE = (self.SOFT_RECON_SYNTHSEG +
                                              ".in_samseg_space")

        self.mri_synthseg_vols_file = (
            f"{self.SYNTHSEG_RESULTS}/UW.mri.synthseg.volumes.csv")
        self.soft_synthseg_vols_file = (
            f"{self.SYNTHSEG_RESULTS}/UW.soft.synthseg.volumes.csv")
        self.hard_synthseg_vols_file = (
            f"{self.SYNTHSEG_RESULTS}/UW.hard.synthseg.volumes.csv")

        #### Extract SAMSEG Volumes
        self.HARD_SAMSEG_STATS = f"{self.UW_HARD_SAMSEG}"
        self.SOFT_SAMSEG_STATS = f"{self.UW_SOFT_SAMSEG}"

        self.ALL_LABELS = [
            0,
            2,
            3,
            4,
            5,
            10,
            11,
            12,
            13,
            14,
            17,
            18,
            26,
            28,
            41,
            42,
            43,
            44,
            49,
            50,
            51,
            52,
            53,
            54,
            58,
            60,
        ]
        self.IGNORE_LABELS = [0, 5, 14, 26, 28, 44, 58, 60]
        self.ADDL_IGNORE_LABELS = [7, 8, 15, 16, 46, 47]
        self.LABEL_PAIRS = [
            (2, 41),
            (3, 42),
            (4, 43),
            (10, 49),
            (11, 50),
            (12, 51),
            (13, 52),
            (17, 53),
            (18, 54),
        ]
        self.LABEL_PAIR_NAMES = [
            "White Matter",
            "Cortex",
            "Ventricle",
            "Thalamus",
            "Caudate",
            "Putamen",
            "Pallidum",
            "Hippocampus",
            "Amygdala",
        ]
        self.IGNORE_SUBJECTS = ["18-1343", "18-2260", "19-0019"]

        self._write_config()

        self.required_labels = np.array(
            list(set(self.ALL_LABELS) - set(self.IGNORE_LABELS)))

    def _write_config(self, file_name=None):
        """Write configuration to a file
        Args:
            CONFIG (dict): configuration
        """
        file_name = "config.json" if file_name is None else file_name

        dictionary = self.__dict__
        json_object = json.dumps(dictionary, sort_keys=True, indent=4)

        utils.mkdir(dictionary["SYNTHSEG_RESULTS"])

        config_file = os.path.join(dictionary["SYNTHSEG_RESULTS"], file_name)

        with open(config_file, "w") as outfile:
            outfile.write(json_object)


def convert_to_single_channel(config, folder_str):

    file_list = utils.list_images_in_folder(
        config.__getattribute__(folder_str))
    for file in file_list:
        im, aff, hdr = utils.load_volume(file, im_only=False)

        if im.ndim == 4 and im.shape[-1] == 3:
            im = np.mean(im, axis=-1)

        utils.save_volume(im, aff, hdr, file)
    pass


if __name__ == "__main__":
    # !!! START HERE !!!
    project_dir = "/space/calico/1/users/Harsha/SynthSeg"
    project_prefix = "UW"
    results_folder = "new"

    config = Configuration(project_dir, project_prefix, results_folder)
    copy_relevant_files(config, file_gather_dict)

    # # It looks like SAMSEG doesn't need 3Channel images.
    # # TODO: Do away with the following 2 lines of code
    # convert_to_single_channel(config, "HARD_RECONS")
    convert_to_single_channel(config, "SOFT_RECONS")

    # print("Running SynthSeg...")
    # # Due to some code incompatibility issues, the following lines of code
    # # have to be run separately on MLSC or this entire script can be run on MLSC
    # run_make_target(config, "hard")
    # run_make_target(config, "soft")
    # run_make_target(config, "scans")

    copy_relevant_files(config, samseg_gather_dict)
    # write_correlations_to_file(config)
    move_volumes_into_target_spaces(config, mri_convert_items)
    calculate_dice_for_dict(config, dice2d_dict)
    write_plots(config)
