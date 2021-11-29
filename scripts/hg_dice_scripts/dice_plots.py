import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dice_config import *
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

sns.set(style="whitegrid", rc={
    'text.usetex': True,
    'font.family': 'serif',
})

# # TODO: this file is work in progress
# plt.rcParams.update({"text.usetex": True,
# 'font.family': 'serif'})


def dice_plot_from_df(df, out_file_name, flag):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax = sns.boxplot(x="struct",
                     y="score",
                     hue="type",
                     data=df,
                     palette="Greys_r")

    # Setting width and color of outer box
    [i.set_linewidth(1.5) for i in ax.spines.values()]
    [i.set_edgecolor('k') for i in ax.spines.values()]

    # Set y-ticks
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.tick_params(axis="y",
                   direction="in",
                   which='both',
                   left='on',
                   right='on',
                   length=5,
                   width=1.25)
    plt.grid(axis='y', which='minor')

    ax.set_ylim(-0.0, 1.0)
    ax.set_xlim(-1, 9)
    [
        ax.axvline(x + .5, color='k', linestyle=':', lw=0.5)
        for x in ax.get_xticks()
    ]
    [ax.axvline(x, 0, 0.020, color='k', lw=1) for x in ax.get_xticks()]
    [ax.axvline(x, 0.98, 1, color='k', lw=1) for x in ax.get_xticks()]

    # Adding title
    plt.title(f"2D Dice Scores (For {flag} reconstruction)", fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('Dice Overlap', fontsize=20, fontweight='bold')
    # LABEL_PAIR_NAMES = [fr"\textbf{{{item}}}" for item in LABEL_PAIR_NAMES]
    ax.set_xticklabels(LABEL_PAIR_NAMES,
                       rotation=45,
                       color='k',
                       fontweight='bold',
                       ha='right')

    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    # Working with Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(handles=handles,
              labels=labels,
              fontsize=20,
              frameon=True,
              edgecolor='black')

    plt.savefig(os.path.join(SYNTHSEG_RESULTS, out_file_name))


def construct_dice_plots_from_files(file1, file2, merge_flag, hard_or_soft,
                                    out_name):
    data1 = extract_scores(file1, merge_flag)
    data2 = extract_scores(file2, merge_flag)

    df = create_single_dataframe(data1, data2)
    dice_plot_from_df(df, out_name, hard_or_soft)


def extract_scores(in_file_name, merge=0):
    # Load json
    hard_dice_json = os.path.join(SYNTHSEG_RESULTS, in_file_name)
    with open(hard_dice_json, 'r') as fp:
        hard_dice = json.load(fp)

    if merge:
        dice_pair_dict = dict()
        for label_idx1, label_idx2 in LABEL_PAIRS:
            dice_pair_dict[label_idx1] = []

        for subject in hard_dice:
            for label_idx1, _ in LABEL_PAIRS:
                dice_pair = hard_dice[subject].get(str(label_idx1), 0)

                # if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
                dice_pair_dict[label_idx1].append(dice_pair)

        data = []
        for label_idx in dice_pair_dict:
            data.append(dice_pair_dict[label_idx])
    else:
        dice_pair_dict = dict()
        for label_pair in LABEL_PAIRS:
            dice_pair_dict[label_pair] = []

        for subject in hard_dice:
            for label_pair in LABEL_PAIRS:
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


def create_single_dataframe(data1, data2):
    ha1 = pd.DataFrame(data1, index=LABEL_PAIRS)
    ha2 = pd.DataFrame(data2, index=LABEL_PAIRS)

    ha1 = ha1.stack().reset_index()
    ha1 = ha1.rename(
        columns=dict(zip(ha1.columns, ['struct', 'subject', 'score'])))
    ha1['type'] = 'samseg'

    ha2 = ha2.stack().reset_index()
    ha2 = ha2.rename(
        columns=dict(zip(ha2.columns, ['struct', 'subject', 'score'])))
    ha2['type'] = 'synthseg'

    ha = pd.concat([ha1, ha2], axis=0, ignore_index=True)

    return ha


def write_plots():

    construct_dice_plots_from_files(
        'hard_manual_vs_hard_sam_in_sam_space_no-merge.json',
        'hard_manual_vs_hard_synth_in_sam_space_no-merge.json', 0, 'hard',
        'hard_reconstruction_no-merge.png')

    construct_dice_plots_from_files(
        'hard_manual_vs_hard_sam_in_sam_space_merge.json',
        'hard_manual_vs_hard_synth_in_sam_space_merge.json', 1, 'hard',
        'hard_reconstruction_merge.png')

    construct_dice_plots_from_files(
        'soft_manual_vs_soft_sam_in_sam_space_no-merge.json',
        'soft_manual_vs_soft_synth_in_sam_space_no-merge.json', 0, 'soft',
        'soft_reconstruction_no-merge.png')

    construct_dice_plots_from_files(
        'soft_manual_vs_soft_sam_in_sam_space_merge.json',
        'soft_manual_vs_soft_synth_in_sam_space_merge.json', 1, 'soft',
        'soft_reconstruction_merge.png')


if __name__ == '__main__':
    write_plots()
