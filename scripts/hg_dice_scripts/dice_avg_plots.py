import glob
import json
import os
import re
from pathlib import Path
from tracemalloc import start

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dice_config import *
from dice_plots import create_single_dataframe
from matplotlib import rcParams
from PIL import Image
from uw_config import PLOTS_LIST

rcParams.update({"figure.autolayout": True})

sns.set(
    style="whitegrid",
    rc={
        "text.usetex": True,
        "font.family": "serif",
    },
)

RESULTS_DIR = "/space/calico/1/users/Harsha/SynthSeg/results/20220411/new-recons-skip-4"
SYNTHSEG_RESULTS = RESULTS_DIR

LABEL_PAIRS = [
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
LABEL_PAIR_NAMES = [
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


def dice_plot_from_df(df, out_file_name, something="avg_images"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax = sns.boxplot(x="struct", y="score", hue="type", data=df, palette="Greys_r")

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
    [ax.axvline(x + 0.5, color="k", linestyle=":", lw=0.5) for x in ax.get_xticks()]
    [ax.axvline(x, 0, 0.020, color="k", lw=1) for x in ax.get_xticks()]
    [ax.axvline(x, 0.98, 1, color="k", lw=1) for x in ax.get_xticks()]

    # HACK: To print a meaningful title (for models with SxxRxx namming)
    title_string, _ = os.path.splitext(os.path.basename(out_file_name))
    model_name, recon_type, _, sam_type, _ = title_string.split("_")

    os.makedirs(os.path.join(SYNTHSEG_RESULTS, something), exist_ok=True)
    old_file = os.path.join(SYNTHSEG_RESULTS, something, out_file_name)
    if os.path.exists(old_file):
        os.remove(old_file)

    new_title = f"Model: {model_name}, Recon: {recon_type.capitalize()}, SAMSEG Type: {sam_type.upper()}"

    # Adding title
    # plt.title(f"2D Dice Scores (For {flag} reconstruction)", fontsize=20)
    new_title = plt.title(new_title, fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("Dice Overlap", fontsize=20, fontweight="bold")
    # LABEL_PAIR_NAMES = [fr"\textbf{{{item}}}" for item in LABEL_PAIR_NAMES]
    ax.set_xticklabels(
        LABEL_PAIR_NAMES, rotation=45, color="k", fontweight="bold", ha="right"
    )

    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    # Working with Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(
        handles=handles, labels=labels, fontsize=10, frameon=True, edgecolor="black"
    )

    plt.savefig(os.path.join(SYNTHSEG_RESULTS, something, out_file_name))
    plt.close()


def extract_scores(hard_dice_json, merge=0):

    if not os.path.isfile(hard_dice_json):
        print(f"File DNE: {hard_dice_json}")
        return None

    with open(hard_dice_json, "r") as fp:
        hard_dice = json.load(fp)

    dice_pair_dict = dict()
    if merge:
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
        for label_pair in LABEL_PAIRS:
            dice_pair_dict[label_pair] = []

        for subject in hard_dice:
            for label_pair in LABEL_PAIRS:
                dice_pair = [
                    hard_dice[subject].get(str(label), 0) for label in label_pair
                ]

                # if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
                dice_pair_dict[label_pair].append(dice_pair)

        data = []
        for label_pair in dice_pair_dict:
            data.append(np.mean(dice_pair_dict[label_pair], 1))

    return data


def create_single_dataframe(data1, flag1=None, flag2=None):
    ha1 = pd.DataFrame(data1, index=LABEL_PAIRS)

    ha1 = ha1.stack().reset_index()
    ha1 = ha1.rename(columns=dict(zip(ha1.columns, ["struct", "subject", "score"])))
    if flag1:
        ha1["type"] = flag1
    if flag2:
        ha1["model_key"] = flag2

    return ha1


def create_single_dataframe1(data1, flag1=None, flag2=None):
    ha1 = pd.DataFrame(data1, index=LABEL_PAIRS)

    ha1 = ha1.stack().reset_index()
    ha1 = ha1.rename(columns=dict(zip(ha1.columns, ["struct", "subject", "score"])))

    if flag1:
        ha1["type"] = flag1

    ha1 = ha1.set_index(ha1.columns.difference(["score"], sort=False).tolist())

    return ha1


def construct_average_dice_plots_from_files():

    all_folders = sorted(os.listdir(RESULTS_DIR))
    for search_str in [
        "^S[0-9].+[0-9]n$",
        "^S[0-9].+noflip$",
        "^VS0[0-9]n$",
        "^VS0[0-9]-noflip$",
        "^VS0[0-9]-accordion-noflip$",
        "^VS0[0-9]n-accordion$",
    ]:
        results_folders = [
            f for f in all_folders if re.search(search_str, os.path.basename(f))
        ]
        if results_folders[0].startswith("S"):
            if "-" in results_folders[0]:
                start_str = results_folders[0][:3] + results_folders[0][6:]
            else:
                start_str = results_folders[0][:3]
        elif results_folders[0].startswith("VS"):
            start_str = "VS" + results_folders[0][4:]
        else:
            print("Something's Off")

        if len(results_folders) == 0:
            continue

        for plot_item_idx, item in enumerate(PLOTS_LIST):
            if plot_item_idx in [0, 4]:
                continue
            file1, file2, merge_flag, _, out_name = item

            dice_flag = ["samseg", "synthseg"]
            plot_df = []
            for flag_idx, dice_file in enumerate([file1, file2]):

                dice_data_df_list = []
                for folder in results_folders:
                    folder = os.path.join(RESULTS_DIR, folder)
                    data = extract_scores(
                        os.path.join(folder, "dice_files", dice_file), merge_flag
                    )
                    data_df = create_single_dataframe(data, dice_flag[flag_idx])

                    dice_data_df_list.append(data_df)

                plot_df.append(pd.concat(dice_data_df_list, axis=0, ignore_index=True))

            plot_df = pd.concat(plot_df, axis=0, ignore_index=True)

            out_name = f"{start_str}_{out_name}"
            dice_plot_from_df(plot_df, out_name)


def collect_images_into_pdf(target_dir_str):
    """[summary]

    Args:
        target_dir_str ([str]): string relative to RESULTS_DIR
    """
    target_dir = os.path.join(RESULTS_DIR, target_dir_str)
    out_file = "_".join(target_dir.split("/")[-2:])
    out_file = out_file + ".pdf"
    out_file = os.path.join(RESULTS_DIR, out_file)

    images = sorted(glob.glob(os.path.join(target_dir, "*")))

    pdf_img_list = []
    for image in images:
        img = Image.open(image)
        img = img.convert("RGB")
        pdf_img_list.append(img)

    pdf_img_list[0].save(out_file, save_all=True, append_images=pdf_img_list[1:])


# def construct_comparison_dice_plots_from_files(key_pair):
#     # for each plot type
#     for plot_item_idx, item in enumerate(PLOTS_LIST):
#         # skipping couple of redundant plots
#         if plot_item_idx in [0, 4]:
#             continue
#         file1, file2, merge_flag, _, out_name = item

#         # list all models in that folder
#         all_folders = sorted(os.listdir(RESULTS_DIR))

#         model_dice_dict = dict()
#         # dice_data_df_list = []
#         # use search string to extract the right model (to average)
#         start_name_list = []
#         for search_str in key_pair:
#             results_folders = [
#                 f for f in all_folders
#                 if re.search(search_str, os.path.basename(f))
#             ]

#             # specify the outputname
#             if results_folders[0].startswith('S'):
#                 if '-' in results_folders[0]:
#                     start_str = results_folders[0][:3] + results_folders[0][6:]
#                 else:
#                     start_str = results_folders[0][:3]
#             elif results_folders[0].startswith('VS'):
#                 start_str = 'VS' + results_folders[0][4:]
#             else:
#                 print("Something's Off")

#             start_name_list.append(start_str)

#             # move on if no models exist
#             if len(results_folders) == 0:
#                 continue

#             dice_flag = ['samseg', 'synthseg']
#             plot_df = []
#             for flag_idx, dice_file in enumerate([file1, file2]):

#                 if dice_flag[flag_idx] in model_dice_dict:
#                     continue

#                 dice_data_df_list = []
#                 for folder in results_folders:
#                     folder = os.path.join(RESULTS_DIR, folder)
#                     data = extract_scores(
#                         os.path.join(folder, 'dice_files', dice_file),
#                         merge_flag)

#                     if flag_idx == 0:
#                         val = dice_flag[flag_idx]
#                     else:
#                         val = start_str + '-' + dice_flag[flag_idx]

#                     data_df = create_single_dataframe1(data, val)

#                     dice_data_df_list.append(data_df)

#                 chacha = pd.concat(dice_data_df_list, axis=1).mean(
#                     axis=1).reset_index().rename(columns={0: 'score'})

#                 if val not in model_dice_dict:
#                     model_dice_dict[val] = chacha

#         out_df = pd.concat(model_dice_dict.values(), axis=0, ignore_index=True)

#         prefix_str = '-vs-'.join(start_name_list)
#         out_name = f'{prefix_str}_{out_name}'

#         # now that we got the big df, let's start plotting
#         dice_plot_from_df(out_df, out_name, "comp_images")

#     return


def match_model_dirs(all_folders, search_str):
    """Filter all folders to match the search string

    Args:
        all_folders (sequence): List of folders
        search_str (string): A predefined regex string to filter models

    Returns:
        List: folders matching the regular expression
    """
    return [f for f in all_folders if re.search(search_str, os.path.basename(f))]


def model_output_string(results_folders):
    if results_folders[0].startswith("S"):
        if "-" in results_folders[0]:
            start_str = results_folders[0][:3] + results_folders[0][6:]
        else:
            start_str = results_folders[0][:3]
    elif results_folders[0].startswith("VS"):
        start_str = "VS" + results_folders[0][4:]
    else:
        print("Something's Off")

    return start_str


def make_model_dice_df(results_folders, dice_file, key, merge_flag):
    dice_data_df_list = []
    for folder in results_folders:
        folder = os.path.join(RESULTS_DIR, folder)
        data = extract_scores(os.path.join(folder, "dice_files", dice_file), merge_flag)

        data_df = create_single_dataframe1(data, key)

        dice_data_df_list.append(data_df)

    return dice_data_df_list


def avg_dice_across_models(df_list):
    return (
        pd.concat(df_list, axis=1)
        .replace(0, np.NaN)
        .mean(axis=1)
        .reset_index()
        .rename(columns={0: "score"})
    )


def make_dict_key(flag_idx, start_str):
    dice_flag = ["samseg", "synthseg"]
    if flag_idx == 0:
        key = dice_flag[flag_idx]
    else:
        key = start_str + "-" + dice_flag[flag_idx]
    return key


def all_in_one_comparison(key_pair, out_folder):
    """Plot all models in one figure

    Args:
        key_pair (sequence): A list/tuple of regexp's to match the model
    """
    # list all models in that folder
    all_folders = sorted(os.listdir(RESULTS_DIR))

    # for each plot type desired
    for plot_item_idx, item in enumerate(PLOTS_LIST):

        # skipping couple of redundant plots
        # HACK: This is a hack to ignore the first and fifth plots
        if plot_item_idx in [0, 4]:
            continue

        file1, file2, merge_flag, _, out_name = item

        # This dictionary will be used to store the dice scores of each model
        model_dice_dict = dict()

        # use search string to extract the right model (to average)
        start_name_list = []

        for search_str in key_pair:
            matching_models = match_model_dirs(all_folders, search_str)
            if not matching_models:
                continue

            # specify the output name
            start_str = model_output_string(matching_models)
            start_name_list.append(start_str)

            for flag_idx, dice_file in enumerate([file1, file2]):

                key = make_dict_key(flag_idx, start_str)
                if key in model_dice_dict:
                    continue

                dice_data_df_list = make_model_dice_df(
                    matching_models, dice_file, key, merge_flag
                )
                model_dice_dict[key] = avg_dice_across_models(dice_data_df_list)

        out_df = pd.concat(model_dice_dict.values(), axis=0, ignore_index=True)

        prefix_str = "-vs-".join(start_name_list) if len(key_pair) < 2 else "All"
        out_name = f"{prefix_str}_{out_name}"

        # now that we got the big df, let's make a boxplot
        dice_plot_from_df(out_df, out_name, out_folder)

    return


if __name__ == "__main__":
    regex_list = [
        "^S[0-9].+[0-9]n$",
        "^S[0-9].+noflip$",
        "^VS0[0-9]n$",
        "^VS0[0-9]-noflip$",
        "^VS0[0-9]n-accordion$",
        "^VS0[0-9]-accordion-noflip$",
    ]

    key_pairs = [
        [regex_list[0], regex_list[2]],
        [regex_list[1], regex_list[3]],
        [regex_list[0], regex_list[4]],
        [regex_list[1], regex_list[5]],
    ]

    # The following 2 lines are used to plot pairs of models in one file
    # for key_pair in key_pairs:
    #     construct_comparison_dice_plots_from_files(key_pair, "comp_images")
    # collect_images_into_pdf('comp_images')

    # The following 2 lines are used to plot "all" models in one file
    all_in_one_comparison(regex_list, "all_images")
    collect_images_into_pdf("all_images")

    # NOTE: I guess the following function needs a cleanup
    # The following 2 lines are used to plot "all" models in one file
    # construct_average_dice_plots_from_files(key_pair)
    # collect_images_into_pdf('avg_images')
