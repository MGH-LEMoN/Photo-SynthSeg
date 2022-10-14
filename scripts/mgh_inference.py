import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ext.lab2im import utils
from matplotlib.backends.backend_pdf import PdfPages

sns.set_theme(style="ticks")
plt.rcParams.update({"text.usetex": True})

labels_mapping = {
    2: "left cerebral white matter",
    3: "left cerebral cortex",
    4: "left lateral ventricle",
    5: "left inferior lateral ventricle",
    7: "left cerebellum white matter",
    8: "left cerebellum cortex",
    10: "left thalamus",
    11: "left caudate",
    12: "left putamen",
    13: "left pallidum",
    14: "3rd ventricle",
    15: "4th ventricle",
    16: "brain-stem",
    17: "left hippocampus",
    18: "left amygdala",
    26: "left accumbens area",
    28: "left ventral DC",
}

IGNORE_SUBJECTS = [
    "2604_whole",
    "2705_left",
    "2706_whole",
    "2707_whole",
    "2708_left",
    "2710_whole",
    "2711_left",
    "2712_left",
    "2687_left",
    "2689_left",
    "2691_right",
    "2715_right",
    "2723_left",
    "2724_whole",
    "2729_whole",
]

lh_to_rh_mapping = {
    2: 41,
    3: 42,
    4: 43,
    5: 44,
    10: 49,
    11: 50,
    12: 51,
    13: 52,
    17: 53,
    18: 54,
    26: 58,
    28: 60,
}

IGNORE_SUBJECTS = [2708, 2719, 2722, 2630, 2629, 2668, 2605, 2711]


def combine_pairs(df, mapping):
    """Combine pairs of volumes according to a given mapping."""
    for label_pair in mapping.items():
        try:
            df[label_pair[0]] = (df[label_pair[0]] + df[label_pair[1]]) / 1.0
            df = df.drop(columns=[label_pair[1]])
        except KeyError:
            continue
    return df


def relabel_volume(volume, mapping):
    """Relabel a volume according to a given mapping.

    Args:
        volume (np.array): input volume.
        mapping (dict): mapping between labels.

    Returns:
        np.array: relabeled volume.
    """
    for i in mapping:
        volume[volume == i] = mapping[i]
    return volume


def convert_to_single_channel(file):
    """Convert RGB volume to Grayscale volume

    Args:
        config ([type]): [description]
        folder_str ([type]): [description]
    """
    im, aff, hdr = utils.load_volume(file, im_only=False)

    if im.ndim == 4 and im.shape[-1] == 3:
        im = np.mean(im, axis=-1)

    utils.save_volume(im, aff, hdr, file)


def print_disp_commands(synth_dir, side=None):
    # print command for freeview
    if side == "right":
        for file in sorted(glob.glob(os.path.join(synth_dir, "*rh2lh*.mgz"))):
            subject_id = os.path.basename(file).split("_")[0]
            command = f"freeview /cluster/vive/MGH_photo_recon/{subject_id}_right/recon/photo_recon.mgz:rgb=1 \
                                {file.replace('rh2lh', 'lh2rh')}:colormap=lut:opacity=0.5:visible=1 \
                                {file.replace('_synthseg', '').replace('synthseg', 'recon')}:rgb=1 \
                                {file}:colormap=lut:opacity=0.5:visible=1"
            command = " ".join(command.split())
            print(command, "\n")
    elif side == "left" or side == "whole":
        for file in sorted(glob.glob(os.path.join(synth_dir, "*.mgz"))):
            subject_id = os.path.basename(file).split("_")[0]
            command = f"freeview /cluster/vive/MGH_photo_recon/{subject_id}_{side}/recon/photo_recon.mgz:rgb=1 \
                                {file}:colormap=lut:opacity=0.5:visible=1"
            command = " ".join(command.split())
            print(command, "\n")
    else:
        pass

    return


def flip_hemi(synth_dir):
    for file in sorted(glob.glob(os.path.join(synth_dir, "*rh2lh*.mgz"))):
        im, aff, hdr = utils.load_volume(file, im_only=False)
        im_flip = np.flip(im, 1)
        im_flip = relabel_volume(im_flip, lh_to_rh_mapping)
        utils.save_volume(im_flip, aff, hdr, file.replace("rh2lh", "lh2rh"))


def perform_segmentation():
    SIDE = "whole"
    # "left" | "right" | "whole"
    PRJCT_DIR = "/cluster/vive/MGH_photo_recon"
    subjects = sorted(glob.glob(os.path.join(PRJCT_DIR, f"*{SIDE}*")))
    subjects_ids = [os.path.basename(s) for s in subjects]

    # print(subjects_ids)

    MODEL = "VS01n-accordion"
    DICE_IDX = 100
    H5_FILE = f"/space/calico/1/users/Harsha/SynthSeg/models/models-2022/{MODEL}/dice_{DICE_IDX:03d}.h5"
    LABEL_LIST = "/space/calico/1/users/Harsha/SynthSeg/models/jei-model/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.label_list.npy"

    MISC_DIR = os.path.join(
        "/space/calico/1/users/Harsha/SynthSeg",
        "results",
        "mgh_inference_20221011",
    )
    RECON_DIR = os.path.join(
        MISC_DIR, f"mgh.surf.recon.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d}"
    )
    SYNTH_DIR = os.path.join(
        MISC_DIR,
        f"mgh.surf.synthseg.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d}",
    )

    os.makedirs(RECON_DIR, exist_ok=True)
    os.makedirs(SYNTH_DIR, exist_ok=True)

    for subject in subjects:
        if os.path.basename(subject) in IGNORE_SUBJECTS:
            print(f"{os.path.basename(subject)} - Ignored")
            continue

        # Cases: yet to be processed
        if not os.path.exists(os.path.join(subject, "recon")):
            print(f"{os.path.basename(subject)} - TBD")
            continue

        src = os.path.join(subject, "recon", "photo_recon.mgz")

        # Cases: already processed but missing recon
        if not os.path.isfile(src):
            print(f"{os.path.basename(subject)} - Missing recon")
            continue

        print(os.path.basename(subject))
        if SIDE == "right":
            # Load Label Map
            im, aff, hdr = utils.load_volume(src, im_only=False)
            im_flip = np.flip(im, 1)
            dst = os.path.join(
                RECON_DIR,
                f"{os.path.basename(subject)}.mgz".replace("right", "rh2lh"),
            )
            utils.save_volume(im_flip, aff, hdr, dst)
        else:
            dst = os.path.join(RECON_DIR, f"{os.path.basename(subject)}.mgz")
            shutil.copyfile(src, dst)

        convert_to_single_channel(dst)

    command = f"python {os.getcwd()}/scripts/commands/predict.py \
    --smoothing 0.5 \
    --biggest_component \
    --padding 256 \
    --vol {MISC_DIR}/mgh.surf.volumes.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d} \
    --neutral_labels 5 \
    --post {MISC_DIR}/mgh.surf.posterior.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d} \
    --flip \
    {RECON_DIR} \
    {SYNTH_DIR} \
    {H5_FILE} \
    {LABEL_LIST}"

    command = " ".join(command.split())
    print(command)

    os.system(command)

    # map/convert left hemisphere to right hemisphere
    if SIDE == "right":
        flip_hemi(SYNTH_DIR)

    print_disp_commands(SYNTH_DIR, SIDE)


def calculate_correlation():
    NORMALIZE = True
    sides = ["whole", "left", "right"]
    models = [
        "VS01n-accordion",
        "VS01n-accordion-lh-finetune",
        "VS01n-accordion-lh-finetune",
    ]
    dice_idx = 100
    results_dir = (
        "/space/calico/1/users/Harsha/SynthSeg/results/mgh_inference_20221011"
    )

    # load demographic data
    demo_df = pd.read_csv("/cluster/vive/MGH_photo_recon/mgh_demographics.csv")
    demo_df = demo_df.dropna(axis=0)
    demo_df[["Subject", "Side"]] = demo_df["Subject"].str.split(
        "_", expand=True
    )

    vol_df_list = []
    for side, model in zip(sides, models):
        vol_file = os.path.join(
            results_dir,
            f"mgh.surf.volumes.{side}.{model.lower()}.epoch_{dice_idx:03d}.csv",
        )

        # load volume data
        vol_df = pd.read_csv(vol_file)
        vol_df["subjects"] = vol_df["subjects"].str.split("_").str[0]
        vol_df = vol_df.set_index("subjects")
        vol_df.columns = vol_df.columns.astype(int)

        # combine volume data with demographic data (on subject ID)
        vol_df = vol_df.merge(
            demo_df, how="left", left_on="subjects", right_on="Subject"
        ).set_index("Subject")

        vol_columns = [item for item in vol_df.columns if isinstance(item, int)]

        # Divide Men's volumes by 1.12
        vol_df.loc[vol_df.Gender == "M", vol_columns] = (
            vol_df.loc[vol_df.Gender == "M", vol_columns] / 1.12
        )

        if NORMALIZE:
            vol_df.loc[:, vol_columns] = vol_df.loc[:, vol_columns].div(
                vol_df.sum(axis=1), axis=0
            )

        # combine left and right hemispheres into one hemisphere
        if side == "whole":
            comb_df = combine_pairs(vol_df, lh_to_rh_mapping)
            comb_df["type"] = "whole"
        else:
            vol_df.loc[:, vol_columns] = vol_df.loc[:, vol_columns] * 1
            comb_df = vol_df.copy()
            comb_df["type"] = "hemi"

        vol_df_list.append(comb_df)

    all_df = pd.concat(vol_df_list, axis=0)

    all_df = all_df.dropna(axis=0)
    # demo_df.to_csv("demo_df0.csv")

    all_df.rename(columns=labels_mapping, inplace=True)
    print(all_df.shape)

    all_df = all_df[~all_df.index.isin(list(map(str, IGNORE_SUBJECTS)))]
    print(all_df.shape)

    all_df.to_csv("mgh_corr.csv")

    return


def hue_regplot(data, x, y, hue, palette=None, **kwargs):
    from matplotlib.cm import get_cmap

    regplots = []

    levels = data[hue].unique()

    if palette is None:
        default_colors = get_cmap("tab10")
        palette = {k: default_colors(i) for i, k in enumerate(levels)}

    for key in levels:
        regplots.append(
            sns.regplot(
                x=x,
                y=y,
                data=data[data[hue] == key],
                color=palette[key],
                ci=None,
                label=key,
                **kwargs,
            )
        )

    return regplots


def plot_correlation_new():
    corr_in_file = os.path.join(os.getcwd(), "mgh_corr.csv")
    corr_out_file = os.path.join(os.getcwd(), "mgh_corr.pdf")

    df = pd.read_csv(corr_in_file)
    pp = PdfPages(corr_out_file)

    df["All"] = "All"
    for idx in lh_to_rh_mapping.keys():
        column = labels_mapping[idx]
        if column in df.columns:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            hue_regplot(data=df, x="Age", y=column, hue="Side", ax=ax1)
            hue_regplot(data=df, x="Age", y=column, hue="type", ax=ax2)
            hue_regplot(data=df, x="Age", y=column, hue="All", ax=ax3)

            ax1.set_xlabel("")
            ax3.set_xlabel("")

            ax1.set_ylabel("Volume (Proportion)")
            ax2.set_ylabel("")
            ax3.set_ylabel("")

            ax2.set_title(
                column.replace("left", "").replace("right", "").strip()
            )

            ax1.legend(loc="best")
            ax2.legend(loc="best")
            ax3.legend(loc="best")

            pp.savefig(f)
    pp.close()


def plot_correlation_old():
    corr_in_file = os.path.join(os.getcwd(), "mgh_corr.csv")
    corr_out_file = os.path.join(os.getcwd(), "mgh_corr.pdf")

    df = pd.read_csv(corr_in_file)
    pp = PdfPages(corr_out_file)

    for column in lh_to_rh_mapping.keys():
        column = str(column)
        if column in df.columns:
            fig, ax = plt.subplots()
            g = sns.lmplot(
                data=df,
                x="Age",
                y=column,
                palette="muted",
                ci=None,
                height=4,
                scatter_kws={"s": 50, "alpha": 1},
            )
            plt.title(labels_mapping[int(column)])
            pp.savefig(fig)

    # pp.close()


if __name__ == "__main__":
    # perform_segmentation()
    calculate_correlation()
    plot_correlation_new()

# demo_df = demo_df.merge(
#     vol_df, how="left", left_on="Subject", right_on="subjects"
# )

# for column in lh_to_rh_mapping.keys():
#     column = str(column)
#     try:
#         demo_df.loc[demo_df.Gender == "M", column] = demo_df[column] / 1.12
#     except:
#         pass
