import glob
import os
import numpy as np
import shutil

from ext.lab2im import utils

IGNORE_SUBJECTS = [
    "2705_left",
    "2706_whole",
    "2707_whole",
    "2708_left",
    "2710_whole",
    "2711_left",
]

lh_to_rh_mapping = {
    2: 41,
    3: 42,
    4: 43,
    5: 44,
    7: 46,
    8: 47,
    10: 49,
    11: 50,
    12: 51,
    13: 52,
    17: 53,
    18: 54,
    26: 58,
    28: 60,
}


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


if __name__ == "__main__":
    SIDE = "left"
    # "left" | "right" | "whole"
    PRJCT_DIR = "/cluster/vive/MGH_photo_recon"
    subjects = sorted(glob.glob(os.path.join(PRJCT_DIR, f"*{SIDE}*")))
    subjects_ids = [os.path.basename(s) for s in subjects]

    print(subjects_ids)

    MODEL = "VS01n-lh"
    DICE_IDX = 100
    H5_FILE = f"/space/calico/1/users/Harsha/SynthSeg/models/models-2022/{MODEL}/dice_{DICE_IDX:03d}.h5"
    LABEL_LIST = "/space/calico/1/users/Harsha/SynthSeg/models/jei-model/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.label_list_lh.npy"

    MISC_DIR = os.path.join(
        "/space/calico/1/users/Harsha/SynthSeg",
        "results",
        "mgh_inference_20221003",
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

    # --flip \
    command = f"python {os.getcwd()}/scripts/commands/predict.py \
    --smoothing 0.5 \
    --biggest_component \
    --padding 256 \
    --vol {MISC_DIR}/mgh.surf.volumes.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d} \
    --neutral_labels 5 \
    --post {MISC_DIR}/mgh.surf.posterior.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d} \
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
