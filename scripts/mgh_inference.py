import glob
import os
import numpy as np
import shutil

from ext.lab2im import utils

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


if __name__ == "__main__":
    SIDE = "right"  # "left" | "right" | "whole"
    PRJCT_DIR = "/cluster/vive/MGH_photo_recon"
    subjects = sorted(glob.glob(os.path.join(PRJCT_DIR, f"*{SIDE}*")))
    subjects_ids = [os.path.basename(s) for s in subjects]

    MODEL = "VS01n-accordion-lh"
    DICE_IDX = 30
    H5_FILE = f"/space/calico/1/users/Harsha/SynthSeg/models/models-2022/{MODEL}/dice_{DICE_IDX:03d}.h5"
    LABEL_LIST = "/space/calico/1/users/Harsha/SynthSeg/models/jei-model/SynthSegPhotos_no_brainstem_or_cerebellum_4mm.label_list_lh.npy"

    MISC_DIR = os.path.join(
        "/space/calico/1/users/Harsha/SynthSeg", "results", "mgh_20220928"
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
        if not os.path.exists(os.path.join(subject, "recon_new")):
            print(f"{os.path.basename(subject)} - Skipping")
            continue

        src = os.path.join(subject, "recon_new", "photo_recon.mgz")

        if not os.path.isfile(src):
            print(f"{os.path.basename(subject)} - Skipping")
            continue

        print(os.path.basename(subject))
        if SIDE == "right":
            # Load Label Map
            im, aff, hdr = utils.load_volume(src, im_only=False)
            im_flip = np.flip(im, 0)
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
        for file in glob.glob(os.path.join(SYNTH_DIR, "*.mgz")):
            im, aff, hdr = utils.load_volume(file, im_only=False)
            im_flip = np.flip(im, 0)
            im = relabel_volume(im, lh_to_rh_mapping)
            utils.save_volume(im, aff, hdr, file.replace("rh2lh", "lh2rh"))
