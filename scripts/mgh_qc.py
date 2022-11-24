#!/usr/bin/env python3

import os
import glob
import argparse

## !!! INSTRUCTIONS !!! ## (You do this only once)
# 1. Place this script in your prefered directory (/path/to/script)
# 2. Run chmod +x /path/to/script
# 3. Create an alias in your ~/.bashrc file: alias mgh_qc='/path/to/script'
# 4. source ~/.bashrc
# 5. Example: mgh_qc <subject_id>

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_id", type=str)
    args = parser.parse_args()

    PRJCT_DIR = "/cluster/vive/MGH_photo_recon"
    subject_dir = glob.glob(os.path.join(PRJCT_DIR, f"{args.subject_id}*"))

    if not len(subject_dir):
        print(f"Subject: {args.subject_id} Does Not Exist")
        exit()
    elif len(subject_dir) > 1:
        print(f"Multiple folders for subject: {args.subject_id}")
        exit()
    else:
        subject_dir = subject_dir[0]

    subject_id = os.path.basename(subject_dir)
    SID, SIDE = subject_id.split("_")

    sides = ["whole", "left", "right"]
    models = [
        "VS01n-accordion",
        "VS01n-accordion-lh-finetune",
        "VS01n-accordion-lh-finetune",
    ]
    model_dict = dict(zip(sides, models))

    MODEL_DIR = os.path.join(
        "/space/calico/1/users/Harsha/SynthSeg", "models", "models-2022"
    )
    MODEL = model_dict[SIDE]
    DICE_IDX = 100
    H5_FILE = f"{MODEL_DIR}/{MODEL}/dice_{DICE_IDX:03d}.h5"

    MISC_DIR = os.path.join(
        "/space/calico/1/users/Harsha/SynthSeg",
        "results",
        "mgh_inference_20221122",
    )
    RECON_DIR = os.path.join(
        MISC_DIR, f"mgh.surf.recon.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d}"
    )
    SYNTH_DIR = os.path.join(
        MISC_DIR,
        f"mgh.surf.synthseg.{SIDE}.{MODEL.lower()}.epoch_{DICE_IDX:03d}",
    )

    synth_file = glob.glob(os.path.join(SYNTH_DIR, f"{SID}*_synthseg.mgz"))[0]
    recon_file = f"{PRJCT_DIR}/{subject_id}/recon_202212/photo_recon.mgz"
    surf_file = (
        f"{PRJCT_DIR}/{subject_id}/recon_202212/registered_reference.surf"
    )

    if not os.path.isfile(synth_file):
        print(f"DNE: {synth_file}")
        exit()

    if not os.path.isfile(recon_file):
        print(f"DNE: {recon_file}")
        exit()

    if not os.path.isfile(surf_file):
        print(f"DNE: {surf_file}")
        exit()

    command = f"freeview {recon_file}:rgb=1 \
                        {synth_file}:colormap=lut:opacity=0.5:visible=1 \
                        -f {surf_file}"
    command = " ".join(command.split())

    print(command)
    os.system(command)
