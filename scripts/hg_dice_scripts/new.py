import glob
import json
import os
import subprocess
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ext.lab2im import utils

from dice_calculations import calculate_dice_for_dict
from dice_gather import copy_relevant_files
from dice_mri_utils import (convert_to_single_channel,
                            move_volumes_into_target_spaces, perform_overlay)
from dice_plots import write_plots
from dice_utils import run_make_target
from dice_volumes import write_correlations_to_file, write_volumes_to_file
from uw_config import (CORRELATIONS_LIST, DICE2D_LIST, PLOTS_LIST,
                       SAMSEG_GATHER_DICT, VOLUMES_LIST)

# use this dictionary to gather files from source to destination
file_gather_dict = {
    "mri_scan": {
        "source": "UW_MRI_SCAN",
        "destination": "MRI_SCANS",
        "expr": ["*.rotated.mgz"],
        "message": "Original 3D Scans",
    },
    "image_ref": {
        "source": "UW_MRI_SCAN",
        "destination": "MRI_SCANS_REF",
        "expr": ["*.rotated.masked.mgz"],
        "message": "3D Volume Masks",
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
        "expr": ["ref_mask_skip_2", "*recon.mgz"],
        "message": "Hard Reconstructions",
    },
    "soft_recon": {
        "source": "UW_SOFT_RECON",
        "destination": ["SOFT_RECONS3C", "SOFT_RECONS"],
        "expr": ["ref_soft_mask_skip_2", "*recon.mgz"],
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
        "expr": ["ref_soft_mask_skip_2", "registered_reference.mgz"],
        "message": "Soft Warped References",
    },
    # "hard_samseg": {
    #     "source": "UW_HARD_SAMSEG",
    #     "destination": "HARD_SAMSEG",
    #     "expr": ["*seg.mgz"],
    #     "message": "Hard SAMSEG",
    # },
    # "soft_samseg": {
    #     "source": "UW_SOFT_SAMSEG",
    #     "destination": "SOFT_SAMSEG",
    #     "expr": ["*seg.mgz"],
    #     "message": "Soft SAMSEG",
    # },
    "hard_gt_labels": {
        "source": "UW_HARD_RECON",
        "destination": "HARD_MANUAL_LABELS_MERGED",
        "expr": ["ref_mask_skip_2", "propagated_labels", "*_seg_output.mgz"],
        "message": "Hard Ground Truth",
    },
    "soft_gt_labels": {
        "source": "UW_SOFT_RECON",
        "destination": "SOFT_MANUAL_LABELS_MERGED",
        "expr": ["ref_soft_mask_2", "propagated_labels", "*seg_output.mgz"],
        "message": "Soft Ground Truth",
    },
}

mri_convert_items = [
    {
        "source": "MRI_SCANS_SYNTHSEG",
        "reference": "HARD_SAMSEG_C0",
        "target": "MRI_SYNTHSEG_IN_SAMSEG_SPACE",
    },
    {
        "source": "HARD_SYNTHSEG",
        "reference": "HARD_SAMSEG_C0",
        "target": "HARD_SYNTHSEG_IN_SAMSEG_SPACE",
    },
    {
        "source": "HARD_SYNTHSEG",
        "reference": "MRI_SCANS_SYNTHSEG",
        "target": "HARD_SYNTHSEG_IN_MRISEG_SPACE",
    },
    {
        "source": "MRI_SCANS_SYNTHSEG",
        "reference": "SOFT_SAMSEG_C0",
        "target": "MRI_SYNTHSEG_IN_SAMSEG_SPACE",
    },
    {
        "source": "SOFT_SYNTHSEG",
        "reference": "SOFT_SAMSEG_C0",
        "target": "SOFT_SYNTHSEG_IN_SAMSEG_SPACE",
    },
]


class Configuration:
    def __init__(self, project_dir, args):
        self._git_url = subprocess.check_output(
            'git config --get remote.origin.url'.split()).decode(
                'ascii').strip()
        self._git_branch = subprocess.check_output(
            'git rev-parse --abbrev-ref HEAD'.split()).decode('ascii').strip()
        self._git_commit_hash = subprocess.check_output(
            'git rev-parse --short HEAD'.split()).decode('ascii').strip()

        self.model_name = args.model_name
        self.SYNTHSEG_PRJCT = project_dir
        self.SYNTHSEG_RESULTS = os.path.join(project_dir, 'results',
                                             args.out_dir_name,
                                             f'{args.recon_flag}-recons-skip2',
                                             self.model_name)

        # input folders
        self.UW_HARD_RECON = "/cluster/vive/UW_photo_recon/Photo_data/"
        self.UW_SOFT_RECON = "/cluster/vive/UW_photo_recon/Photo_data/"
        self.UW_MRI_SCAN = "/cluster/vive/UW_photo_recon/FLAIR_Scan_Data"

        self.SAMSEG_OUTPUT_HARD_C0 = f"{self.SYNTHSEG_RESULTS}/SAMSEG_OUTPUT_HARD_C0"
        self.SAMSEG_OUTPUT_SOFT_C0 = f"{self.SYNTHSEG_RESULTS}/SAMSEG_OUTPUT_SOFT_C0"
        self.SAMSEG_OUTPUT_HARD_C1 = f"{self.SYNTHSEG_RESULTS}/SAMSEG_OUTPUT_HARD_C1"
        self.SAMSEG_OUTPUT_SOFT_C1 = f"{self.SYNTHSEG_RESULTS}/SAMSEG_OUTPUT_SOFT_C1"
        self.SAMSEG_OUTPUT_HARD_C2 = f"{self.SYNTHSEG_RESULTS}/SAMSEG_OUTPUT_HARD_C2"
        self.SAMSEG_OUTPUT_SOFT_C2 = f"{self.SYNTHSEG_RESULTS}/SAMSEG_OUTPUT_SOFT_C2"

        # output folders
        self.MRI_SCANS = f"{self.SYNTHSEG_RESULTS}/mri.scans"
        self.MRI_SCANS_REF = f"{self.SYNTHSEG_RESULTS}/mri.scans.ref"
        self.MRI_SCANS_SYNTHSEG = f"{self.SYNTHSEG_RESULTS}/mri.synthseg"

        self.MRI_SCANS_SYNTHSEG_RESAMPLED = self.MRI_SCANS_SYNTHSEG + ".resampled"
        self.MRI_SCANS_SYNTHSEG_REG_RES = (self.MRI_SCANS_SYNTHSEG_RESAMPLED +
                                           ".registered")

        self.HARD_RECONS3C = f"{self.SYNTHSEG_RESULTS}/hard.recon3c"
        self.HARD_REF = f"{self.SYNTHSEG_RESULTS}/hard.ref"
        self.HARD_REF_WARPED = f"{self.SYNTHSEG_RESULTS}/hard.ref.warped"
        self.HARD_RECONS = f"{self.SYNTHSEG_RESULTS}/hard.recon"
        self.HARD_SYNTHSEG = f"{self.SYNTHSEG_RESULTS}/hard.synthseg"
        self.HARD_SAMSEG = f"{self.SYNTHSEG_RESULTS}/hard.samseg"
        self.HARD_SAMSEG_C0 = f"{self.SYNTHSEG_RESULTS}/hard.samseg.c0"
        self.HARD_SAMSEG_C1 = f"{self.SYNTHSEG_RESULTS}/hard.samseg.c1"
        self.HARD_SAMSEG_C2 = f"{self.SYNTHSEG_RESULTS}/hard.samseg.c2"
        self.HARD_MANUAL_LABELS_MERGED = f"{self.SYNTHSEG_RESULTS}/hard.manual.labels"

        self.SOFT_RECONS3C = f"{self.SYNTHSEG_RESULTS}/soft.recon3c"
        self.SOFT_RECONS = f"{self.SYNTHSEG_RESULTS}/soft.recon"
        self.SOFT_REF_WARPED = f"{self.SYNTHSEG_RESULTS}/soft.ref.warped"
        self.SOFT_SYNTHSEG = f"{self.SYNTHSEG_RESULTS}/soft.synthseg"
        self.SOFT_SAMSEG = f"{self.SYNTHSEG_RESULTS}/soft.samseg"
        self.SOFT_SAMSEG_C1 = f"{self.SYNTHSEG_RESULTS}/soft.samseg.c1"
        self.SOFT_SAMSEG_C2 = f"{self.SYNTHSEG_RESULTS}/soft.samseg.c2"
        self.SOFT_SAMSEG_C0 = f"{self.SYNTHSEG_RESULTS}/soft.samseg.c0"
        self.SOFT_MANUAL_LABELS_MERGED = f"{self.SYNTHSEG_RESULTS}/soft.manual.labels"

        # Note: All of these are in photo RAS space (just resampling based on reference)
        self.MRI_SYNTHSEG_IN_SAMSEG_SPACE = (self.MRI_SCANS_SYNTHSEG_REG_RES +
                                             ".in_samseg_space")
        self.MRI_SYNTHSEG_IN_SOFTSAMSEG_SPACE = (
            self.MRI_SCANS_SYNTHSEG_REG_RES + ".in_softsamseg_space")
        self.HARD_SYNTHSEG_IN_SAMSEG_SPACE = self.HARD_SYNTHSEG + ".in_samseg_space"
        self.HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE = self.HARD_SYNTHSEG + ".in_mri_space"
        self.SOFT_SYNTHSEG_IN_SAMSEG_SPACE = self.SOFT_SYNTHSEG + ".in_samseg_space"

        self.mri_synthseg_vols_file = (
            f"{self.SYNTHSEG_RESULTS}/volumes/mri.synthseg.volumes.csv")
        self.soft_synthseg_vols_file = (
            f"{self.SYNTHSEG_RESULTS}/volumes/soft.synthseg.volumes.csv")
        self.hard_synthseg_vols_file = (
            f"{self.SYNTHSEG_RESULTS}/volumes/hard.synthseg.volumes.csv")

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
        # self.IGNORE_SUBJECTS = ["18-1343", "18-2260", "19-0100"]
        self.IGNORE_SUBJECTS = ["19-0019"]

        self.required_labels = list(
            set(self.ALL_LABELS) - set(self.IGNORE_LABELS))

        self._make_dirs()

    def _write_config(self, file_name=None):
        """Write configuration to a file
        Args:
            CONFIG (dict): configuration
        """
        file_name = "config.json" if file_name is None else file_name

        dictionary = self.__dict__
        json_object = json.dumps(dictionary, sort_keys=True, indent=4)

        if not dictionary.get("SYNTHSEG_RESULTS", None):
            return

        utils.mkdir(dictionary["SYNTHSEG_RESULTS"])

        config_file = os.path.join(dictionary["SYNTHSEG_RESULTS"], file_name)

        with open(config_file, "w") as outfile:
            outfile.write(json_object)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        self._write_config()

    def _make_dirs(self):
        self.dice_dir = os.path.join(self.SYNTHSEG_RESULTS, "dice_files")
        self.volumes_dir = os.path.join(self.SYNTHSEG_RESULTS, "volumes")

        os.makedirs(self.dice_dir, exist_ok=True)
        os.makedirs(self.volumes_dir, exist_ok=True)


if __name__ == "__main__":
    # !!! START HERE !!!
    PRJCT_DIR = "/space/calico/1/users/Harsha/SynthSeg"

    parser = ArgumentParser()
    parser.add_argument("--recon_flag",
                        type=str,
                        dest="recon_flag",
                        default=None)
    parser.add_argument("--out_dir_name",
                        type=str,
                        dest="out_dir_name",
                        default=None)
    parser.add_argument("--model_name",
                        type=str,
                        dest="model_name",
                        default=None)
    parser.add_argument("--part", type=int, dest="part", default=None)
    args = parser.parse_args()

    config = Configuration(PRJCT_DIR, args)

    if args.part == 1:
        copy_relevant_files(config, file_gather_dict)

        # # It looks like SAMSEG doesn't need 3Channel images.
        # # TODO: Do away with the following 2 lines of code
        convert_to_single_channel(config, "HARD_RECONS")
        convert_to_single_channel(config, "SOFT_RECONS")

        # print("Running SynthSeg...")
        # # Due to some code incompatibility issues, the following lines of code
        # # have to be run separately on MLSC or this entire script can be run on MLSC
        # run_make_target(config, "hard")
        # run_make_target(config, "soft")
        # run_make_target(config, "scans")

    if args.part == 2:
        SAMSEG_LIST = glob.glob(
            os.path.join(os.path.dirname(getattr(config, "SYNTHSEG_RESULTS")),
                         'SAMSEG_OUTPUT_*'))
        for src in SAMSEG_LIST:
            basename = os.path.basename(src)
            dst = os.path.join(getattr(config, "SYNTHSEG_RESULTS"), basename)
            try:
                os.symlink(src, dst)
            except OSError as e:
                pass

        copy_relevant_files(config, SAMSEG_GATHER_DICT)
        write_volumes_to_file(config, VOLUMES_LIST)
        for item in CORRELATIONS_LIST:
            write_correlations_to_file(config, VOLUMES_LIST, *item)

        move_volumes_into_target_spaces(config, mri_convert_items)
        calculate_dice_for_dict(config, DICE2D_LIST)

    if args.part == 3:
        write_plots(config, PLOTS_LIST)
