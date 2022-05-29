import json
import os

import numpy as np
from ext.lab2im import utils


class Configuration:
    """
    This configuration object is a collection of all variables relevant to the analysis
    """

    def __init__(self):
        self.PROJECT_ID = "UW.photos"
        self.SYNTHSEG_PRJCT = "/space/calico/1/users/Harsha/SynthSeg"
        self.SYNTHSEG_RESULTS = f"{self.SYNTHSEG_PRJCT}/results/jei-model-new-recons"

        self.UW_RECON = "/cluster/vive/UW_photo_recon/Photo_data"
        self.UW_MRI_SCAN = "/cluster/vive/UW_photo_recon/FLAIR_Scan_Data"

        self.PREFIX = f"{self.SYNTHSEG_RESULTS}/{self.PROJECT_ID}"

        self.MRI_SCANS = f"{self.PREFIX}.mri.scans"
        self.MRI_SCANS_SEG = f"{self.PREFIX}.mri.scans.segmentations"
        self.MRI_SCANS_REG = f"{self.PREFIX}.mri.scans.registered"
        self.MRI_SCANS_SEG_RESAMPLED = self.MRI_SCANS_SEG + ".resampled"
        self.MRI_SCANS_SEG_REG_RES = self.MRI_SCANS_SEG_RESAMPLED + ".registered"

        self.HARD_RECONS = f"{self.PREFIX}.hard.recon"
        self.HARD_RECONS3C = f"{self.PREFIX}.hard.recon3c"
        self.HARD_RECONS_HISTCORR = f"{self.PREFIX}.hard.recon.histcorr"

        self.HARD_RECON_SYNTHSEG = f"{self.PREFIX}.hard.recon.segmentations.jei"
        self.HARD_RECON_SAMSEG = f"{self.PREFIX}.hard.samseg.segmentations"
        self.HARD_MANUAL_LABELS_MERGED = f"{self.PREFIX}.hard.manual.labels"
        self.HARD_MASK_HENRY = f"{self.PREFIX}.hard.mask0"
        self.HARD_MASK = f"{self.PREFIX}.hard.mask"

        self.SOFT_RECONS = f"{self.PREFIX}.soft.recon"
        self.SOFT_RECON_REG = self.SOFT_RECONS + ".registered"
        self.SOFT_RECON_SYNTHSEG = f"{self.PREFIX}.soft.recon.segmentations.jei"
        self.SOFT_RECON_SAMSEG = f"{self.PREFIX}.soft.samseg.segmentations"
        self.SOFT_MANUAL_LABELS_MERGED = f"{self.PREFIX}.soft.manual.labels"
        self.SOFT_MASK = f"{self.SYNTHSEG_RESULTS}/UW.photos.soft.mask"
        self.SOFT_MASK_HENRY = f"{self.PREFIX}.soft.mask0"

        self.SURFACE_RECONS = f"{self.PREFIX}.surface.recon"

        # Note: All of these are in photo RAS space (just resampling based on reference)
        self.MRI_SYNTHSEG_IN_SAMSEG_SPACE = (
            self.MRI_SCANS_SEG_REG_RES + ".in_samseg_space"
        )
        self.MRI_SYNTHSEG_IN_SOFTSAMSEG_SPACE = (
            self.MRI_SCANS_SEG_REG_RES + ".in_softsamseg_space"
        )
        self.HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE = (
            self.HARD_RECON_SYNTHSEG + ".in_samseg_space"
        )
        self.HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE = (
            self.HARD_RECON_SYNTHSEG + ".in_mri_space"
        )
        self.SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE = (
            self.SOFT_RECON_SYNTHSEG + ".in_samseg_space"
        )

        self.mri_synthseg_vols_file = f"{self.PREFIX}.mri.scans.segmentations.csv"
        self.soft_synthseg_vols_file = f"{self.PREFIX}.soft.recon.segmentations.jei.csv"
        self.hard_synthseg_vols_file = f"{self.PREFIX}.hard.recon.segmentations.jei.csv"

        #### Extract SAMSEG Volumes
        self.UW_HARD_RECON_HENRY = (
            "/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard"
        )
        self.UW_SOFT_RECON_HENRY = (
            "/cluster/vive/UW_photo_recon/recons/results_Henry/Results_soft"
        )

        self.HARD_SAMSEG_STATS = f"{self.UW_HARD_RECON_HENRY}/SAMSEG/"
        self.SOFT_SAMSEG_STATS = f"{self.UW_SOFT_RECON_HENRY}/SAMSEG/"

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
        # self.IGNORE_SUBJECTS = ['18-1343', '18-2260', '19-0019']
        self.IGNORE_SUBJECTS = []

        self._write_config()

        self.required_labels = np.array(
            list(set(self.ALL_LABELS) - set(self.IGNORE_LABELS))
        )

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


if __name__ == "__main__":
    config = Configuration()
