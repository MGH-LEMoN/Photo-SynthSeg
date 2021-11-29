import json
import os

import numpy as np
from ext.lab2im import utils


class Configuration():
    def __init__(self):
        self.SYNTHSEG_PRJCT = '/space/calico/1/users/Harsha/SynthSeg'
        self.SYNTHSEG_RESULTS = f'{self.SYNTHSEG_PRJCT}/results/jei-model'

        self.UW_HARD_RECON = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard'
        self.UW_SOFT_RECON = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_soft'
        self.UW_MRI_SCAN = '/cluster/vive/UW_photo_recon/FLAIR_Scan_Data'

        self.MRI_SCANS = f'{self.SYNTHSEG_RESULTS}/UW.photos.mri.scans'
        self.MRI_SCANS_SEG = f'{self.SYNTHSEG_RESULTS}/UW.photos.mri.scans.segmentations'
        self.MRI_SCANS_REG = f'{self.SYNTHSEG_RESULTS}/UW.photos.mri.scans.registered'
        self.MRI_SCANS_SEG_RESAMPLED = self.MRI_SCANS_SEG + '.resampled'
        self.MRI_SCANS_SEG_REG_RES = self.MRI_SCANS_SEG_RESAMPLED + '.registered'

        self.HARD_RECONS = f'{self.SYNTHSEG_RESULTS}/UW.photos.hard.recon'
        self.HARD_RECON_SYNTHSEG = f'{self.SYNTHSEG_RESULTS}/UW.photos.hard.recon.segmentations.jei'
        self.HARD_RECON_SAMSEG = f'{self.SYNTHSEG_RESULTS}/UW.photos.hard.samseg.segmentations'
        self.HARD_MANUAL_LABELS_MERGED = f'{self.SYNTHSEG_RESULTS}/UW.photos.hard.manual.labels'

        self.SOFT_RECONS = f'{self.SYNTHSEG_RESULTS}/UW.photos.soft.recon'
        self.SOFT_RECON_REG = self.SOFT_RECONS + '.registered'
        self.SOFT_RECON_SYNTHSEG = f'{self.SYNTHSEG_RESULTS}/UW.photos.soft.recon.segmentations.jei'
        self.SOFT_RECON_SAMSEG = f'{self.SYNTHSEG_RESULTS}/UW.photos.soft.samseg.segmentations'
        self.SOFT_MANUAL_LABELS_MERGED = f'{self.SYNTHSEG_RESULTS}/UW.photos.soft.manual.labels'

        # Note: All of these are in photo RAS space (just resampling based on reference)
        self.MRI_SYNTHSEG_IN_SAMSEG_SPACE = self.MRI_SCANS_SEG_REG_RES + '.in_samseg_space'
        self.MRI_SYNTHSEG_IN_SOFTSAMSEG_SPACE = self.MRI_SCANS_SEG_REG_RES + '.in_softsamseg_space'
        self.HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE = self.HARD_RECON_SYNTHSEG + '.in_samseg_space'
        self.HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE = self.HARD_RECON_SYNTHSEG + '.in_mri_space'
        self.SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE = self.SOFT_RECON_SYNTHSEG + '.in_samseg_space'

        self.mri_synthseg_vols_file = f'{self.SYNTHSEG_RESULTS}/UW.photos.mri.scans.segmentations.csv'
        self.soft_synthseg_vols_file = f'{self.SYNTHSEG_RESULTS}/UW.photos.soft.recon.segmentations.jei.csv'
        self.hard_synthseg_vols_file = f'{self.SYNTHSEG_RESULTS}/UW.photos.hard.recon.segmentations.jei.csv'

        #### Extract SAMSEG Volumes
        self.HARD_SAMSEG_STATS = f'{self.UW_HARD_RECON}/SAMSEG/'
        self.SOFT_SAMSEG_STATS = f'{self.UW_SOFT_RECON}/SAMSEG/'

        self.ALL_LABELS = [
            0, 2, 3, 4, 5, 10, 11, 12, 13, 14, 17, 18, 26, 28, 41, 42, 43, 44,
            49, 50, 51, 52, 53, 54, 58, 60
        ]
        self.IGNORE_LABELS = [0, 5, 14, 26, 28, 44, 58, 60]
        self.ADDL_IGNORE_LABELS = [7, 8, 15, 16, 46, 47]
        self.LABEL_PAIRS = [(2, 41), (3, 42), (4, 43), (10, 49), (11, 50),
                            (12, 51), (13, 52), (17, 53), (18, 54)]
        self.LABEL_PAIR_NAMES = [
            'White Matter', 'Cortex', 'Ventricle', 'Thalamus', 'Caudate',
            'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala'
        ]
        self.IGNORE_SUBJECTS = ['18-1343', '18-2260', '19-0019']

        config_dict = vars()
        self._write_config(config_dict, 'config.json')

        required_labels = np.array(
            list(set(self.ALL_LABELS) - set(self.IGNORE_LABELS)))

    def _write_config(self, dictionary, file_name=None):
        """Write configuration to a file
        Args:
            CONFIG (dict): configuration
        """
        file_name = 'config.json' if file_name is None else file_name

        json_object = json.dumps(dictionary, sort_keys=True, indent=4)

        utils.mkdir(dictionary['SYNTHSEG_RESULTS'])

        config_file = os.path.join(dictionary['SYNTHSEG_RESULTS'], file_name)

        with open(config_file, "w") as outfile:
            outfile.write(json_object)
