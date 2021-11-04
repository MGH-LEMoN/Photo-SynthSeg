import glob
import os
import re
from shutil import copyfile

import numpy as np
from ext.lab2im import utils

from dice_config import *
from dice_utils import files_at_path


def copy_uw_recon_vols(src_path, dest_path, flag_list):
    """[summary]

    Args:
        src_path (Path String)
        dest_path (Path String)
        flag_list ([type]): [description]

    Raises:
        Exception: [description]
    """
    os.makedirs(dest_path, exist_ok=True)

    folder_list = files_at_path(src_path)

    subject_list = [
        folder for folder in folder_list
        if os.path.split(folder)[-1][0].isdigit()
    ]

    print('Copying...')
    count = 0
    for subject in subject_list:
        reconstructed_file = glob.glob(os.path.join(subject, *flag_list))

        if not len(reconstructed_file):
            continue

        if len(reconstructed_file) > 1:
            raise Exception('There are more than one reconstructed volumes')

        _, file_name = os.path.split(reconstructed_file[0])
        file_name, file_ext = os.path.splitext(file_name)

        print(file_name)
        new_file_name = '.'.join([file_name, 'grayscale']) + file_ext

        im, aff, header = utils.load_volume(reconstructed_file[0],
                                            im_only=False)

        # If n_channels = 3 convert to 1 channel by averaging
        if im.ndim == 4 and im.shape[-1] == 3:
            im = np.mean(im, axis=-1).astype('int')

        save_path = os.path.join(dest_path, new_file_name)
        utils.save_volume(im, aff, header, save_path)

        count += 1

    print(f'Copied {count} files')


def copy_uw_mri_scans(src_path, dest_path):
    """Copy MRI Scans from {src_path} to {dest_path}

    Args:
        src_path (Path String)
        dest_path (Path String)

    Notes:
        The 'NP' prefix for files at {src_path} has been
        removed and replaced '_' with '-' for consistency
        across hard and soft reconstruction names
    """
    os.makedirs(dest_path, exist_ok=True)

    src_scan_files = sorted(glob.glob(os.path.join(src_path, '*.rotated.mgz')))

    print('Copying...')
    count = 0
    for src_scan_file in src_scan_files:
        _, file_name = os.path.split(src_scan_file)

        if not re.search('^NP[0-9]*', file_name):
            continue

        print(file_name)

        dest_scan_file = file_name[2:].replace('_', '-')
        dst_scan_file = os.path.join(dest_path, dest_scan_file)

        copyfile(src_scan_file, dst_scan_file)

        count += 1

    print(f'Copied {count} files')


def copy_relevant_files():
    src_file_suffix = {
        'hard1': ['*.hard.recon.mgz'],
        'soft1': ['soft', '*_soft.mgz'],
        'hard2': ['*.hard.warped_ref.mgz'],
        'soft2': ['soft', '*_soft_regatlas.mgz'],
        'hard3': ['*samseg*.mgz'],
        'soft3': ['soft', '*samseg*.mgz'],
        'hard4': ['*manualLabel_merged.mgz'],
        'soft4': ['soft', '*manualLabel_merged.mgz']
    }

    print('\nCopying MRI Scans')
    copy_uw_mri_scans(UW_MRI_SCAN, MRI_SCANS)

    print('\nCopying Hard Reconstructions')
    copy_uw_recon_vols(UW_HARD_RECON, HARD_RECONS, src_file_suffix['hard1'])

    print('\nCopying Soft Reconstructions')
    copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECONS, src_file_suffix['soft1'])

    print('\nCopying Registered (to hard) MRI Scans')
    copy_uw_recon_vols(UW_HARD_RECON, MRI_SCANS_REG, src_file_suffix['hard2'])

    print('\nCopying I really dont know what this is')
    copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECON_REG, src_file_suffix['soft2'])

    print('\nCopying SAMSEG Segmentations (Hard)')
    copy_uw_recon_vols(UW_HARD_RECON, HARD_RECON_SAMSEG,
                       src_file_suffix['hard3'])

    print('\nCopying SAMSEG Segmentations (Soft)')
    copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECON_SAMSEG,
                       src_file_suffix['soft3'])

    print('\nCopying Hard Manual Labels')
    copy_uw_recon_vols(UW_HARD_RECON, HARD_MANUAL_LABELS_MERGED,
                       src_file_suffix['hard4'])

    print('\nCopying Soft Manual Labels')
    copy_uw_recon_vols(UW_SOFT_RECON, SOFT_MANUAL_LABELS_MERGED,
                       src_file_suffix['soft4'])


if __name__ == '__main__':
    copy_relevant_files()
