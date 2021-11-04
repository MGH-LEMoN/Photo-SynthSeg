import glob
import os

from dice_config import *


def files_at_path(path_str):
    return sorted(glob.glob(os.path.join(path_str, '*')))


def id_check(scan_reg, mri_resampled_seg):
    scan_reg_fn = os.path.split(scan_reg)[-1]
    mri_resampled_seg_fn = os.path.split(mri_resampled_seg)[-1]

    assert scan_reg_fn[:7] == mri_resampled_seg_fn[:7], 'File MisMatch'

    if scan_reg_fn[:7] in IGNORE_SUBJECTS:
        return 0
    else:
        print(scan_reg_fn[:7])
        return 1


def return_common_subjects(file_list1, file_list2):
    if len(file_list1) != len(file_list2):
        print('Mismatch: Length of input files != Length of Reference files')

        input_names = {
            os.path.split(input_file)[-1][:7]: input_file
            for input_file in file_list1
        }
        reference_names = {
            os.path.split(reference_file)[-1][:7]: reference_file
            for reference_file in file_list2
        }

        common_names = set(input_names.keys()).intersection(
            reference_names.keys())

        file_list1 = [input_names[key] for key in common_names]
        file_list2 = [reference_names[key] for key in common_names]

    return file_list1, file_list2
