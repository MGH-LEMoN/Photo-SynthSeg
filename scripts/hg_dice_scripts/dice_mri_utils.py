import os

from ext.lab2im import utils
from nipype.interfaces.freesurfer import MRIConvert

from dice_config import *
from dice_utils import files_at_path, id_check, return_common_subjects


def run_mri_convert(in_file, ref_file, out_file):
    mc = MRIConvert()
    mc.terminal_output = 'none'
    mc.inputs.in_file = in_file
    mc.inputs.out_file = out_file
    mc.inputs.reslice_like = ref_file
    mc.inputs.out_type = 'mgz'
    mc.inputs.out_datatype = 'float'
    mc.inputs.resample_type = 'nearest'

    mc.run()


def perform_overlay():
    mri_scans_reg = files_at_path(MRI_SCANS_REG)
    mri_resampled_segs = files_at_path(MRI_SCANS_SEG_RESAMPLED)

    mri_scans_reg, mri_resampled_segs = return_common_subjects(
        mri_scans_reg, mri_resampled_segs)

    os.makedirs(MRI_SCANS_SEG_REG_RES, exist_ok=True)

    print('Creating...')
    for scan_reg, mri_resampled_seg in zip(mri_scans_reg, mri_resampled_segs):
        id_check(scan_reg, mri_resampled_seg)

        _, scan_reg_aff, scan_reg_head = utils.load_volume(scan_reg,
                                                           im_only=False)
        mrs_im = utils.load_volume(mri_resampled_seg)

        _, file_name = os.path.split(mri_resampled_seg)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + '.reg' + file_ext
        out_file = os.path.join(MRI_SCANS_SEG_REG_RES, out_file)

        # We can now combine the segmentation voxels with the registered header.
        utils.save_volume(mrs_im, scan_reg_aff, scan_reg_head, out_file)

        # this new file should overlay with the 3D photo reconstruction
