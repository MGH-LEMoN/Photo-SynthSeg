"""Contains code to run my trained models on Henry's reconstructed volumes
"""

import glob
import os
import re
from pprint import pprint
from shutil import copyfile

import numpy as np
from nipype.interfaces.freesurfer import MRIConvert

from ext.lab2im import utils
from SynthSeg.evaluate import fast_dice

SYNTHSEG_PRJCT = '/space/calico/1/users/Harsha/SynthSeg'

UW_HARD_RECON_PATH = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard'
UW_SOFT_RECON_PATH = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_soft'
UW_MRI_SCAN_PATH = '/cluster/vive/UW_photo_recon/FLAIR_Scan_Data'

MRI_SCANS_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.mri.scans'
MRI_SCANS_SEG_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.mri.scans.segmentations'
MRI_SCANS_SEG_RESAMPLED_PATH = MRI_SCANS_SEG_PATH + '.resampled'
MRI_SCANS_SEG_REG_PATH = MRI_SCANS_SEG_RESAMPLED_PATH + '.registered'

HARD_RECONS_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.hard.recon'
HARD_RECON_SEG_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.hard.recon.segmentations.jei'
HARD_RECON_SEG_RESAMPLED_PATH = HARD_RECON_SEG_PATH + '.resampled'
HARD_RECON_REG_PATH = HARD_RECONS_PATH + '.reg'

SOFT_RECONS_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.soft.recon'
SOFT_RECON_SEG_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.soft.recon.segmentations.jei'
SOFT_RECON_SEG_RESAMPLED_PATH = SOFT_RECON_SEG_PATH + '.resampled'
SOFT_RECON_REG_PATH = SOFT_RECONS_PATH + '.reg'


def files_at_path(path_str):
    return sorted(glob.glob(os.path.join(path_str, '*')))


def copy_uw_recon_vols(src_path, dest_path, flag_list):
    """[summary]

    Raises:
        Exception: [description]
    """

    os.makedirs(dest_path, exist_ok=True)

    folder_list = files_at_path(src_path)

    subject_list = [
        folder for folder in folder_list if re.search('[0-9]', folder)
    ]

    print('Copying...')
    for subject in subject_list:
        reconstructed_file = glob.glob(os.path.join(subject, *flag_list))
        if len(reconstructed_file) > 1:
            raise Exception('There are more than one reconstructed volumes')

        _, file_name = os.path.split(reconstructed_file[0])
        file_name, file_ext = os.path.splitext(file_name)

        print(file_name)
        new_file_name = '.'.join([file_name, 'grayscale']) + file_ext
        file_name = 'NP' + file_name.replace('-', '_')

        im, aff, header = utils.load_volume(reconstructed_file[0],
                                            im_only=False)
        im = np.mean(im, axis=-1).astype('int')

        save_path = os.path.join(dest_path, new_file_name)
        utils.save_volume(im, aff, header, save_path)


def copy_uw_mri_scans(src_path, dest_path):
    os.makedirs(dest_path, exist_ok=True)

    folder_list = sorted(os.listdir(os.path.join(UW_HARD_RECON_PATH)))
    subject_list = [
        folder.replace('-', '_') for folder in folder_list
        if re.search('[0-9]', folder)
    ]

    print('Copying...')
    for subject in subject_list:
        print(subject)
        scan_file = 'NP' + subject + '.rotated.mgz'
        src_scan_file = os.path.join(src_path, scan_file)
        dst_scan_file = os.path.join(dest_path, scan_file)

        copyfile(src_scan_file, dst_scan_file)


def run_mri_convert(in_file, out_file, ref_file):
    mc = MRIConvert()
    mc.terminal_output = 'none'
    mc.inputs.in_file = in_file
    mc.inputs.out_file = out_file
    mc.inputs.reslice_like = ref_file
    mc.inputs.out_type = 'mgz'
    mc.inputs.out_datatype = 'float'
    mc.inputs.resample_type = 'nearest'

    mc.run()


def run_make_target(flag):
    os.system(f'make -C {SYNTHSEG_PRJCT} predict-{flag}')


def perform_registration(segmentations_path, reference_path, output_path):
    mri_scan_segs = files_at_path(segmentations_path)
    mri_scans = files_at_path(reference_path)

    os.makedirs(output_path, exist_ok=True)

    print('Creating...')
    for mri_scan_seg, mri_scan in zip(mri_scan_segs, mri_scans):
        assert mri_scan[:10] == mri_scan_seg[:10]

        _, file_name = os.path.split(mri_scan)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + '.res' + file_ext
        print(out_file)
        out_file = os.path.join(output_path, out_file)

        run_mri_convert(mri_scan_seg, out_file, mri_scan)


def perform_overlay():
    mri_scans = files_at_path(MRI_SCANS_PATH)
    mri_segs_resampled = files_at_path(MRI_SCANS_SEG_RESAMPLED_PATH)
    MRI_SCANS_REG_PATH = None
    mri_reg_scans = files_at_path(MRI_SCANS_REG_PATH)

    os.makedirs(MRI_SCANS_SEG_REG_PATH, exist_ok=True)

    for scan, reg_scan, resampled_seg in zip(mri_scans, mri_reg_scans,
                                             mri_segs_resampled):
        assert scan[:10] == reg_scan[:10] == resampled_seg[:10], 'File MisMatch'

        _, file_name = os.path.split(resampled_seg)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + '.reg' + file_ext
        print(out_file)
        out_file = os.path.join(MRI_SCANS_SEG_REG_PATH, out_file)

        # read in the MRI and SynthSeg segmentation, which now live in the same space
        scan_im, scan_aff, scan_head = utils.load_volume(scan, im_only=False)
        scan_seg_im = utils.load_volume(resampled_seg)

        # now read the registered MRI, which is essentially the same voxels but with a different header. This scan should overlay with the 3D photo reconstruction.
        scan_reg_im, scan_reg_aff, scan_reg_head = utils.load_volume(reg_scan)

        # We can now combine the segmentation voxels with the registered header.
        utils.save_volume(scan_seg_im, scan_reg_head, out_file)
        # this new file should overlay with the 3D photo reconstruction


def some_function():
    # put the synthseg segmentation in the same space as the input
    perform_registration(MRI_SCANS_SEG_PATH, MRI_SCANS_PATH,
                         MRI_SCANS_SEG_RESAMPLED_PATH)

    perform_overlay()

    # The photo segmentation (photo_synthseg.mgz) and the ground truth segmentation in photo space (registered_segmentation.mgz)
    # % overlay in Freeview but do not live in the same voxel space as they have different headers. You need to resample the segmentation of the photos on
    # % the space of the ground truth (which is cleaner than the other way around, me thinks? I don’t know. It shouldn’t be too different…
    # mri_convert photo_synthseg.mgz photo_synthseg_resampled.mgz -rl registered_segmentation.mgz -rt nearest -odt float

    perform_registration(HARD_RECON_SEG_PATH, REG_SEG_PATH,
                         HARD_RECON_SEG_RESAMPLED_PATH)


def calculate_dice(reference_path, segmentation_path):
    reference_list = files_at_path(reference_path)
    segmentation_list = files_at_path(segmentation_path)

    for ref_file, seg_file in zip(reference_list, segmentation_list):
        assert ref_file[:10] == seg_file[:10], 'File Mismatch'

        ref_vol = utils.load_volume(ref_file)
        seg_vol = utils.load_volume(seg_file)

        assert ref_vol.shape == seg_vol.shape, "Shape mismatch"

        ref_vol_labels, seg_vol_labels = np.unique(ref_vol), np.unique(seg_vol)

    common_labels = np.array(
        list(set(ref_vol_labels).intersection(seg_vol_labels)))

    dice_coeff = fast_dice(ref_vol, seg_vol, common_labels)


def main():
    src_file_suffix = {
        'hard1': ['*.hard.recon.mgz'],
        'soft1': ['soft', '*_soft.mgz'],
        'hard2': ['*.hard.warped_ref.mgz'],
        'soft2': ['soft', '*_soft_regatlas.mgz']
    }

    # copy_uw_mri_scans(UW_MRI_SCAN_PATH, MRI_SCANS_PATH)
    # copy_uw_recon_vols(UW_HARD_RECON_PATH, HARD_RECONS_PATH,
    #                    src_file_suffix['hard1'])
    # copy_uw_recon_vols(UW_SOFT_RECON_PATH, SOFT_RECONS_PATH,
    #                    src_file_suffix['soft1'])
    # copy_uw_recon_vols(UW_HARD_RECON_PATH, HARD_RECON_REG_PATH,
    #                    src_file_suffix['hard2'])
    # copy_uw_recon_vols(UW_SOFT_RECON_PATH, SOFT_RECON_REG_PATH,
    #                    src_file_suffix['soft2'])

    # run_make_target('hard')  # Run this on mlsc
    # run_make_target('soft')  # Run this on mlsc

    # some_function()
    # calculate_dice(MRI_SCANS_SEG_RESAMPLED_PATH,
    #                HARD_RECON_SEG_RESAMPLED_PATH)  # for hard
    # calculate_dice(MRI_SCANS_SEG_RESAMPLED_PATH,
    #                SOFT_RECON_SEG_RESAMPLED_PATH)  # for soft- check with Henry


if __name__ == '__main__':
    main()
