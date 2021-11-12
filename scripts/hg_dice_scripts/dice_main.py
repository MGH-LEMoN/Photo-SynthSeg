"""Contains code to run my trained models on Henry's reconstructed volumes
"""

import json
import os

import numpy as np
from dice_config import *
from dice_gather import copy_relevant_files, files_at_path
from dice_mri_utils import perform_overlay, run_mri_convert
from dice_plots import write_plots
from dice_utils import id_check, return_common_subjects
from dice_volumes import write_correlations_to_file

from ext.lab2im import utils
from SynthSeg.evaluate import fast_dice


def run_make_target(flag):
    os.system(f'make -C {SYNTHSEG_PRJCT} predict-{flag}')


def perform_registration(input_path, reference_path, output_path):
    input_files = files_at_path(input_path)
    reference_files = files_at_path(reference_path)

    input_files, reference_files = return_common_subjects(
        input_files, reference_files)

    os.makedirs(output_path, exist_ok=True)

    print('Creating...')
    for input_file, reference_file in zip(input_files, reference_files):
        id_check(input_file, reference_file)

        _, file_name = os.path.split(input_file)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + '.res' + file_ext
        out_file = os.path.join(output_path, out_file)

        run_mri_convert(input_file, reference_file, out_file)


def calculate_dice(ground_truth_segs_path, estimated_segs_path, file_name):
    ground_truths = files_at_path(ground_truth_segs_path)
    estimated_segs = files_at_path(estimated_segs_path)

    ground_truths, estimated_segs = return_common_subjects(
        ground_truths, estimated_segs)

    final_dice_scores = dict()
    for ground_truth, estimated_seg in zip(ground_truths, estimated_segs):
        if not id_check(ground_truth, estimated_seg):
            continue

        subject_id = os.path.split(ground_truth)[-1][:7]

        ground_truth_vol = utils.load_volume(ground_truth)
        estimated_seg_vol = utils.load_volume(estimated_seg)

        assert ground_truth_vol.shape == estimated_seg_vol.shape, "Shape mismatch"

        required_labels = np.array(list(set(ALL_LABELS) - set(IGNORE_LABELS)))

        dice_coeff = fast_dice(ground_truth_vol, estimated_seg_vol,
                               required_labels)

        required_labels = required_labels.astype('int').tolist()

        final_dice_scores[subject_id] = dict(zip(required_labels, dice_coeff))

    with open(os.path.join(SYNTHSEG_RESULTS, file_name), 'w',
              encoding='utf-8') as fp:
        json.dump(final_dice_scores, fp, sort_keys=True, indent=4)


def combine_pairs(df, pair_list):
    for label_pair in pair_list:
        label_pair = tuple(str(item) for item in label_pair)
        df[f'{label_pair}'] = df[label_pair[0]] + df[label_pair[1]]
        df = df.drop(columns=list(label_pair))

    return df


if __name__ == '__main__':

    # copy_relevant_files()

    # run_make_target('hard')  # Run this on mlsc
    # run_make_target('soft')  # Run this on mlsc
    # run_make_target('scans')  # Run this on mlsc

    # print('\nPut MRI SynthSeg Segmentation in the same space as MRI')
    # perform_registration(MRI_SCANS_SEG, MRI_SCANS, MRI_SCANS_SEG_RESAMPLED)

    # print('\nCombining MRI_Seg Volume and MRI_Vol Header')
    # perform_overlay()

    # print('3D Hard')
    # print('\nDice(MRI_Seg, PhotoReconSAMSEG) in PhotoReconSAMSEG space')
    # perform_registration(MRI_SCANS_SEG_REG_RES, HARD_RECON_SAMSEG,
    #                      MRI_SYNTHSEG_IN_SAMSEG_SPACE)
    # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE, HARD_RECON_SAMSEG,
    #                'mri_synth_vs_hard_samseg_in_sam_space.json')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(HARD_RECON_SYNTHSEG, HARD_RECON_SAMSEG,
    #                      HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE)
    # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE,
    #                HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    #                'mri_synth_vs_hard_synth_in_sam_space.json')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(HARD_RECON_SYNTHSEG, MRI_SCANS_SEG_REG_RES,
    #                      HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE)
    # calculate_dice(MRI_SCANS_SEG_REG_RES, HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE,
    #                'mri_synth_vs_hard_synth_in_mri_space.json')

    # print('3D Soft')
    # print('\nDice(MRI_Seg, PhotoReconSAMSEG) in PhotoReconSAMSEG space')
    # perform_registration(MRI_SCANS_SEG_REG_RES, SOFT_RECON_SAMSEG,
    #                      MRI_SYNTHSEG_IN_SAMSEG_SPACE)
    # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE, SOFT_RECON_SAMSEG,
    #                'mri_synth_vs_soft_samseg_in_sam_space.json')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(SOFT_RECON_SYNTHSEG, SOFT_RECON_SAMSEG,
    #                      SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE)
    # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE,
    #                HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    #                'mri_synth_vs_soft_synth_in_sam_space.json')

    # write_correlations_to_file()

    write_plots()
