import json
import os

import numpy as np
from dice_config import Configuration
from dice_utils import files_at_path, id_check, return_common_subjects

from ext.lab2im import utils
from SynthSeg.evaluate import fast_dice


def calculate_dice_2d(config, folder1, folder2, file_name, merge=0):
    folder1_list, folder2_list = files_at_path(folder1), files_at_path(folder2)

    folder1_list, folder2_list = return_common_subjects(
        folder1_list, folder2_list)

    final_dice_scores = dict()
    for file1, file2 in zip(folder1_list, folder2_list):
        if not id_check(config, file1, file2):
            continue

        subject_id = os.path.split(file1)[-1][:7]

        img1 = utils.load_volume(file1)
        img2 = utils.load_volume(file2)

        assert img2.shape == img2.shape, "Shape Mismatch"

        slice_idx = np.argmax((img1 > 1).sum(0).sum(0))

        x = img1[:, :, slice_idx].astype('int')
        y = img2[:, :, slice_idx].astype('int')

        required_labels = config.required_labels

        if merge:
            x, y, required_labels = merge_labels_in_image(config, x, y)

        dice_coeff = fast_dice(x, y, required_labels)
        required_labels = required_labels.astype('int').tolist()
        final_dice_scores[subject_id] = dict(zip(required_labels, dice_coeff))

    merge_tag = 'merge' if merge else 'no-merge'

    with open(os.path.join(config.SYNTHSEG_RESULTS,
                           f'{file_name}_{merge_tag}.json'),
              'w',
              encoding='utf-8') as fp:
        json.dump(final_dice_scores, fp, sort_keys=True, indent=4)


def merge_labels_in_image(config, x, y):
    merge_required_labels = []
    for (id1, id2) in config.LABEL_PAIRS:
        x[x == id2] = id1
        y[y == id2] = id1

        merge_required_labels.append(id1)

    merge_required_labels = np.array(merge_required_labels)

    return x, y, merge_required_labels


def calculate_dice_for_slices(config):
    # ### Work for Hard segmentations
    print('Printing 2D Hard Dices')
    print('Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space')
    calculate_dice_2d(config, config.HARD_MANUAL_LABELS_MERGED,
                      config.HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
                      'hard_manual_vs_hard_synth_in_sam_space')
    calculate_dice_2d(config, config.HARD_MANUAL_LABELS_MERGED,
                      config.HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
                      'hard_manual_vs_hard_synth_in_sam_space', 1)

    print('Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space')
    calculate_dice_2d(config, config.HARD_MANUAL_LABELS_MERGED,
                      config.HARD_RECON_SAMSEG,
                      'hard_manual_vs_hard_sam_in_sam_space')
    calculate_dice_2d(config, config.HARD_MANUAL_LABELS_MERGED,
                      config.HARD_RECON_SAMSEG,
                      'hard_manual_vs_hard_sam_in_sam_space', 1)

    # # ### Work for Soft segmentations
    print('Printing 2D Soft Dices')
    print('Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space')
    calculate_dice_2d(config, config.SOFT_MANUAL_LABELS_MERGED,
                      config.SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
                      'soft_manual_vs_soft_synth_in_sam_space')
    calculate_dice_2d(config, config.SOFT_MANUAL_LABELS_MERGED,
                      config.SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
                      'soft_manual_vs_soft_synth_in_sam_space', 1)

    print('Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space')
    calculate_dice_2d(config, config.SOFT_MANUAL_LABELS_MERGED,
                      config.SOFT_RECON_SAMSEG,
                      'soft_manual_vs_soft_sam_in_sam_space')
    calculate_dice_2d(config, config.SOFT_MANUAL_LABELS_MERGED,
                      config.SOFT_RECON_SAMSEG,
                      'soft_manual_vs_soft_sam_in_sam_space', 1)


if __name__ == '__main__':
    config = Configuration()
    calculate_dice_for_slices(config)
