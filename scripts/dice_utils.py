"""Contains code to run my trained models on Henry's reconstructed volumes
"""

import glob
import json
from mmap import ALLOCATIONGRANULARITY
import os
import re
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from nipype.interfaces.freesurfer import MRIConvert

from ext.lab2im import utils
from SynthSeg.evaluate import fast_dice

# TODO: this file is work in progress
plt.rcParams.update({"text.usetex": False, 'font.family': 'sans-serif'})

SYNTHSEG_PRJCT = '/space/calico/1/users/Harsha/SynthSeg'

UW_HARD_RECON_PATH = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard'
UW_SOFT_RECON_PATH = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_soft'
UW_MRI_SCAN_PATH = '/cluster/vive/UW_photo_recon/FLAIR_Scan_Data'

MRI_SCANS_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.mri.scans'
MRI_SCANS_SEG_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.mri.scans.segmentations'
MRI_SCANS_REG_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.mri.scans.registered'
MRI_SCANS_SEG_RESAMPLED_PATH = MRI_SCANS_SEG_PATH + '.resampled'
MRI_SCANS_SEG_REG_PATH = MRI_SCANS_SEG_RESAMPLED_PATH + '.registered'

HARD_RECONS_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.hard.recon'
HARD_RECON_SEG_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.hard.recon.segmentations.jei'
HARD_RECON_SEG_RESAMPLED_PATH = HARD_RECON_SEG_PATH + '.resampled'
# HARD_RECON_REG_PATH = HARD_RECONS_PATH + '.reg'

SOFT_RECONS_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.soft.recon'
SOFT_RECON_SEG_PATH = f'{SYNTHSEG_PRJCT}/results/UW.photos.soft.recon.segmentations.jei'
SOFT_RECON_SEG_RESAMPLED_PATH = SOFT_RECON_SEG_PATH + '.resampled'
# SOFT_RECON_REG_PATH = SOFT_RECONS_PATH + '.reg'

ALL_LABELS = [
    0, 2, 3, 4, 5, 10, 11, 12, 13, 14, 17, 18, 26, 28, 41, 42, 43, 44, 49, 50,
    51, 52, 53, 54, 58, 60
]
IGNORE_LABELS = [0, 5, 14, 26, 28, 44, 58, 60]
LABEL_PAIRS = [(2, 41), (3, 42), (4, 43), (10, 49), (11, 50), (12, 51),
               (13, 52), (17, 53), (18, 54)]


def files_at_path(path_str):
    return sorted(glob.glob(os.path.join(path_str, '*')))


def copy_uw_recon_vols(src_path, dest_path, flag_list):
    """[summary]

    Args:
        src_path ([type]): [description]
        dest_path ([type]): [description]
        flag_list ([type]): [description]

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

        im, aff, header = utils.load_volume(reconstructed_file[0],
                                            im_only=False)

        # If n_channels = 3 convert to 1 channel by averaging
        if im.ndim == 4 and im.shape[-1] == 3:
            im = np.mean(im, axis=-1).astype('int')

        save_path = os.path.join(dest_path, new_file_name)
        utils.save_volume(im, aff, header, save_path)


def copy_uw_mri_scans(src_path, dest_path):
    """Copy MRI Scans from {src_path} to {dest_path}

    Args:
        src_path (Path String):
        dest_path (Path String):

    Notes:
        The 'NP' prefix for files at {src_path} have been
        removed and replaced '_' with '-' for consistency
        across hard and soft reconstruction names
    """
    os.makedirs(dest_path, exist_ok=True)

    folder_list = sorted(os.listdir(os.path.join(UW_HARD_RECON_PATH)))
    subject_list = [
        folder for folder in folder_list if re.search('[0-9]', folder)
    ]

    print('Copying...')
    for subject in subject_list:
        src_scan_file = 'NP' + subject.replace('-', '_') + '.rotated.mgz'
        print(src_scan_file)
        src_scan_file = os.path.join(src_path, src_scan_file)

        dest_scan_file = subject + '.rotated.mgz'
        dst_scan_file = os.path.join(dest_path, dest_scan_file)

        copyfile(src_scan_file, dst_scan_file)


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


def run_make_target(flag):
    os.system(f'make -C {SYNTHSEG_PRJCT} predict-{flag}')


def perform_registration(input_path, reference_path, output_path):
    input_files = files_at_path(input_path)
    reference_files = files_at_path(reference_path)

    os.makedirs(output_path, exist_ok=True)

    print('Creating...')
    for input_file, reference_file in zip(input_files, reference_files):
        id_check(input_file, reference_file)

        _, file_name = os.path.split(input_file)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + '.res' + file_ext
        out_file = os.path.join(output_path, out_file)

        run_mri_convert(input_file, reference_file, out_file)

        # Testing
        _, ref_aff, _ = utils.load_volume(reference_file, im_only=False)
        _, out_aff, _ = utils.load_volume(out_file, im_only=False)

        assert np.allclose(ref_aff, out_aff) == True, "Mismatched Affine"


def id_check(scan_reg, mri_resampled_seg):
    scan_reg_fn = os.path.split(scan_reg)[-1]
    mri_resampled_seg_fn = os.path.split(mri_resampled_seg)[-1]

    assert scan_reg_fn[:7] == mri_resampled_seg_fn[:7], 'File MisMatch'
    print(scan_reg_fn[:7])


def perform_overlay():
    mri_scans_reg = files_at_path(MRI_SCANS_REG_PATH)
    mri_resampled_segs = files_at_path(MRI_SCANS_SEG_RESAMPLED_PATH)

    os.makedirs(MRI_SCANS_SEG_REG_PATH, exist_ok=True)

    print('Creating...')
    for scan_reg, mri_resampled_seg in zip(mri_scans_reg, mri_resampled_segs):
        id_check(scan_reg, mri_resampled_seg)

        _, scan_reg_aff, scan_reg_head = utils.load_volume(scan_reg,
                                                           im_only=False)
        mrs_im = utils.load_volume(mri_resampled_seg)

        _, file_name = os.path.split(mri_resampled_seg)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + '.reg' + file_ext
        out_file = os.path.join(MRI_SCANS_SEG_REG_PATH, out_file)

        # We can now combine the segmentation voxels with the registered header.
        utils.save_volume(mrs_im, scan_reg_aff, scan_reg_head, out_file)

        # this new file should overlay with the 3D photo reconstruction


def calculate_dice(ground_truth_segs_path, estimated_segs_path):
    ground_truths = files_at_path(ground_truth_segs_path)
    estimated_segs = files_at_path(estimated_segs_path)

    final_dice_scores = dict()
    for ground_truth, estimated_seg in zip(ground_truths, estimated_segs):
        id_check(ground_truth, estimated_seg)

        subject_id = os.path.split(ground_truth)[-1][:7]

        ground_truth_vol = utils.load_volume(ground_truth)
        estimated_seg_vol = utils.load_volume(estimated_seg)

        assert ground_truth_vol.shape == estimated_seg_vol.shape, "Shape mismatch"

        required_labels = np.array(list(set(ALL_LABELS) - set(IGNORE_LABELS)))

        dice_coeff = fast_dice(ground_truth_vol, estimated_seg_vol,
                               required_labels)

        common_labels = common_labels.astype('int').tolist()

        final_dice_scores[subject_id] = dict(zip(common_labels, dice_coeff))

    with open('hard_recon_dice.json', 'w', encoding='utf-8') as fp:
        json.dump(final_dice_scores,
                  fp,
                  ensure_ascii=False,
                  sort_keys=True,
                  indent=4)


def hard_recon_box_plot():
    # Load json
    hard_dice_json = os.path.join(SYNTHSEG_PRJCT, 'hard_recon_dice.json')
    with open(hard_dice_json, 'r') as fp:
        hard_dice = json.load(fp)

    dice_pair_dict = dict()
    for label_pair in LABEL_PAIRS:
        dice_pair_dict[label_pair] = []

    for subject in hard_dice:
        print(f'{subject} - {len(hard_dice[subject])}')
        for label_pair in LABEL_PAIRS:
            dice_pair = [
                hard_dice[subject].get(str(label), 0) for label in label_pair
            ]

            if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
                dice_pair_dict[label_pair].append(dice_pair)

    data = []
    for label_pair in dice_pair_dict:
        data.append(np.mean(dice_pair_dict[label_pair], 1))

    plt.rcParams.update({"text.usetex": False})
    pp = PdfPages('hard_dice_boxplot1.pdf')

    # https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist=True, notch='True')

    # colors = ['#0000FF', '#00FF00',
    #         '#FFFF00', '#FF00FF']

    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B', linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='red', linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    # x-axis labels
    ax.set_xticklabels(LABEL_PAIRS, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)

    # Adding title
    plt.title("Measured 3D Surface - Dice Scores", fontsize=20)

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # show plot
    plt.show()

    pp.savefig(fig)
    pp.close()


def combine_pairs(df, pair_list):
    # drop ignore columns
    # df = df.drop(columns=[str(item) for item in IGNORE_LIST])

    for label_pair in pair_list:
        label_pair = tuple(str(item) for item in label_pair)
        df[f'{label_pair}'] = df[label_pair[0]] + df[label_pair[1]]
        df = df.drop(columns=list(label_pair))

    return df


def get_volume_correlations():
    hard_seg_vols_file = os.path.join(SYNTHSEG_PRJCT, 'results',
                                      'UW.photos.hard.recon.volumes.jei.csv')
    soft_seg_vols_file = os.path.join(SYNTHSEG_PRJCT, 'results',
                                      'UW.photos.soft.recon.volumes.jei.csv')
    mri_seg_vols_file = os.path.join(SYNTHSEG_PRJCT, 'results',
                                     'UW.photos.mri.scans.segmentations.csv')

    mri_seg_vols = pd.read_csv(mri_seg_vols_file, header=None)
    hard_seg_vols = pd.read_csv(hard_seg_vols_file, header=0)
    soft_seg_vols = pd.read_csv(soft_seg_vols_file, header=0)

    label_names = mri_seg_vols.iloc[0, 1:].values
    label_idx = mri_seg_vols.iloc[1, 1:].values

    label_dict = list(zip(label_idx, label_names))

    mri_seg_vols = mri_seg_vols.drop([0])
    mri_seg_vols.loc[1, 0] = 'subjects'
    mri_seg_vols.columns = mri_seg_vols.iloc[0]
    mri_seg_vols = mri_seg_vols.drop([1])
    mri_seg_vols = mri_seg_vols.reset_index(drop=True)

    mri_seg_vols = combine_pairs(mri_seg_vols, LABEL_PAIRS)
    hard_seg_vols = combine_pairs(hard_seg_vols, LABEL_PAIRS)
    soft_seg_vols = combine_pairs(soft_seg_vols, LABEL_PAIRS)


if __name__ == '__main__':
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
    # copy_uw_recon_vols(UW_HARD_RECON_PATH, MRI_SCANS_REG_PATH,
    #                    src_file_suffix['hard2'])
    # copy_uw_recon_vols(UW_SOFT_RECON_PATH, SOFT_RECON_REG_PATH,
    #                    src_file_suffix['soft2'])

    # run_make_target('hard')   # Run this on mlsc
    # run_make_target('soft')   # Run this on mlsc
    # run_make_target('scans')  # Run this on mlsc

    # put the synthseg segmentation in the same space as the input
    # perform_registration(MRI_SCANS_SEG_PATH, MRI_SCANS_PATH,
    #                      MRI_SCANS_SEG_RESAMPLED_PATH)

    # perform_overlay()

    # perform_registration(HARD_RECON_SEG_PATH, MRI_SCANS_SEG_REG_PATH,
    #                      HARD_RECON_SEG_RESAMPLED_PATH)

    # calculate_dice(MRI_SCANS_SEG_RESAMPLED_PATH,
    #                HARD_RECON_SEG_RESAMPLED_PATH)  # for hard
    # calculate_dice(MRI_SCANS_SEG_RESAMPLED_PATH,
    #                SOFT_RECON_SEG_RESAMPLED_PATH)  # for soft- check with Henry

    # hard_recon_box_plot()
    # get_volume_correlations()
