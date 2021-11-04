"""Contains code to run my trained models on Henry's reconstructed volumes
"""

import glob
import json
import os
import re
import sys
from shutil import copyfile
import math
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ext.lab2im import utils
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from nipype.interfaces.freesurfer import MRIConvert
from scipy.stats.stats import pearsonr
from SynthSeg.evaluate import fast_dice

from fs_lut import fs_lut

rcParams.update({'figure.autolayout': True})

sns.set(style="whitegrid", rc={'text.usetex': True})

# TODO: this file is work in progress
plt.rcParams.update({"text.usetex": True, 'font.family': 'sans-serif'})

LUT, REVERSE_LUT = fs_lut()

SYNTHSEG_PRJCT = '/space/calico/1/users/Harsha/SynthSeg'
SYNTHSEG_RESULTS = f'{SYNTHSEG_PRJCT}/results'

UW_HARD_RECON = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard'
UW_SOFT_RECON = '/cluster/vive/UW_photo_recon/recons/results_Henry/Results_soft'
UW_MRI_SCAN = '/cluster/vive/UW_photo_recon/FLAIR_Scan_Data'

MRI_SCANS = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans'
MRI_SCANS_SEG = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans.segmentations'
MRI_SCANS_REG = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans.registered'
MRI_SCANS_SEG_RESAMPLED = MRI_SCANS_SEG + '.resampled'
MRI_SCANS_SEG_REG_RES = MRI_SCANS_SEG_RESAMPLED + '.registered'

HARD_RECONS = f'{SYNTHSEG_RESULTS}/UW.photos.hard.recon'
HARD_RECON_SYNTHSEG = f'{SYNTHSEG_RESULTS}/UW.photos.hard.recon.segmentations.jei'
HARD_RECON_SAMSEG = f'{SYNTHSEG_RESULTS}/UW.photos.hard.samseg.segmentations'
HARD_MANUAL_LABELS_MERGED = f'{SYNTHSEG_RESULTS}/UW.photos.hard.manual.labels'

SOFT_RECONS = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon'
SOFT_RECON_REG = SOFT_RECONS + '.registered'
SOFT_RECON_SYNTHSEG = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon.segmentations.jei'
SOFT_RECON_SAMSEG = f'{SYNTHSEG_RESULTS}/UW.photos.soft.samseg.segmentations'
SOFT_MANUAL_LABELS_MERGED = f'{SYNTHSEG_RESULTS}/UW.photos.soft.manual.labels'

# Note: All of these are in photo RAS space (just resampling based on reference)
MRI_SYNTHSEG_IN_SAMSEG_SPACE = MRI_SCANS_SEG_REG_RES + '.in_samseg_space'
MRI_SYNTHSEG_IN_SOFTSAMSEG_SPACE = MRI_SCANS_SEG_REG_RES + '.in_softsamseg_space'
HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE = HARD_RECON_SYNTHSEG + '.in_samseg_space'
HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE = HARD_RECON_SYNTHSEG + '.in_mri_space'
SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE = SOFT_RECON_SYNTHSEG + '.in_samseg_space'

mri_synthseg_vols_file = f'{SYNTHSEG_RESULTS}/UW.photos.mri.scans.segmentations.csv'
soft_synthseg_vols_file = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon.segmentations.jei.csv'
hard_synthseg_vols_file = f'{SYNTHSEG_RESULTS}/UW.photos.hard.recon.segmentations.jei.csv'

#### Extract SAMSEG Volumes
HARD_SAMSEG_STATS = f'{UW_HARD_RECON}/SAMSEG/'
SOFT_SAMSEG_STATS = f'{UW_SOFT_RECON}/SAMSEG/'

ALL_LABELS = [
    0, 2, 3, 4, 5, 10, 11, 12, 13, 14, 17, 18, 26, 28, 41, 42, 43, 44, 49, 50,
    51, 52, 53, 54, 58, 60
]
IGNORE_LABELS = [0, 5, 14, 26, 28, 44, 58, 60]
ADDL_IGNORE_LABELS = [7, 8, 15, 16, 46, 47]
LABEL_PAIRS = [(2, 41), (3, 42), (4, 43), (10, 49), (11, 50), (12, 51),
               (13, 52), (17, 53), (18, 54)]
LABEL_PAIR_NAMES = [
    'White Matter', 'Cortex', 'Ventricle', 'Thalamus', 'Caudate', 'Putamen',
    'Pallidum', 'Hippocampus', 'Amygdala'
]
IGNORE_SUBJECTS = ['18-1343', '18-2260', '19-0019']


def files_at_path(path_str):
    return sorted(glob.glob(os.path.join(path_str, '*')))


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

        # Testing
        # _, ref_aff, _ = utils.load_volume(reference_file, im_only=False)
        # _, out_aff, _ = utils.load_volume(out_file, im_only=False)

        # assert np.allclose(ref_aff, out_aff) == True, "Mismatched Affine"


def id_check(scan_reg, mri_resampled_seg):
    scan_reg_fn = os.path.split(scan_reg)[-1]
    mri_resampled_seg_fn = os.path.split(mri_resampled_seg)[-1]

    assert scan_reg_fn[:7] == mri_resampled_seg_fn[:7], 'File MisMatch'

    if scan_reg_fn[:7] in IGNORE_SUBJECTS:
        return 0
    else:
        print(scan_reg_fn[:7])
        return 1


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


def print_correlations(x, y, file_name=None):
    if file_name is None:
        raise Exception('Please enter a file name to print correlations')
    col_names = x.columns

    corr_dict = dict()
    for col_name in col_names:
        corr_dict[col_name] = pearsonr(x[col_name], y[col_name])[0]

    with open(os.path.join(SYNTHSEG_RESULTS, file_name), 'w',
              encoding='utf-8') as fp:
        json.dump(corr_dict, fp, sort_keys=True, indent=4)


def extract_synthseg_vols(file_name, flag):
    skiprows = 1 if flag == 'mri' else None
    df = pd.read_csv(file_name, skiprows=skiprows, header=0)

    if flag == 'mri':
        df = df.rename(columns={'Unnamed: 0': 'subjects'})

    df['subjects'] = df['subjects'].str.slice(0, 7)
    df = df.set_index('subjects')

    df.index.name = None

    df = combine_pairs(df, LABEL_PAIRS)
    df = df.drop(
        columns=[column for column in df.columns if '(' not in column])
    df = df.drop(labels=IGNORE_SUBJECTS)

    return df


def plot_correlation_pairs(x, y, z, file_name=None, flag=None):
    common_labels = x.index.intersection(y.index).intersection(z.index)
    x = x.loc[common_labels]
    y = y.loc[common_labels]
    z = z.loc[common_labels]

    col_names = x.columns


def fisherZ(r):
	return (.5*math.log((1.0+r)/(1.0-r)))

def calculate_pval(r12, r13, r23, n):

	z12 = fisherZ(r12)
	z13 = fisherZ(r13)
	z23 = fisherZ(r23)

	r1sq = ((r12 + r13)/2.0)*((r12 + r13)/2.0)
	variance = (1.0 / ((1-r1sq)*(1-r1sq)))  * (r23*(1.0-2.0*r1sq)-.5*r1sq*(1-2.0*r1sq-(r23*r23)))
	variance2 = np.sqrt((2.0-2.0*variance)/(n-3.0))
			
	p = (z12 - z13)/variance2
	alpha = norm.sf(p)

	return p, alpha


def print_correlation_pairs(x, y, z, file_name=None, flag=None):
    common_labels = x.index.intersection(y.index).intersection(z.index)
    x = x.loc[common_labels]
    y = y.loc[common_labels]
    z = z.loc[common_labels]

    col_names = x.columns

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(os.path.join(SYNTHSEG_RESULTS, 'volume_correlations'),
              'a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        print(f'{flag} RECONSTRUCTIONS')
        print('{:^15}{:^15}{:^15}{:^15}'.format('label', 'SAMSEG', 'SYNTHSEG', 'p-value'))
        print('=' * 65)
        print('CORRELATIONS')
        print('=' * 65)
        for col_name, name in zip(col_names, LABEL_PAIR_NAMES):
            a = pearsonr(x[col_name], y[col_name])[0]
            b = pearsonr(x[col_name], z[col_name])[0]
            k = pearsonr(y[col_name], z[col_name])[0]
            _, alpha = calculate_pval(b, a, k, len(x[col_name]))

            print(f'{name:^15}{a:^15.3f}{b:^15.3f}{alpha:^15.6f}')
        print('=' * 65)

        print('MEAN ABSOLUTE RESIDUALS')
        print('=' * 45)
        for col_name, name in zip(col_names, LABEL_PAIR_NAMES):
            a = np.mean(np.abs(x[col_name] - y[col_name]) / x[col_name]) * 100
            b = np.mean(np.abs(x[col_name] - z[col_name]) / x[col_name]) * 100

            print(f'{name:^15}{a:^15.3f}{b:^15.3f}')
        print('=' * 45)

        print('MEAN RESIDUALS')
        print('=' * 45)
        for col_name, name in zip(col_names, LABEL_PAIR_NAMES):
            a = np.mean((x[col_name] - y[col_name]) / x[col_name]) * 100
            b = np.mean((x[col_name] - z[col_name]) / x[col_name]) * 100

            print(f'{name:^15}{a:^15.3f}{b:^15.3f}')
        print('=' * 45)
        print()
        sys.stdout = original_stdout  # Reset the standard output to its original value

    return


def combine_pairs(df, pair_list):
    for label_pair in pair_list:
        label_pair = tuple(str(item) for item in label_pair)
        df[f'{label_pair}'] = df[label_pair[0]] + df[label_pair[1]]
        df = df.drop(columns=list(label_pair))

    return df


def extract_samseg_volumes(folder_path, flag):
    df_list = []

    hard_folder_list = sorted(glob.glob(os.path.join(folder_path, '*')))

    for folder in hard_folder_list:
        _, folder_name = os.path.split(folder)

        if flag == 'hard':
            subject_id = folder_name.split('.')[0]
        elif flag == 'soft':
            subject_id = folder_name.split('_')[0]
        else:
            raise Exception('Incorrect Flag')

        if subject_id in IGNORE_SUBJECTS:
            continue

        df = pd.read_csv(os.path.join(folder, 'samseg.stats'),
                         header=None,
                         names=['label', 'volume', 'units'])

        # drop column 'units'
        df = df.drop(columns=['units'])

        # remove '# measure' from 'label' column
        df['label'] = df['label'].str.replace(r'# Measure ', '')

        # map 'label' to 'idx'
        df['idx'] = df['label'].map(REVERSE_LUT)

        # drop 'label' column
        df = df.drop(columns=['label'])

        # drop 'nan' rows
        df = df[df['idx'].notna()]

        # make 'idx' the new index
        df = df.set_index('idx').sort_index()

        df = df.rename(columns={'volume': subject_id})

        df.index.name = None

        df_list.append(df)

    df1 = pd.concat(df_list, axis=1)
    df2 = df1.T

    df2 = combine_pairs(df2, LABEL_PAIRS)
    hard_samseg_df = df2.drop(
        columns=[column for column in df2.columns if '(' not in column])

    return hard_samseg_df


def print_correlations(x, y, file_name=None):
    if file_name is None:
        raise Exception('Please enter a file name to print correlations')
    col_names = x.columns

    corr_dict = dict()
    for col_name in col_names:
        corr_dict[col_name] = pearsonr(x[col_name], y[col_name])[0]

    with open(os.path.join(SYNTHSEG_RESULTS, file_name), 'w',
              encoding='utf-8') as fp:
        json.dump(corr_dict, fp, sort_keys=True, indent=4)


def extract_scores(in_file_name, merge=0):
    # Load json
    hard_dice_json = os.path.join(SYNTHSEG_RESULTS, in_file_name)
    with open(hard_dice_json, 'r') as fp:
        hard_dice = json.load(fp)

    if merge:
        dice_pair_dict = dict()
        for label_idx1, label_idx2 in LABEL_PAIRS:
            dice_pair_dict[label_idx1] = []

        for subject in hard_dice:
            for label_idx1, _ in LABEL_PAIRS:
                dice_pair = hard_dice[subject].get(str(label_idx1), 0)

                # if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
                dice_pair_dict[label_idx1].append(dice_pair)

        data = []
        for label_idx in dice_pair_dict:
            data.append(dice_pair_dict[label_idx])
    else:
        dice_pair_dict = dict()
        for label_pair in LABEL_PAIRS:
            dice_pair_dict[label_pair] = []

        for subject in hard_dice:
            for label_pair in LABEL_PAIRS:
                dice_pair = [
                    hard_dice[subject].get(str(label), 0)
                    for label in label_pair
                ]

                # if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
                dice_pair_dict[label_pair].append(dice_pair)

        data = []
        for label_pair in dice_pair_dict:
            data.append(np.mean(dice_pair_dict[label_pair], 1))

    return data


def create_single_dataframe(data1, data2):
    ha1 = pd.DataFrame(data1, index=LABEL_PAIRS)
    ha2 = pd.DataFrame(data2, index=LABEL_PAIRS)

    ha1 = ha1.stack().reset_index()
    ha1 = ha1.rename(
        columns=dict(zip(ha1.columns, ['struct', 'subject', 'score'])))
    ha1['type'] = 'samseg'

    ha2 = ha2.stack().reset_index()
    ha2 = ha2.rename(
        columns=dict(zip(ha2.columns, ['struct', 'subject', 'score'])))
    ha2['type'] = 'synthseg'

    ha = pd.concat([ha1, ha2], axis=0, ignore_index=True)

    return ha


def dice_plot_from_df(df, out_file_name, flag):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax = sns.boxplot(x="struct",
                     y="score",
                     hue="type",
                     data=df,
                     palette="Set3")
    ax.set_xlim(-0.5, 8.49)
    ax.set_ylim(-0.025, 1.025)
    [
        ax.axvline(x + .5, color='k', linestyle=':', lw=0.5)
        for x in ax.get_xticks()
    ]
    [i.set_linewidth(1) for i in ax.spines.values()]
    [i.set_edgecolor('k') for i in ax.spines.values()]

    # Adding title
    plt.title(f"2D Dice Scores (For {flag} reconstruction)", fontsize=20)
    plt.yticks(fontsize=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(LABEL_PAIR_NAMES, fontsize=15, rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(handles=handles, labels=labels, fontsize=20, frameon=False)

    plt.savefig(os.path.join(SYNTHSEG_RESULTS, out_file_name))


def merge_labels_in_image(x, y):
    merge_required_labels = []
    for (id1, id2) in LABEL_PAIRS:
        x[x == id2] = id1
        y[y == id2] = id1

        merge_required_labels.append(id1)

    merge_required_labels = np.array(merge_required_labels)

    return x, y, merge_required_labels


def calculate_dice_2d(folder1, folder2, file_name, merge=0):
    folder1_list, folder2_list = files_at_path(folder1), files_at_path(folder2)

    folder1_list, folder2_list = return_common_subjects(
        folder1_list, folder2_list)

    final_dice_scores = dict()
    for file1, file2 in zip(folder1_list, folder2_list):
        if not id_check(file1, file2):
            continue

        subject_id = os.path.split(file1)[-1][:7]

        img1 = utils.load_volume(file1)
        img2 = utils.load_volume(file2)

        assert img2.shape == img2.shape, "Shape Mismatch"

        slice_idx = np.argmax((img1 > 1).sum(0).sum(0))

        x = img1[:, :, slice_idx].astype('int')
        y = img2[:, :, slice_idx].astype('int')

        required_labels = np.array(list(set(ALL_LABELS) - set(IGNORE_LABELS)))

        if merge:
            x, y, required_labels = merge_labels_in_image(x, y)

        dice_coeff = fast_dice(x, y, required_labels)
        required_labels = required_labels.astype('int').tolist()
        final_dice_scores[subject_id] = dict(zip(required_labels, dice_coeff))

    merge_tag = 'merge' if merge else 'no-merge'

    with open(os.path.join(SYNTHSEG_RESULTS, f'{file_name}_{merge_tag}.json'),
              'w',
              encoding='utf-8') as fp:
        json.dump(final_dice_scores, fp, sort_keys=True, indent=4)


def construct_dice_plots_from_files(file1, file2, merge_flag, hard_or_soft,
                                    out_name):
    data1 = extract_scores(file1, merge_flag)
    data2 = extract_scores(file2, merge_flag)

    df = create_single_dataframe(data1, data2)
    dice_plot_from_df(df, out_name, hard_or_soft)


if __name__ == '__main__':
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

    # # print('\nCopying MRI Scans')
    # # copy_uw_mri_scans(UW_MRI_SCAN, MRI_SCANS)

    # # print('\nCopying Hard Reconstructions')
    # # copy_uw_recon_vols(UW_HARD_RECON, HARD_RECONS, src_file_suffix['hard1'])

    # # print('\nCopying Soft Reconstructions')
    # # copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECONS, src_file_suffix['soft1'])

    # # print('\nCopying Registered (to hard) MRI Scans')
    # # copy_uw_recon_vols(UW_HARD_RECON, MRI_SCANS_REG, src_file_suffix['hard2'])

    # # print('\nCopying I really dont know what this is')
    # # copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECON_REG, src_file_suffix['soft2'])

    # # print('\nCopying Hard Manual Labels')
    # # copy_uw_recon_vols(UW_HARD_RECON, HARD_MANUAL_LABELS_MERGED,
    # #                    src_file_suffix['hard4'])

    # # print('\nCopying Soft Manual Labels')
    # # copy_uw_recon_vols(UW_SOFT_RECON, SOFT_MANUAL_LABELS_MERGED,
    # #                    src_file_suffix['soft4'])

    # # run_make_target('hard')  # Run this on mlsc
    # # run_make_target('soft')  # Run this on mlsc
    # # run_make_target('scans')  # Run this on mlsc

    # # print('\nCopying SAMSEG Segmentations (Hard)')
    # # copy_uw_recon_vols(UW_HARD_RECON, HARD_RECON_SAMSEG,
    # #                    src_file_suffix['hard3'])

    # # print('\nCopying SAMSEG Segmentations (Soft)')
    # # copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECON_SAMSEG,
    # #                    src_file_suffix['soft3'])

    # # print('\nPut MRI SynthSeg Segmentation in the same space as MRI')
    # # perform_registration(MRI_SCANS_SEG, MRI_SCANS, MRI_SCANS_SEG_RESAMPLED)

    # # print('\nCombining MRI_Seg Volume and MRI_Vol Header')
    # # perform_overlay()

    # print('3D Hard')
    # print('\nDice(MRI_Seg, PhotoReconSAMSEG) in PhotoReconSAMSEG space')
    # perform_registration(MRI_SCANS_SEG_REG_RES, HARD_RECON_SAMSEG,
    #                      MRI_SYNTHSEG_IN_SAMSEG_SPACE)
    # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE, HARD_RECON_SAMSEG,
    #                'mri_synth_vs_hard_samseg_in_sam_space.json')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(HARD_RECON_SYNTHSEG, HARD_RECON_SAMSEG,
    #                      HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE)
    # # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE,
    # #                HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    # #                'mri_synth_vs_hard_synth_in_sam_space.json')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(HARD_RECON_SYNTHSEG, MRI_SCANS_SEG_REG_RES,
    #                      HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE)
    # # calculate_dice(MRI_SCANS_SEG_REG_RES, HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE,
    # #                'mri_synth_vs_hard_synth_in_mri_space.json')

    # print('3D Soft')
    # print('\nDice(MRI_Seg, PhotoReconSAMSEG) in PhotoReconSAMSEG space')
    # perform_registration(MRI_SCANS_SEG_REG_RES, SOFT_RECON_SAMSEG,
    #                      MRI_SYNTHSEG_IN_SAMSEG_SPACE)
    # # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE, SOFT_RECON_SAMSEG,
    # #                'mri_synth_vs_soft_samseg_in_sam_space.json')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(SOFT_RECON_SYNTHSEG, SOFT_RECON_SAMSEG,
    #                      SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE)
    # # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE,
    # #                HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    # #                'mri_synth_vs_soft_synth_in_sam_space.json')


    print('Extracting SYNTHSEG Volumes')
    mri_synthseg_vols = extract_synthseg_vols(mri_synthseg_vols_file, 'mri')
    hard_synthseg_vols = extract_synthseg_vols(hard_synthseg_vols_file, 'hard')
    soft_synthseg_vols = extract_synthseg_vols(soft_synthseg_vols_file, 'soft')

    print('Extracting SAMSEG Volumes')
    hard_samseg_vols = extract_samseg_volumes(HARD_SAMSEG_STATS, 'hard')
    soft_samseg_vols = extract_samseg_volumes(SOFT_SAMSEG_STATS, 'soft')

    print('Writing Correlations to File')
    print_correlation_pairs(mri_synthseg_vols,
                            hard_samseg_vols,
                            hard_synthseg_vols,
                            flag='HARD')
    # plot_correlation_pairs(mri_synthseg_vols,
    #                         hard_samseg_vols,
    #                         hard_synthseg_vols,
    #                         flag='HARD')

    print_correlation_pairs(mri_synthseg_vols,
                            soft_samseg_vols,
                            soft_synthseg_vols,
                            flag='SOFT')
    # plot_correlation_pairs(mri_synthseg_vols,
    #                         soft_samseg_vols,
    #                         soft_synthseg_vols,
    #                         flag='SOFT')

    # # ### Work for Hard segmentations
    # print('Printing 2D Hard Dices')
    # print('Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space')
    # calculate_dice_2d(HARD_MANUAL_LABELS_MERGED,
    #                   HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    #                   'hard_manual_vs_hard_synth_in_sam_space')
    # calculate_dice_2d(HARD_MANUAL_LABELS_MERGED,
    #                   HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    #                   'hard_manual_vs_hard_synth_in_sam_space', 1)

    # print('Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space')
    # calculate_dice_2d(HARD_MANUAL_LABELS_MERGED, HARD_RECON_SAMSEG,
    #                   'hard_manual_vs_hard_sam_in_sam_space')
    # calculate_dice_2d(HARD_MANUAL_LABELS_MERGED, HARD_RECON_SAMSEG,
    #                   'hard_manual_vs_hard_sam_in_sam_space', 1)

    # construct_dice_plots_from_files(
    #     'hard_manual_vs_hard_sam_in_sam_space_no-merge.json',
    #     'hard_manual_vs_hard_synth_in_sam_space_no-merge.json', 0, 'hard',
    #     'hard_reconstruction_no-merge.png')

    # construct_dice_plots_from_files(
    #     'hard_manual_vs_hard_sam_in_sam_space_merge.json',
    #     'hard_manual_vs_hard_synth_in_sam_space_merge.json', 1, 'hard',
    #     'hard_reconstruction_merge.png')

    # # # ### Work for Soft segmentations
    # print('Printing 2D Soft Dices')
    # print('Dice_2D(PhotoManualLabel, PhotoSynthSeg) in PhotoSAMSEG space')
    # calculate_dice_2d(SOFT_MANUAL_LABELS_MERGED,
    #                   SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    #                   'soft_manual_vs_soft_synth_in_sam_space')
    # calculate_dice_2d(SOFT_MANUAL_LABELS_MERGED,
    #                   SOFT_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    #                   'soft_manual_vs_soft_synth_in_sam_space', 1)

    # print('Dice_2D(PhotoManualLabel, PhotoSamSeg) in PhotoSAMSEG space')
    # calculate_dice_2d(SOFT_MANUAL_LABELS_MERGED, SOFT_RECON_SAMSEG,
    #                   'soft_manual_vs_soft_sam_in_sam_space')
    # calculate_dice_2d(SOFT_MANUAL_LABELS_MERGED, SOFT_RECON_SAMSEG,
    #                   'soft_manual_vs_soft_sam_in_sam_space', 1)

    # construct_dice_plots_from_files(
    #     'soft_manual_vs_soft_sam_in_sam_space_no-merge.json',
    #     'soft_manual_vs_soft_synth_in_sam_space_no-merge.json', 0, 'soft',
    #     'soft_reconstruction_no-merge.png')

    # construct_dice_plots_from_files(
    #     'soft_manual_vs_soft_sam_in_sam_space_merge.json',
    #     'soft_manual_vs_soft_synth_in_sam_space_merge.json', 1, 'soft',
    #     'soft_reconstruction_merge.png')
