"""Contains code to run my trained models on Henry's reconstructed volumes
"""

import glob
import json
import os
import re
import sys
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ext.lab2im import utils
from matplotlib.backends.backend_pdf import PdfPages
from nipype.interfaces.freesurfer import MRIConvert
from scipy.stats.stats import pearsonr
from SynthSeg.evaluate import fast_dice
from fs_lut import fs_lut

# TODO: this file is work in progress
plt.rcParams.update({"text.usetex": False, 'font.family': 'sans-serif'})

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
HARD_RECON_SYNTHSEG_RESAMPLED = HARD_RECON_SYNTHSEG + '.resampled'
HARD_RECON_SAMSEG = f'{SYNTHSEG_RESULTS}/UW.photos.hard.samseg.segmentations'
HARD_RECON_SAMSEG_RESAMPLED = HARD_RECON_SAMSEG + '.resampled'

SOFT_RECONS = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon'
SOFT_RECON_REG = SOFT_RECONS + '.registered'
SOFT_RECON_SYNTHSEG = f'{SYNTHSEG_RESULTS}/UW.photos.soft.recon.segmentations.jei'
SOFT_RECON_SYNTHSEG_RESAMPLED = SOFT_RECON_SYNTHSEG + '.resampled'
SOFT_RECON_SAMSEG = f'{SYNTHSEG_RESULTS}/UW.photos.soft.samseg.segmentations'
SOFT_RECON_SAMSEG_RESAMPLED = SOFT_RECON_SAMSEG + '.resampled'

# Note: All of these are in photo RAS space (just resampling based on reference)
MRI_SYNTHSEG_IN_SAMSEG_SPACE = MRI_SCANS_SEG_REG_RES + '.in_samseg_space'
HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE = HARD_RECON_SYNTHSEG + '.in_samseg_space'
HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE = HARD_RECON_SYNTHSEG + '.in_mri_space'

mri_synthseg_vols_file = os.path.join(SYNTHSEG_RESULTS,
                                    'UW.photos.mri.scans.segmentations.csv')
soft_synthseg_vols_file = os.path.join(SYNTHSEG_RESULTS,
                                'UW.photos.soft.recon.volumes.jei.csv')
hard_synthseg_vols_file = os.path.join(SYNTHSEG_RESULTS,
                                'UW.photos.hard.recon.volumes.jei.csv')
                                
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


def hard_recon_box_plot(in_file_name, out_file_name):
    # Load json
    hard_dice_json = os.path.join(SYNTHSEG_RESULTS, in_file_name)
    with open(hard_dice_json, 'r') as fp:
        hard_dice = json.load(fp)

    dice_pair_dict = dict()
    for label_pair in LABEL_PAIRS:
        dice_pair_dict[label_pair] = []

    for subject in hard_dice:
        for label_pair in LABEL_PAIRS:
            dice_pair = [
                hard_dice[subject].get(str(label), 0) for label in label_pair
            ]

            # if np.all(dice_pair):  # Remove (0, x)/(x, 0)/(0, 0)
            dice_pair_dict[label_pair].append(dice_pair)

    data = []
    for label_pair in dice_pair_dict:
        data.append(np.mean(dice_pair_dict[label_pair], 1))

    plt.rcParams.update({"text.usetex": False})
    pp = PdfPages(os.path.join(SYNTHSEG_RESULTS, out_file_name))

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
    # plt.show()

    pp.savefig(fig)
    pp.close()


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


def get_volume_correlations(flag):
    hard_seg_vols_file = os.path.join(SYNTHSEG_RESULTS,
                                      'UW.photos.hard.recon.volumes.jei.csv')
    soft_seg_vols_file = os.path.join(SYNTHSEG_RESULTS,
                                      'UW.photos.soft.recon.volumes.jei.csv')
    mri_seg_vols_file = os.path.join(SYNTHSEG_RESULTS,
                                     'UW.photos.mri.scans.segmentations.csv')

    mri_seg_vols = pd.read_csv(mri_seg_vols_file, header=None)
    hard_seg_vols = pd.read_csv(hard_seg_vols_file, header=0)
    soft_seg_vols = pd.read_csv(soft_seg_vols_file, header=0)

    mri_seg_vols = mri_seg_vols.drop([0])
    mri_seg_vols.loc[1, 0] = 'subjects'
    mri_seg_vols.columns = mri_seg_vols.iloc[0]
    mri_seg_vols = mri_seg_vols.drop([1])
    mri_seg_vols = mri_seg_vols.reset_index(drop=True)

    drop_cols = [str(label) for label in IGNORE_LABELS
                 ] + ADDL_IGNORE_LABELS + ['subjects']
    mri_seg_vols = mri_seg_vols.drop(columns=drop_cols, errors='ignore')
    hard_seg_vols = hard_seg_vols.drop(columns=drop_cols, errors='ignore')
    soft_seg_vols = soft_seg_vols.drop(columns=drop_cols, errors='ignore')

    mri_seg_vols = mri_seg_vols.astype('float32')
    hard_seg_vols = hard_seg_vols.astype('float32')
    soft_seg_vols = soft_seg_vols.astype('float32')

    mri_seg_vols = combine_pairs(mri_seg_vols, LABEL_PAIRS)
    hard_seg_vols = combine_pairs(hard_seg_vols, LABEL_PAIRS)
    soft_seg_vols = combine_pairs(soft_seg_vols, LABEL_PAIRS)

    print('Saving Hard Reconstruction Correlations')
    print_correlations(mri_seg_vols, hard_seg_vols,
                       f'{flag}_hard_recon_correlations.json')
    print('Saving Soft Reconstruction Correlations')
    print_correlations(mri_seg_vols, soft_seg_vols,
                       f'{flag}_soft_recon_correlations.json')


def extract_synthseg_vols(file_name, flag):
    skiprows = 1 if flag == 'mri' else None
    df = pd.read_csv(file_name, skiprows=skiprows, header=0)
    
    if flag == 'mri':
        df = df.rename(columns={'Unnamed: 0': 'subjects'})

    df['subjects'] = df['subjects'].str.slice(0, 7)
    df = df.set_index('subjects')
    
    df.index.name = None
    
    df = combine_pairs(df, LABEL_PAIRS)
    df = df.drop(columns=[column for column in df.columns if '(' not in column])
    df = df.drop(labels=IGNORE_SUBJECTS)
    
    return df


def print_correlation_pairs(x, y, z, file_name=None, flag=None):
    common_labels = x.index.intersection(y.index).intersection(z.index)
    x = x.loc[common_labels]
    y = y.loc[common_labels]
    z = z.loc[common_labels]

    col_names = x.columns

    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(os.path.join(SYNTHSEG_RESULTS, 'volume_correlations'), 'a+') as f:
        sys.stdout = f # Change the standard output to the file we created.

        print(f'{flag} RECONSTRUCTIONS')
        print('{:^15}{:^15}{:^15}'.format('label', 'SAMSEG', 'SYNTHSEG'))
        print('='*45)   
        print('CORRELATIONS')
        print('='*45)
        for col_name in col_names:
            a = pearsonr(x[col_name], y[col_name])[0]
            b = pearsonr(x[col_name], z[col_name])[0]
            
            print(f'{col_name:^15}{a:^15.3f}{b:^15.3f}')
        print('='*45)
        
        print('MEAN ABSOLUTE RESIDUALS')
        print('='*45)
        for col_name in col_names:
            a = np.mean(np.abs(x[col_name] - y[col_name]) / x[col_name]) * 100   
            b = np.mean(np.abs(x[col_name] - z[col_name]) / x[col_name]) * 100
            
            print(f'{col_name:^15}{a:^15.3f}{b:^15.3f}')
        print('='*45)
        
        print('MEAN RESIDUALS')
        print('='*45)
        for col_name in col_names:
            a = np.mean((x[col_name] - y[col_name]) / x[col_name]) * 100   
            b = np.mean((x[col_name] - z[col_name]) / x[col_name]) * 100
            
            print(f'{col_name:^15}{a:^15.3f}{b:^15.3f}')
        print('='*45)
        print()
        sys.stdout = original_stdout # Reset the standard output to its original value
        
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

        df = pd.read_csv(os.path.join(folder, 'samseg.stats'), header=None, names = ['label', 'volume', 'units'])

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
    hard_samseg_df = df2.drop(columns=[column for column in df2.columns if '(' not in column])
    
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


if __name__ == '__main__':
    src_file_suffix = {
        'hard1': ['*.hard.recon.mgz'],
        'soft1': ['soft', '*_soft.mgz'],
        'hard2': ['*.hard.warped_ref.mgz'],
        'soft2': ['soft', '*_soft_regatlas.mgz'],
        'hard3': ['*samseg*.mgz'],
        'soft3': ['soft', '*samseg*.mgz']
    }

    # print('\nCopying MRI Scans')
    # copy_uw_mri_scans(UW_MRI_SCAN, MRI_SCANS)

    # print('\nCopying Hard Reconsructions')
    # copy_uw_recon_vols(UW_HARD_RECON, HARD_RECONS, src_file_suffix['hard1'])

    # print('\nCopying Soft Reconstructions')
    # copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECONS, src_file_suffix['soft1'])

    # print('\nCopying Registered (to hard) MRI Scans')
    # copy_uw_recon_vols(UW_HARD_RECON, MRI_SCANS_REG, src_file_suffix['hard2'])

    # print('\nCopying I really dont know what this is')
    # copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECON_REG, src_file_suffix['soft2'])

    # run_make_target('hard')  # Run this on mlsc
    # run_make_target('soft')  # Run this on mlsc
    # run_make_target('scans')  # Run this on mlsc

    # print('\nCopying SAMSEG Segmentations (Hard)')
    # copy_uw_recon_vols(UW_HARD_RECON, HARD_RECON_SAMSEG,
    #                    src_file_suffix['hard3'])

    # print('\nCopying SAMSEG Segmentations (Soft)')
    # copy_uw_recon_vols(UW_SOFT_RECON, SOFT_RECON_SAMSEG,
    #                    src_file_suffix['soft3'])

    # print('\nPut MRI SynthSeg Segmentation in the same space as MRI')
    # perform_registration(MRI_SCANS_SEG, MRI_SCANS, MRI_SCANS_SEG_RESAMPLED)

    # print('\nCombining MRI_Seg Volume and MRI_Vol Header')
    # perform_overlay()

    # print('\nDice(MRI_Seg, PhotoReconSAMSEG) in PhotoReconSAMSEG space')
    # perform_registration(MRI_SCANS_SEG_REG_RES, HARD_RECON_SAMSEG,
    #                      MRI_SYNTHSEG_IN_SAMSEG_SPACE)
    # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE, HARD_RECON_SAMSEG,
    #                'mri_synth_vs_hard_samseg_in_sam_space.json')
    # hard_recon_box_plot('mri_synth_vs_hard_samseg_in_sam_space.json',
    #                     'mri_synth_vs_hard_samseg_in_sam_space.pdf')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(HARD_RECON_SYNTHSEG, HARD_RECON_SAMSEG,
    #                      HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE)
    # calculate_dice(MRI_SYNTHSEG_IN_SAMSEG_SPACE,
    #                HARD_RECON_SYNTHSEG_IN_SAMSEG_SPACE,
    #                'mri_synth_vs_hard_synth_in_sam_space.json')
    # hard_recon_box_plot('mri_synth_vs_hard_synth_in_sam_space.json',
    #                     'mri_synth_vs_hard_synth_in_sam_space.pdf')

    # print('\nDice(MRI_Seg, PhotoReconSYNTHSEG) in PhotoReconSAMSEG space')
    # perform_registration(HARD_RECON_SYNTHSEG, MRI_SCANS_SEG_REG_RES,
    #                      HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE)
    # calculate_dice(MRI_SCANS_SEG_REG_RES, HARD_RECON_SYNTHSEG_IN_MRISEG_SPACE,
    #                'mri_synth_vs_hard_synth_in_mri_space.json')
    # hard_recon_box_plot('mri_synth_vs_hard_synth_in_mri_space.json',
    #                     'mri_synth_vs_hard_synth_in_mri_space.pdf')

    #### Extract SYNTHSEG Volumes
    mri_synthseg_vols = extract_synthseg_vols(mri_synthseg_vols_file, 'mri')
    hard_synthseg_vols = extract_synthseg_vols(hard_synthseg_vols_file, 'hard')
    soft_synthseg_vols = extract_synthseg_vols(soft_synthseg_vols_file, 'soft')

    hard_samseg_vols = extract_samseg_volumes(HARD_SAMSEG_STATS, 'hard')
    soft_samseg_vols = extract_samseg_volumes(SOFT_SAMSEG_STATS, 'soft')
            
    print_correlation_pairs(mri_synthseg_vols, hard_samseg_vols, hard_synthseg_vols, flag='HARD')
    print_correlation_pairs(mri_synthseg_vols, soft_samseg_vols, soft_synthseg_vols, flag='SOFT')

    ### Work for Soft segmentations

    # get_volume_correlations('synthseg')

    # ## START for SAMSEG

    # perform_registration(HARD_RECON_SAMSEG, MRI_SCANS_SEG_REG,
    #                      HARD_RECON_SAMSEG_RESAMPLED)

    # calculate_dice(MRI_SCANS_SEG_RESAMPLED,
    #                HARD_RECON_SAMSEG_RESAMPLED, 'hard_samseg_dice.json')

    # hard_recon_box_plot('hard_samseg_dice.json',
    #                     'hard_samseg_dice_boxplot.pdf')
    # get_volume_correlations('samseg')
