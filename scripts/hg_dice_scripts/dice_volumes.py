import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

from dice_config import *
from dice_stats import calculate_pval


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
        print('{:^15}{:^15}{:^15}{:^15}'.format('label', 'SAMSEG', 'SYNTHSEG',
                                                'p-value'))
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


def write_correlations_to_file():
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

    print_correlation_pairs(mri_synthseg_vols,
                            soft_samseg_vols,
                            soft_synthseg_vols,
                            flag='SOFT')


if __name__ == '__main__':
    write_correlations_to_file()
