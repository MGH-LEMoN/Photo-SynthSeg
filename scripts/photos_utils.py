import glob
import json
import os

import nibabel as nib
import numpy as np
from ext.lab2im import utils

import scripts.photos_config as config

NUM_COPIES = 50


def create_destination_name(source, idx):
    """Create destination file names given source

    Args:
        source (string): source file name 
        idx (int): integer to append (while creating a copy)

    Returns:
        string: destination file name
    """
    base_dir, file_name_with_ext = os.path.split(source)
    file_name, ext = file_name_with_ext.split(os.extsep, 1)

    file_name = '_'.join([file_name, 'copy', f'{idx:02}'])
    new_file_name = os.extsep.join([file_name, ext])

    return os.path.join(base_dir, new_file_name)


def create_symlink(source, num_copies=10):
    """Create num_copies number of symlinks for source

    Args:
        source (string): file to be copied
        num_copies (int, optional): number of copies to create. Defaults to 10.
    """
    list(
        map(lambda x: os.symlink(source, create_destination_name(source, x)),
            range(1, num_copies)))

    return


def make_data_copy():
    subject_files = sorted(
        glob.glob(os.path.join(config.LABEL_MAPS_DIR, 'subject*.nii.gz')))

    for file in subject_files:
        print(f'Creating copies of {os.path.split(file)[-1]}')
        create_symlink(file, NUM_COPIES)

    return


def write_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    utils.mkdir(dictionary['model_dir'])

    config_file = os.path.join(dictionary['model_dir'], 'config.json')

    with open(config_file, "w") as outfile:
        outfile.write(json_object)


#TODO: Improve this function and add doc strings
def find_label_differences():
    output_file = 'label_comparison'
    file_list = [
        file for file in sorted(
            glob.glob(os.path.join(config.LABEL_MAPS_DIR, '*.nii.gz')))
        if 'copy' not in file
    ]

    with open('label_comparison', 'a+') as f:
        generation_labels = np.load(config.GENERATION_LABELS)
        print(f'generation_labels\n{generation_labels}', file=f)

        segmentation_labels = np.load(config.SEGMENTATION_LABELS)
        print(f'segmentation_labels\n{segmentation_labels}', file=f)

        generation_classes = np.load(config.GENERATION_CLASSES)
        print(f'generation_classes\n{generation_classes}', file=f)

    with open('label_comparison', 'a+') as f:
        for file in file_list:
            _, subject = os.path.split(file)

            img = nib.load(file)
            img_data = img.get_fdata()

            uniq_labels = np.unique(img_data)
            extra_labels = set(uniq_labels) - set(generation_labels)
            extra_labels1 = set(generation_labels) - set(uniq_labels)

            print(f'{subject:60s}\t{extra_labels}\t{extra_labels1}', file=f)

    uniq = []
    for file in file_list:
        _, subject = os.path.split(file)

        img = nib.load(file)
        img_data = img.get_fdata()

        uniq.extend(np.unique(img_data))

    final_uniq = np.unique(uniq)

    extra_labels = set(final_uniq) - set(generation_labels)
    extra_labels1 = set(generation_labels) - set(final_uniq)

    print(f'{extra_labels}\t{extra_labels1}')


if __name__ == '__main__':
    make_data_copy()
