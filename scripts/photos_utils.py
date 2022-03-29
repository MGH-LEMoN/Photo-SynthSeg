import argparse
import glob
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import csv
import glob
import os

import nibabel as nib
import numpy as np
import tensorflow as tf
from PIL import Image

import scripts.photos_config as config
from ext.lab2im import utils
from ext.lab2im import utils as l2m_utils
from ext.neuron import utils as nrn_utils

'''
1. git clone https://github.com/bbillot/SynthSeg
2. export PYTHONPATH=${PWD}/SynthSeg
3. cd SynthSeg 
'''

NUM_COPIES = 50

VIVE_DIR = '/cluster/vive/UW_photo_recon/Photo_data/'
PRJCT_DIR = '/space/calico/1/users/Harsha/SynthSeg'
DATA_DIR = os.path.join(PRJCT_DIR, 'data')
MODEL_DIR = os.path.join(PRJCT_DIR, 'models')
RESULTS_DIR = os.path.join(PRJCT_DIR, 'results')


def collect_images_into_pdf(target_dir_str):
    """[summary]

    Args:
        target_dir_str ([str]): string relative to RESULTS_DIR
    """
    target_dir = os.path.join(RESULTS_DIR, target_dir_str)
    out_file = os.path.basename(target_dir) + '.pdf'
    out_file = os.path.join(target_dir, out_file)

    model_dirs = sorted(glob.glob(os.path.join(target_dir, '*')))

    pdf_img_list = []
    for model_dir in model_dirs:
        image_dir = os.path.join(model_dir, 'images')
        images = sorted(glob.glob(os.path.join(image_dir, '*')))

        for image in images:
            img = Image.open(image)
            img = img.convert('RGB')
            pdf_img_list.append(img)

    pdf_img_list[0].save(out_file,
                         save_all=True,
                         append_images=pdf_img_list[1:])


def check_size_of_labels():
    LABEL_MAPS_DIR = os.path.join(
        DATA_DIR,
        'SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem')
    # For now let's assume a batch size of 1.

    # Let's say the nonlin_shape_factor is as follows for "one sample"
    nonlin_shape_factor = [0.0625, 0.25, 0.0625]

    file_list = sorted(os.listdir(LABEL_MAPS_DIR))

    for file in file_list:
        # Load label map
        in_vol = l2m_utils.load_volume(os.path.join(LABEL_MAPS_DIR, file))

        print(in_vol.shape)


def nonlinear_deformation():
    LABEL_MAPS_DIR = os.path.join(DATA_DIR, 'training_label_maps')
    # For now let's assume a batch size of 1.

    # Let's say the nonlin_shape_factor is as follows for "one sample"
    nonlin_shape_factor = [0.0625, 0.25, 0.0625]

    # Load label map
    in_vol = l2m_utils.load_volume(
        os.path.join(LABEL_MAPS_DIR, 'training_seg_01.nii.gz'))

    in_vol = tf.convert_to_tensor(in_vol)

    print(f'In Volume Shape is: {in_vol.shape}')

    # ndgrid for in_vol
    in_vol_grid = nrn_utils.volshape_to_ndgrid(in_vol.shape)

    #FIXME: Please help!!!
    # From your recording, I am unsure about how to go from [256, 256, 256] to [16, 64, 16]
    # i_prime = i * 0.0625
    # j_prime = j * 0.25

    #-- BEGIN (your code)
    i = in_vol_grid[0]
    j = in_vol_grid[1]
    k = in_vol_grid[2]

    import keras.backend as K
    i_prime = nonlin_shape_factor[0] * K.cast(i, "float")
    j_prime = nonlin_shape_factor[1] * K.cast(j, "float")
    k_prime = nonlin_shape_factor[2] * K.cast(k, "float")

    shrunk_vol_grid = (i_prime, j_prime, k_prime)

    #-- END (yourcode)

    # For the given input shape and nonlin_shape_factor the
    # small shape becomes [256, 256, 256] * [0.0625, 0.25, 0.0625] = [16, 64, 16]
    small_shape = l2m_utils.get_resample_shape(in_vol.shape,
                                               nonlin_shape_factor)
    print(f'Small Volume Shape is: {small_shape}')
    small_vol = tf.random.normal(small_shape)

    # Interpolating small_vol to in_shape
    # out_vol = nrn_utils.interpn(small_vol, in_vol_grid)
    out_vol = nrn_utils.interpn(small_vol, shrunk_vol_grid)
    print(f'Out Volume Shape is: {out_vol.shape}')

    # Do Vector Integration of out_vol [..., 0] is to select out that channel
    int_out_vol = nrn_utils.integrate_vec(out_vol[..., 0], nb_steps=4)

    print(f'Integrated Out Volume Shape is: {int_out_vol.shape}')

    # Perform affine and elastic transformation here (This part is untouched anyway)


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


def write_config(dictionary, file_name=None):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    file_name = 'config.json' if file_name is None else file_name

    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    utils.mkdir(dictionary['model_dir'])

    config_file = os.path.join(dictionary['model_dir'], file_name)

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


def get_recon_shapes():
    folder_list = sorted(glob.glob(os.path.join(VIVE_DIR, '*')))

    for folder in folder_list:
        print(os.path.basename(folder), end='  ')
        file1 = os.path.join(folder, 'ref_mask', 'photo_recon.mgz')
        file2 = os.path.join(folder, 'ref_mask', 'photo_recon1.mgz')
        file3 = os.path.join(folder, 'ref_mask',
                             'manual_labels_merged.elastix.mgz')

        if os.path.exists(file1):
            shape1 = nib.load(file1).dataobj.shape
            print(shape1, end='  ')
        if os.path.exists(file2):
            shape2 = nib.load(file2).dataobj.shape
            print(shape2, end='  ')
        if os.path.exists(file3):
            shape3 = nib.load(file3).dataobj.shape
            print(shape3, end='  ')
        if shape1[1] == shape2[1]:
            print('Equal')
        else:
            print('Not Equal')


def model_dice_map():
    model_dirs = sorted(glob.glob(os.path.join(MODEL_DIR, '*')))
    model_dirs = [
        model_dir for model_dir in model_dirs if os.path.isdir(model_dir)
        and os.path.basename(model_dir).startswith(('VS'))
    ]

    dice_list = []
    for model_dir in model_dirs:
        last_dice_file = sorted(glob.glob(os.path.join(model_dir,
                                                       'dice_*')))[-1]
        dice_idx = os.path.basename(last_dice_file)[5:8]
        dice_list.append([os.path.basename(model_dir), dice_idx, ''])

    with open(os.path.join(PRJCT_DIR, 'dice_ids.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dice_list)


if __name__ == '__main__':

    # make_data_copy()
    # collect_images_into_pdf('20220201/new-recons')
    # check_size_of_labels()
    # nonlinear_deformation()
    model_dice_map()
