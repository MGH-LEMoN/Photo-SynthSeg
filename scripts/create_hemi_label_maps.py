import copy
import glob
import os

import numpy as np
from ext.lab2im import edit_volumes, utils
from scipy import ndimage

from scripts.fs_lut import fs_lut
from scripts.photos_config import *

LUT, REVERSE_LUT = fs_lut()

rh_exceptions = [163, 164]  # Always right
lh_exceptions = [136, 137]  # Always left


def cropLabelVol(V, margin=10, threshold=0):
    # Crop label volume

    # Make sure it's 3D
    margin = np.array(margin)
    if len(margin.shape) < 2:
        margin = [margin, margin, margin]

    if len(V.shape) < 2:
        V = V[..., np.newaxis]
    if len(V.shape) < 3:
        V = V[..., np.newaxis]

    # Now
    idx = np.where(V > threshold)
    i1 = np.max([0, np.min(idx[0]) - margin[0]]).astype("int")
    j1 = np.max([0, np.min(idx[1]) - margin[1]]).astype("int")
    k1 = np.max([0, np.min(idx[2]) - margin[2]]).astype("int")
    i2 = np.min([V.shape[0], np.max(idx[0]) + margin[0] + 1]).astype("int")
    j2 = np.min([V.shape[1], np.max(idx[1]) + margin[1] + 1]).astype("int")
    k2 = np.min([V.shape[2], np.max(idx[2]) + margin[2] + 1]).astype("int")

    cropping = [i1, j1, k1, i2, j2, k2]
    cropped = V[i1:i2, j1:j2, k1:k2]

    return cropped, cropping


def get_label_maps(labels_dir):
    """[summary]

    Args:
        labels_dir ([type]): [description]

    Returns:
        [type]: [description]
    """
    return sorted(glob.glob(os.path.join(labels_dir, "*nii.gz")))


def make_left_hemis(im, PreferLeft):
    """[summary]

    Args:
        im ([type]): [description]
        PreferLeft ([type]): [description]

    Returns:
        [type]: [description]
    """
    im_left = copy.deepcopy(im.astype("int"))

    im_left[PreferLeft == 0] = 0
    im_left, _ = cropLabelVol(im_left)

    return im_left


def make_right_hemis(im, PreferLeft, labels):
    """[summary]

    Args:
        im ([type]): [description]
        PreferLeft ([type]): [description]
        labels ([type]): [description]

    Returns:
        [type]: [description]
    """
    Lright = copy.deepcopy(im.astype("int"))

    Lright[PreferLeft == 1] = 0
    Lright, _ = cropLabelVol(Lright)
    Lright_mapped = np.zeros(Lright.shape)

    for r_label, l_label in zip(labels['right'], labels['left']):
        Lright_mapped[Lright == r_label] = l_label

    for n_label in labels['neutral']:
        Lright_mapped[Lright == n_label] = n_label

    Lright = np.fliplr(Lright_mapped)

    return Lright


def create_hemi_dirs(dir_path):
    """[summary]

    Args:
        dir_path ([type]): [description]
    """
    lh_dir = dir_path + "_lh"
    r2l_dir = dir_path + "_rh2lh"

    os.makedirs(lh_dir, exist_ok=True)
    os.makedirs(r2l_dir, exist_ok=True)

    return


def get_hemi_dir_name(dir_path, side=None):
    """[summary]

    Args:
        dir_path ([type]): [description]
        side ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    if not os.path.isdir(dir_path):
        dir_path = os.path.dirname(dir_path)

    suffix = {"right": "_rh2lh", "left": "_lh"}
    dir_path = dir_path + suffix.get(side, None)

    return dir_path


def get_file_name(file_path):
    """[summary]

    Args:
        file_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    _, file_name = os.path.split(file_path)
    file_name, ext = file_name.split(os.extsep, 1)

    return file_name, ext


def save_hemi_at_location(im, aff, hdr, file_name, side=None):

    hemi_dir = get_hemi_dir_name(file_name, side)
    file_name, ext = get_file_name(file_name)

    file_suffix = {"right": "_rh2lh.", "left": "_lh."}
    lh_file_name = os.path.join(hemi_dir, file_name + file_suffix[side] + ext)

    # Save Volume
    utils.save_volume(im, aff, hdr, lh_file_name)

    return


def get_list_labels(label_list=None,
                    labels_dir=None,
                    save_label_list=None,
                    FS_sort=False):
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    :param label_list: (optional) already computed label_list. Can be a sequence, a 1d numpy array, or the path to
    a numpy 1d array.
    :param labels_dir: (optional) if path_label_list is None, the label list is computed by reading all the label maps
    in the given folder. Can also be the path to a single label map.
    :param save_label_list: (optional) path where to save the label list.
    :param FS_sort: (optional) whether to sort label values according to the FreeSurfer classification.
    If true, the label values will be ordered as follows: neutral labels first (i.e. non-sided), left-side labels,
    and right-side labels. If FS_sort is True, this function also returns the number of neutral labels in label_list.
    :return: the label list (numpy 1d array), and the number of neutral (i.e. non-sided) labels if FS_sort is True.
    If one side of the brain is not represented at all in label_list, all labels are considered as neutral, and
    n_neutral_labels = len(label_list).
    """

    # load label list if previously computed
    if label_list is not None:
        label_list = np.array(
            utils.reformat_to_list(label_list, load_as_numpy=True,
                                   dtype='int'))

    # compute label list from all label files
    elif labels_dir is not None:
        # print('Compiling list of unique labels')
        # go through all labels files and compute unique list of labels
        labels_paths = utils.list_images_in_folder(labels_dir)
        label_list = np.empty(0)
        # loop_info = utils.LoopInfo(len(labels_paths),
        #                            10,
        #                            'processing',
        #                            print_time=True)
        for _, path in enumerate(labels_paths):
            # loop_info.update(lab_idx)
            y = utils.load_volume(path, dtype='int32')
            y_unique = np.unique(y)
            label_list = np.unique(np.concatenate(
                (label_list, y_unique))).astype('int')

    else:
        raise Exception(
            'either label_list, path_label_list or labels_dir should be provided'
        )

    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [
            0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 165, 200, 201, 202, 203, 204, 205,
            206, 207, 208, 209, 210, 251, 252, 253, 254, 255, 258, 259, 260,
            331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 502, 506, 507,
            508, 509, 511, 512, 514, 515, 516, 517, 530, 531, 532, 533, 534,
            535, 536, 537
        ]
        neutral = list(set(label_list) & set(neutral_FS_labels))
        left = list(label_list[((label_list > 0) & (label_list < 14))
                               | ((label_list > 16) & (label_list < 21))
                               | ((label_list > 24) & (label_list < 40))
                               | ((label_list > 135) & (label_list < 138))
                               | ((label_list > 20100) &
                                  (label_list < 20110))])
        right = list(label_list[((label_list > 39) & (label_list < 72))
                                | ((label_list > 162) & (label_list < 165))
                                | ((label_list > 20000) &
                                   (label_list < 20010))])

        missing_labels = set.difference(set(label_list),
                                        set(neutral + left + right))

        if missing_labels:
            raise Exception("labels {} not in our current FS classification, "
                            "please update get_list_labels in utils.py".format(
                                missing_labels))
        label_list = np.concatenate(
            [sorted(neutral), sorted(left),
             sorted(right)])
        if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) &
                                                   (len(right) == 0)):
            n_neutral_labels = len(neutral)
        else:
            n_neutral_labels = len(label_list)

    # save labels if specified
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))

    if FS_sort:
        return np.int32(label_list), n_neutral_labels
    else:
        return np.int32(label_list), None


def return_labels_from_map(label_map):
    """[summary]

    Args:
        label_map ([type]): [description]

    Returns:
        [type]: [description]
    """
    label_list, n_neutral_labels = get_list_labels(None,
                                                   label_map,
                                                   FS_sort=True)

    # label_list is of the form [neutral, left, right]
    neutral = label_list[:n_neutral_labels]
    mid_point = n_neutral_labels + (len(label_list) - n_neutral_labels) // 2

    # Left and Right Labels
    left = label_list[n_neutral_labels:mid_point]
    right = label_list[mid_point:]

    labels = dict(neutral=neutral, left=left, right=right)

    return labels


def compute_binary_mask(img, labels, side=None):
    """[summary]

    Args:
        img ([type]): [description]
        label_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    mask = np.zeros(img.shape)
    mask[np.isin(img, labels[side])] = 1

    return mask


def compute_decision_mask(left_mask, right_mask):
    """[summary]

    Args:
        left_mask ([type]): [description]
        right_mask ([type]): [description]

    Returns:
        [type]: [description]
    """
    Dleft = ndimage.distance_transform_edt(left_mask == 0)
    Dright = ndimage.distance_transform_edt(right_mask == 0)

    PreferLeft = Dleft < Dright

    return PreferLeft


def main():
    create_hemi_dirs(LABEL_MAPS_DIR)
    label_maps = get_label_maps(LABEL_MAPS_DIR)

    total_label_maps = len(label_maps)

    for idx, label_map in enumerate(label_maps, 1):
        print(
            f'{idx:04d} of {total_label_maps} - {get_file_name(label_map)[0]}')

        # Return Neutral, Left and Right Labels
        labels = return_labels_from_map(label_map)

        # Load Label Map
        im, aff, hdr = utils.load_volume(label_map, im_only=False)

        # Compute binary masks for each side
        left_mask = compute_binary_mask(im, labels, 'left')
        right_mask = compute_binary_mask(im, labels, 'right')

        # Compute Distance maps and decision mask
        PreferLeft = compute_decision_mask(left_mask, right_mask)

        # Create Segmentation for left side, crop, write to disk
        im_left = make_left_hemis(im, PreferLeft)
        im_right = make_right_hemis(im, PreferLeft, labels)

        save_hemi_at_location(im_left, aff, hdr, label_map, "left")
        save_hemi_at_location(im_right, aff, hdr, label_map, "right")


def main1():
    npy_files = glob.glob(os.path.join(PARAM_FILES_DIR, '*.npy'))

    lh_param_files_dir = PARAM_FILES_DIR + '_lh'
    os.makedirs(lh_param_files_dir, exist_ok=True)

    for npy_file in npy_files:
        file_name_with_ext = os.path.basename(npy_file)
        file_name, ext = os.path.splitext(file_name_with_ext)

        np_array = np.load(npy_file)
        np_array = np_array[:21]  # TODO: Hardcoded

        new_file_name = file_name + '_lh' + ext
        new_file_name = os.path.join(lh_param_files_dir, new_file_name)

        np.save(new_file_name, np_array)


def pad_hemispheres():
    labels_dir = os.path.join(LABEL_MAPS_DIR + '_*')
    file_list = sorted(
        glob.glob(os.path.join(LABEL_MAPS_DIR + '_*', '*.nii.gz')))

    im_shapes = []
    for file in file_list[:15]:
        im_shapes.append(utils.load_volume(file).shape)

    im_shapes = np.vstack(im_shapes)
    max_shape = np.max(im_shapes, axis=0)

    # compute the closest number (higher) to max_shape that is divisible by 32
    max_shape = max_shape + (32 - max_shape % 32)

    edit_volumes.pad_images_in_dir()


if __name__ == "__main__":
    # main()
    # main1()
    pad_hemispheres()
