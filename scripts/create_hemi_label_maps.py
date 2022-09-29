import copy
import glob
import multiprocessing
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from ext.lab2im import edit_volumes, utils
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

    for r_label, l_label in zip(labels["right"], labels["left"]):
        Lright_mapped[Lright == r_label] = l_label

    for n_label in labels["neutral"]:
        Lright_mapped[Lright == n_label] = n_label

    Lright = np.flip(Lright_mapped, 0)

    return Lright


def create_hemi_dirs(dir_path):
    """[summary]

    Args:
        dir_path ([type]): [description]
    """
    lh_dir = dir_path + "_lh"
    # r2l_dir = dir_path + "_rh2lh"

    os.makedirs(lh_dir, exist_ok=True)
    # os.makedirs(r2l_dir, exist_ok=True)

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

        # suffix = {"right": "_rh2lh", "left": "_lh"}
    suffix = {"right": "_lh", "left": "_lh"}
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


def return_labels_from_map(label_map):
    """[summary]

    Args:
        label_map ([type]): [description]

    Returns:
        [type]: [description]
    """
    label_list, n_neutral_labels = utils.get_list_labels(
        None, label_map, FS_sort=True
    )

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


def main_mp():
    """_summary_
    Args:
            skip (_type_): _description_
            jitter (_type_): _description_
    """
    create_hemi_dirs(LABEL_MAPS_DIR)
    label_maps = get_label_maps(LABEL_MAPS_DIR)

    print(f"Total Label Maps: {len(label_maps)}")

    gettrace = getattr(sys, "gettrace", None)
    n_procs = 1 if gettrace() else multiprocessing.cpu_count()

    with Pool(processes=n_procs) as pool:
        pool.starmap(
            generate_hemisphere_masks,
            [*enumerate(label_maps, 1)],
        )


def generate_hemisphere_masks(idx, label_map):
    print(f"{idx:04d} - {get_file_name(label_map)[0]}")

    # Return Neutral, Left and Right Labels
    labels = return_labels_from_map(label_map)

    # Load Label Map
    im, aff, hdr = utils.load_volume(label_map, im_only=False)

    # Compute binary masks for each side
    left_mask = compute_binary_mask(im, labels, "left")
    right_mask = compute_binary_mask(im, labels, "right")

    # Compute Distance maps and decision mask
    PreferLeft = compute_decision_mask(left_mask, right_mask)

    # Create Segmentation for left side, crop, write to disk
    im_left = make_left_hemis(im, PreferLeft)
    im_right = make_right_hemis(im, PreferLeft, labels)

    save_hemi_at_location(im_left, aff, hdr, label_map, "left")
    save_hemi_at_location(im_right, aff, hdr, label_map, "right")


def main():
    create_hemi_dirs(LABEL_MAPS_DIR)
    label_maps = get_label_maps(LABEL_MAPS_DIR)

    print(f"Total Label Maps: {len(label_maps)}")

    for idx, label_map in enumerate(label_maps, 1):
        generate_hemisphere_masks(idx, label_map)


def main1():
    npy_files = glob.glob(os.path.join(PARAM_FILES_DIR, "*.npy"))

    lh_param_files_dir = PARAM_FILES_DIR + "_lh"
    os.makedirs(lh_param_files_dir, exist_ok=True)

    for npy_file in npy_files:
        file_name_with_ext = os.path.basename(npy_file)
        file_name, ext = os.path.splitext(file_name_with_ext)

        np_array = np.load(npy_file)
        np_array = np_array[:21]  # TODO: Hardcoded

        new_file_name = file_name + "_lh" + ext
        new_file_name = os.path.join(lh_param_files_dir, new_file_name)

        np.save(new_file_name, np_array)


if __name__ == "__main__":
    main_mp()
    # main1()
    # pad_hemispheres()
