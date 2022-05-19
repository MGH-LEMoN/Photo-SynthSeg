import glob
import os
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform
from tqdm import tqdm

from ext.hg_utils import zoom
from ext.lab2im import utils

np.random.seed(0)

# TODO: fix this function
# def get_min_max_idx(t2_file):
#     # 7. Open the T2
#     t2_vol = utils.load_volume(t2_file)

#     # scaling the entire volume
#     t2_vol = 255 * t2_vol / np.max(t2_vol)

#     N = np.zeros(len(spacings))
#     for n in range(len(spacings)):
#         N(n) = minimum used slice index for spacing (skip) n
#         s1= np.max(N)

#     for n in range(len(spacings)):
#         N(n) = maximum used slice index for spacing (skip) n
#         s1= np.min(N)


def get_nonzero_slice_ids(t2_vol):
    """find all non-zero slices"""
    slice_sum = np.sum(t2_vol, axis=(0, 1))
    return np.where(slice_sum > 0)[0]


def slice_ids_method1(args, t2_vol):
    """current method of selecting slices
    Example:
    slice_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    slices:    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  0,  0,  0]
    selected:  [0, 0, 0, 0, 1, 0, 1, 0, 1, 0,  1,  0,  0,  0] (skip = 2)
    selected:  [0, 0, 0, 1, 0, 0, 1, 0, 0, 1,  0,  0,  0,  0] (skip = 3)
    """
    non_zero_slice_ids = get_nonzero_slice_ids(t2_vol)

    first_nz_slice = non_zero_slice_ids[0]
    slice_ids_of_interest = np.where(non_zero_slice_ids % args["SKIP"] == 0)
    slice_ids_of_interest = slice_ids_of_interest[0] + first_nz_slice

    return slice_ids_of_interest


def slice_ids_method2(args, t2_vol):
    """method: skipping from start
    Example:
    slice_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    slices:    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  0,  0,  0]
    selected:  [0, 0, 0, 1, 0, 1, 0, 1, 0, 1,  0,  0,  0,  0] (skip = 2)
    selected:  [0, 0, 0, 1, 0, 0, 1, 0, 0, 1,  0,  0,  0,  0] (skip = 3)
    """
    # another method: skip 5 non zero slices on either end (good)
    non_zero_slice_ids = get_nonzero_slice_ids(t2_vol)

    first_nz_slice = non_zero_slice_ids[0]  # + 5
    last_nz_slice = non_zero_slice_ids[-1]  # - 5

    slice_ids_of_interest = np.arange(first_nz_slice, last_nz_slice + 1,
                                      args["SKIP"])
    return slice_ids_of_interest


def process_t1(args, t1_file, t1_name):

    # 1. Sample 3 rotations about the 3 axes, e.g., between -30 and 30 degrees.
    rotation = np.random.randint(-30, 31, 3)

    # 2. Sample 3 translations along the 3 axes, e.g., between 20 and 20 mm
    translation = np.random.randint(-20, 21, 3)

    # 3. Build a rigid 3D rotation + translation (4x4) matrix using the rotations and shifts
    t1_rigid_mat = utils.create_affine_transformation_matrix(
        3,
        scaling=None,
        rotation=rotation,
        shearing=None,
        translation=translation)

    t1_rigid_out = os.path.join(args["OUT_DIR"], t1_name,
                                f"{t1_name}.rigid.npy")
    np.save(t1_rigid_out, t1_rigid_mat)

    # 4. Open the T1, and premultiply the affine matrix of the header (“vox2ras”) by the matrix from 3.
    volume, aff, hdr = utils.load_volume(t1_file, im_only=False)
    new_aff = np.matmul(t1_rigid_mat, aff)
    hdr.set_sform(new_aff)

    # 5. Binarize the T1 volume by thresholding at 0 and save it with the new header, and call it “mri.mask.mgz”
    t1_out_path = os.path.join(args["OUT_DIR"], t1_name,
                               f"{t1_name}.mri.mask.mgz")
    utils.save_volume(volume > 0, new_aff, hdr, t1_out_path)


def process_t2(args, t2_file, t2_name, jitter=0):

    AFFINE_DIR = os.path.join(args["OUT_DIR"], t2_name, "slice_affines")
    PHOTO_DIR = os.path.join(args["OUT_DIR"], t2_name, "photo_dir")

    # 6. Create a directory “photo_dir"
    os.makedirs(AFFINE_DIR, exist_ok=True)
    os.makedirs(PHOTO_DIR, exist_ok=True)

    # 7. Open the T2 (and scale)
    t2_vol = utils.load_volume(t2_file)
    t2_vol = 255 * t2_vol / np.max(t2_vol)

    slice_ids = slice_ids_method1(args, t2_vol)  # see method 2

    for c, z in enumerate(slice_ids, 1):
        curr_slice = t2_vol[..., z]

        # FIXME: cleanup this function
        if jitter:
            i = 0
            all_idx = []
            while True:
                if i == 15:
                    print(all_idx)
                    print(f"Possiby Bad Case: {t2_name}")
                    break

                # rand_idx = np.random.choice(np.arange(1, jitter + 1))
                # rand_idx = rand_idx if np.random.random(
                # ) < 0.5 else -rand_idx
                rand_idx = random.randrange(-jitter, jitter)

                curr_slice = t2_vol[..., z + rand_idx]

                i += 1
                all_idx.append(rand_idx)

                if np.sum(curr_slice):
                    break

        curr_slice = np.pad(np.rot90(curr_slice), 25)

        # Sample a rotation eg between -20 and 20 degrees
        rotation = np.random.randint(-20, 21, 1)

        # Sample 2 translations along the 2 axes, eg, between -0.5 and 0.5 pixels
        translation = np.random.uniform(-0.5, 0.5, 2)

        # Sample 2 small shears about the 2 axes (eg between -0.1 and 0.1)
        shearing = np.random.uniform(-0.1, 0.1, 2)

        # Build a 2D (3x3) matrix with the rotation, translations, and shears
        translation_mat_1 = np.array([
            [1, 0, -0.5 * curr_slice.shape[0]],
            [0, 1, -0.5 * curr_slice.shape[1]],
            [0, 0, 1],
        ]).astype(float)
        translation_mat_2 = np.array([
            [1, 0, 0.5 * curr_slice.shape[0]],
            [0, 1, 0.5 * curr_slice.shape[1]],
            [0, 0, 1],
        ]).astype(float)
        aff_mat = utils.create_affine_transformation_matrix(
            2,
            scaling=None,
            rotation=rotation,
            shearing=shearing,
            translation=translation,
        )
        slice_aff_mat = np.matmul(translation_mat_2,
                                  np.matmul(aff_mat, translation_mat_1))

        # Save this matrix somewhere for evaluation later on eg as a numpy array
        slice_aff_out = os.path.join(AFFINE_DIR,
                                     f"{t2_name}.slice.{c:03d}.npy")
        np.save(slice_aff_out, slice_aff_mat)

        # Use this matrix to deform the slice
        deformed_slice = affine_transform(curr_slice,
                                          slice_aff_mat,
                                          mode="constant",
                                          order=1)

        # Threshold the deformed slice at zero to get a mask (1 inside, 0 outside)
        mask = deformed_slice > 0
        # write it as photo_dir/image.[c].npy
        # (format the number c with 2 digits so they are in order when listed)
        out_file_name = os.path.join(PHOTO_DIR, f"{t2_name}.image.{c:03d}.npy")
        np.save(out_file_name, mask)

        # add illumination field to the slice
        # Sample a random zero-mean gaussian tensor of size (5x5) and multiply by a small standard deviation (eg 0.1)
        small_vol = 0.1 * np.random.normal(size=(5, 5))

        # Upscale the tensor to the size of the slice
        # edit_volumes.resample_volume(small)
        factors = np.divide(mask.shape, small_vol.shape)

        # Take the pixel-wise exponential of the upsampled tensor to get an illumination field
        bias_result = zoom.scipy_zoom(small_vol, factors, mask.shape)

        # Multiply the deformed slice by the illumination field
        corrupted_image = np.multiply(deformed_slice, bias_result)

        # Write the corrupted image to photo_dir/image.[c].tif
        img_out = os.path.join(PHOTO_DIR, f"{t2_name}.image.{c:03d}.png")
        corrupted_PIL = Image.fromarray(np.uint8(corrupted_image))
        corrupted_PIL.save(img_out, "PNG")


def sub_pipeline(args, t1_file, t2_file):
    # get file name
    t1_fname = os.path.split(t1_file)[1]
    t2_fname = os.path.split(t2_file)[1]

    # get subject ID
    t1_subject_name = t1_fname.split(".")[0]
    t2_subject_name = t2_fname.split(".")[0]

    assert t1_subject_name == t2_subject_name, "Incorrect Subject Name"

    # make output directory for subject
    out_subject_dir = os.path.join(args["OUT_DIR"], t1_subject_name)

    if os.path.isdir(out_subject_dir):
        pass
    else:
        os.makedirs(out_subject_dir, exist_ok=True)

    # create symlinks to source files (T1, T2)
    t1_dst = os.path.join(out_subject_dir, t1_fname)
    t2_dst = os.path.join(out_subject_dir, t2_fname)

    if not os.path.exists(t1_dst):
        os.symlink(t1_file, t1_dst)

    if not os.path.exists(t2_dst):
        os.symlink(t2_file, t2_dst)

    # work on T1 and T2 volumes
    process_t1(args, t1_file, t1_subject_name)
    process_t2(args, t2_file, t2_subject_name, jitter=3)


def pipeline(args):
    vol_pairs = get_t1_t2_pairs(args)

    for i in tqdm(range(len(vol_pairs)), position=0, leave=True):
        t1_file, t2_file = vol_pairs[i]
        sub_pipeline(args, t1_file, t2_file)


def make_args(skip):
    PRJCT_ID = "4harshaHCP"  # '4harshaHCP'
    PRJCT_DIR = "/space/calico/1/users/Harsha/SynthSeg"
    DATA_DIR = os.path.join(PRJCT_DIR, "data")
    RESULTS_DIR = os.path.join(PRJCT_DIR, "test-results/hcp-results")
    IN_DIR = os.path.join(DATA_DIR, PRJCT_ID)
    OUT_DIR = os.path.join(RESULTS_DIR, PRJCT_ID +
                           f"-skip-{skip:02d}-r3")  # '4harshaHCP_extracts'

    return dict(IN_DIR=IN_DIR, OUT_DIR=OUT_DIR, SKIP=skip)


def pipeline_wrapper(idx, args):
    sub_pipeline(args, *get_t1_t2_pairs(args)[idx])


def get_t1_t2_pairs(args):
    t1_files = sorted(glob.glob(os.path.join(args["IN_DIR"], "*T1.nii.gz")))
    t2_files = sorted(glob.glob(os.path.join(args["IN_DIR"], "*T2.nii.gz")))

    assert len(t1_files) == len(t2_files), "Subject Mismatch"

    return list(zip(t1_files, t2_files))


def pipeline_mp(args):
    file_count = len(get_t1_t2_pairs(args))

    with Pool() as pool:
        pool.map(
            partial(pipeline_wrapper, args=args),
            range(file_count),
        )


def main():
    for skip in range(2, 17, 2):
        print(f"Running Skip {skip:02d}")
        args = make_args(skip)
        pipeline(args)
        # pipeline_mp(args)


if __name__ == "__main__":
    main()
